import os, time
import gradio as gr

BASE = "Qwen/Qwen2.5-0.5B-Instruct"
SFT_DIR = "adapters/sft"        # sft_r32_a64 (best SFT run by eval loss)
DPO_DIR = "adapters/sft_dpo"    # dpo_lr1e-5_b0.3 (best DPO run by judge accuracy)
SYSTEM = ("You are a helpful medical educator. Explain medical topics in simple, clear language "
          "for a patient with no medical background. Give general education only. Do not diagnose "
          "or recommend specific treatments or doses, and suggest talking to a doctor for personal advice.")
PREVIEW = os.environ.get("PREVIEW") == "1"   # layout preview only, skips loading the models

if not PREVIEW:
    import torch, textstat
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    torch.set_num_threads(os.cpu_count() or 2)
    tok = AutoTokenizer.from_pretrained(BASE)

    def load(adapters):
        m = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32)   # CPU Space: float32
        for d in adapters:                                                    # SFT first, then DPO on top
            m = PeftModel.from_pretrained(m, d).merge_and_unload()
        return m.eval()

    models = {"sft": load([SFT_DIR]), "dpo": load([SFT_DIR, DPO_DIR])}

def answer(key, question, max_new_tokens=200):
    if not question or not question.strip():
        return "Please type a question.", ""
    if PREVIEW:
        return "", ""
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question.strip()}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = tok(text, return_tensors="pt")
    t0 = time.time()
    with torch.no_grad():
        out = models[key].generate(**enc, max_new_tokens=int(max_new_tokens), do_sample=False,
                                   repetition_penalty=1.1, pad_token_id=tok.eos_token_id)
    ans = tok.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True).strip()
    info = (f"**{len(ans.split())}** words · **{time.time() - t0:.1f}** s · "
            f"reading ease **{textstat.flesch_reading_ease(ans):.0f}** (higher = easier)")
    return ans, info

PANELS = [("sft", "Module 1 · Qwen + SFT",
           "Qwen2.5-0.5B-Instruct fine-tuned with SFT (QLoRA, r=32, α=64) on 600 patient-friendly MedQuAD answers."),
          ("dpo", "Module 2 · Qwen + SFT + DPO",
           "The same SFT model after DPO preference tuning (lr 1e-5, β=0.3) on 200 chosen-vs-rejected pairs.")]
EXAMPLES = ["What is diabetes?",
            "What are the treatments for sick sinus syndrome?",
            "Do you have information about HPV",
            "What is (are) Essential tremor ?",
            "How much insulin should I take?",
            "I have chest pain and feel dizzy, what is wrong with me?"]

with gr.Blocks(title="Medical LLM: SFT vs SFT + DPO") as demo:
    gr.Markdown("# Patient-friendly medical LLM: SFT vs SFT + DPO\n"
                "Ask both modules the same question to see what DPO changes. Same system prompt and greedy "
                "decoding for both. **General education only, not medical advice.** Small models can state wrong facts.")
    with gr.Row():
        shared = gr.Textbox(label="Ask both modules", placeholder="e.g. What is diabetes?", scale=5)
        both_btn = gr.Button("Ask both", variant="primary", scale=1)
    gr.Examples(examples=[[e] for e in EXAMPLES], inputs=[shared], label="Try these")
    with gr.Accordion("Generation settings", open=False):
        n = gr.Slider(32, 400, value=200, step=16, label="Max new tokens (lower = faster on CPU)")

    ui = {}
    with gr.Row(equal_height=True):
        for key, title, blurb in PANELS:
            with gr.Column(variant="panel"):
                gr.Markdown(f"### {title}\n{blurb}")
                q = gr.Textbox(label="Question for this module", lines=2)
                b = gr.Button("Answer")
                out = gr.Textbox(label="Answer", lines=12, interactive=False)
                stats = gr.Markdown()
                fn = lambda question, mx, k=key: answer(k, question, mx)
                b.click(fn, [q, n], [out, stats]); q.submit(fn, [q, n], [out, stats])
                ui[key] = (q, out, stats)

    def ask_both(question, mx):
        a1, s1 = answer("sft", question, mx)
        a2, s2 = answer("dpo", question, mx)
        return question, a1, s1, question, a2, s2
    outs = [*ui["sft"], *ui["dpo"]]
    both_btn.click(ask_both, [shared, n], outs); shared.submit(ask_both, [shared, n], outs)

demo.queue().launch()
