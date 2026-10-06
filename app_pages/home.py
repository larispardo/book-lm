from pathlib import Path

import streamlit as st

pages = st.session_state.pages

st.markdown(
    "A small language model trained on the Sherlock Holmes canon, built one stage at a time. "
    "Each stage adds a panel here so you can **see** what the code does, not just run it."
)

STAGES = [
    {
        "title": "1 · Tokenization",
        "status": ("Ready to explore", "green", ":material/check_circle:"),
        "blurb": "How text becomes the integers a model reads.",
        "questions": [
            "How does BPE pick its merges, one greedy step at a time?",
            "What changes between a 1k, 8k and 21k vocabulary?",
            "Why does SmolLM2 store ` the` as `Ġthe`?",
        ],
        "links": [
            ("own", "Watch BPE learn, step by step", ":material/play_arrow:"),
            ("smol", "Look inside SmolLM2's tokenizer", ":material/hub:"),
            ("compare", "Compare tokenizers on any text", ":material/compare_arrows:"),
        ],
    },
    {
        "title": "2 · Tiny GPT from scratch",
        "status": ("In progress", "orange", ":material/construction:"),
        "blurb": "Attention, blocks and a training loop on our 8k-token corpus.",
        "questions": ["What does the loss curve say?", "What do attention heads look at?"],
    },
    {
        "title": "3 · Decoding",
        "status": ("Planned", "gray", ":material/pending:"),
        "blurb": "Logits, temperature, top-k, top-p, min-p, live.",
        "questions": ["Why does temperature 2 produce nonsense?"],
    },
    {
        "title": "4 · Finetuning",
        "status": ("Planned", "gray", ":material/pending:"),
        "blurb": "LoRA by hand, then peft, on SmolLM2-135M.",
        "questions": ["Scratch vs pretrained vs finetuned: who writes like Doyle?"],
    },
]

for row in (STAGES[:2], STAGES[2:]):
    for col, stage in zip(st.columns(2), row, strict=False):
        with col, st.container(border=True, height="stretch"):
            label, color, icon = stage["status"]
            st.badge(label, color=color, icon=icon)
            st.subheader(stage["title"])
            st.caption(stage["blurb"])
            st.markdown("\n".join(f"- {q}" for q in stage["questions"]))
            for key, text, link_icon in stage.get("links", []):
                st.page_link(pages[key], label=text, icon=link_icon)

with st.expander("Full plan, decisions and experiments backlog", icon=":material/map:"):
    plan = Path("PLAN.md").read_text(encoding="utf-8")
    st.markdown(plan.split("\n", 1)[1])
