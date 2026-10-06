import pandas as pd
import streamlit as st

from app_pages._cache import DEFAULT_VOCAB, smollm2, tokenizer
from booklm import tokenizer_lab as lab

sentence = st.session_state.sentence
tok = smollm2()

st.caption(
    "SmolLM2 ships its own byte-level BPE, trained on a huge general corpus. We can't change "
    "it: the model's embedding table has one row per entry, so the finetuning route must "
    "use it exactly as is."
)

meta = lab.load_meta("smollm2")
with st.container(horizontal=True):
    st.metric("Vocabulary", f"{tok.get_vocab_size():,}", border=True)
    st.metric("Train bytes/token", f"{meta['train']['bytes_per_token']:.2f}", border=True)
    st.metric("Val bytes/token", f"{meta['val']['bytes_per_token']:.2f}", border=True)
    st.metric(
        f"Embedding params (d={lab.D_MODEL})",
        f"{tok.get_vocab_size() * lab.D_MODEL / 1e6:.1f}M",
        border=True,
    )

pieces = lab.smollm2_pieces(sentence)
st.subheader(f"The sample sentence · {len(pieces)} tokens", divider="gray")
st.markdown(lab.chips_markdown([p["text"] for p in pieces]))
st.dataframe(
    pd.DataFrame(pieces),
    hide_index=True,
    column_config={
        "vocab_entry": st.column_config.TextColumn(
            "vocab entry", help="How the token is stored in SmolLM2's vocab file"
        ),
        "text": st.column_config.TextColumn("real text"),
    },
    alt="Each SmolLM2 token with its id, stored vocab entry and real text",
)

st.subheader("Why the vocab says `Ġthe`", divider="gray")
st.markdown(
    "GPT-2-style tokenizers store tokens in a file as text, but raw bytes like space or "
    "newline are invisible or unprintable. So each of the 256 bytes is mapped to a "
    "printable stand-in character before saving: space → `Ġ`, newline → `Ċ`, and so on. "
    "It is only a display trick; `Ġthe` really means ` the`. Our tokenizer does the same "
    "job by escaping instead (`·` in the lab, `repr` in `vocab.txt`)."
)

st.subheader("Try words from outside the Holmes canon", divider="gray")
words = st.text_input(
    "Words (comma separated)",
    value="Holmes, smartphone, Kubernetes, ciudad, 東京",
    key="smol_words",
)
ours = tokenizer(DEFAULT_VOCAB)
st.dataframe(
    pd.DataFrame(
        [
            {
                "word": w,
                "SmolLM2": " | ".join(p["text"] for p in lab.smollm2_pieces(" " + w)),
                "ours (8k)": " | ".join(t for _, t in lab.pieces(ours, " " + w)),
            }
            for w in (w.strip() for w in words.split(","))
            if w
        ]
    ),
    hide_index=True,
    alt="How each tokenizer splits words from outside the corpus",
)
