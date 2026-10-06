import pandas as pd
import streamlit as st

from app_pages._cache import stats, tokenizer
from booklm import tokenizer_lab as lab

sentence = st.session_state.sentence
show_ids = st.toggle("Show token ids", key="compare_ids")

rows = []
for size in (4096, 8192, 16384):
    rows.append((f"Ours {size // 1024}k", lab.pieces(tokenizer(size), sentence), stats(size)))
smol = [(p["id"], p["text"]) for p in lab.smollm2_pieces(sentence)]

for label, pieces, _ in rows + [("SmolLM2 49k", smol, None)]:
    with st.container(border=True):
        st.markdown(f"**{label}** · {len(pieces)} tokens")
        st.markdown(lab.chips_markdown([t for _, t in pieces]))
        if show_ids:
            st.code(str([i for i, _ in pieces]), language=None, wrap_lines=True)

smol_meta = lab.load_meta("smollm2")
summary = [
    {
        "tokenizer": label,
        "vocab": s.vocab_size,
        "val bytes/token": s.val_bytes_per_token,
        "tokens for sentence": len(pieces),
    }
    for label, pieces, s in rows
] + [
    {
        "tokenizer": "SmolLM2 49k",
        "vocab": smol_meta["vocab_size"],
        "val bytes/token": smol_meta["val"]["bytes_per_token"],
        "tokens for sentence": len(smol),
    }
]
st.dataframe(
    pd.DataFrame(summary),
    hide_index=True,
    column_config={"val bytes/token": st.column_config.NumberColumn(format="%.2f")},
    alt="Vocabulary size, compression and sentence length for each tokenizer",
)
