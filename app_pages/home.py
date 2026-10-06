from pathlib import Path

import streamlit as st

plan = Path("PLAN.md").read_text(encoding="utf-8")
st.markdown(plan.split("\n", 1)[1])  # drop the H1; the app already shows a title
