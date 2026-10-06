import streamlit as st

from app_pages._cache import split
from booklm import tokenizer_lab as lab
from booklm.tokenizer_lab import ARTIFACTS

st.set_page_config(page_title="book-lm lab", page_icon=":material/menu_book:", layout="wide")

home = st.Page(
    "app_pages/home.py", title="Start here", icon=":material/rocket_launch:", default=True
)
own = st.Page(
    "app_pages/tokenization_own.py", title="Build our tokenizer", icon=":material/construction:"
)
smol = st.Page("app_pages/tokenization_smollm2.py", title="Inside SmolLM2's", icon=":material/hub:")
compare = st.Page(
    "app_pages/tokenization_compare.py", title="Compare them", icon=":material/compare_arrows:"
)
st.session_state.pages = {"own": own, "smol": smol, "compare": compare}

page = st.navigation({"": [home], "1 · Tokenization": [own, smol, compare]}, position="sidebar")

RANDOM = "Random validation paragraph"


def apply_preset() -> None:
    if st.session_state.preset == RANDOM:
        st.session_state.sentence = lab.random_paragraph(split("val"))
    else:
        st.session_state.sentence = lab.SAMPLES[st.session_state.preset]


if "sentence" not in st.session_state:
    st.session_state.sentence = next(iter(lab.SAMPLES.values()))

with st.sidebar:
    st.subheader("Sample text", divider="gray")
    st.selectbox("Preset", [*lab.SAMPLES, RANDOM], key="preset", on_change=apply_preset)
    if st.session_state.preset == RANDOM:
        st.button("Another paragraph", icon=":material/casino:", on_click=apply_preset)
    st.text_area("Edit freely", key="sentence", height=140, help="Shared by every panel.")

if page.title != "Start here" and not (ARTIFACTS / "data" / "clean" / "train.txt").exists():
    st.error(
        "No cleaned corpus found. Run `python -m booklm.download` and "
        "`python -m booklm.tokenize_own` first.",
        icon=":material/error:",
    )
    st.stop()

st.title(page.title, icon=page.icon)
page.run()
