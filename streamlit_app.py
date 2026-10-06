import streamlit as st

from app_pages._cache import split
from booklm import tokenizer_lab as lab
from booklm.tokenizer_lab import ARTIFACTS

st.set_page_config(page_title="book-lm lab", page_icon=":material/menu_book:", layout="wide")

page = st.navigation(
    {
        "": [st.Page("app_pages/home.py", title="Roadmap", icon=":material/map:", default=True)],
        "Tokenization": [
            st.Page("app_pages/tokenization_own.py", title="Our BPE", icon=":material/token:"),
            st.Page("app_pages/tokenization_smollm2.py", title="SmolLM2", icon=":material/hub:"),
            st.Page(
                "app_pages/tokenization_compare.py",
                title="Side by side",
                icon=":material/compare_arrows:",
            ),
        ],
    },
    position="top",
)

RANDOM = "Random validation paragraph"


def apply_preset() -> None:
    if st.session_state.preset == RANDOM:
        st.session_state.sentence = lab.random_paragraph(split("val"))
    else:
        st.session_state.sentence = lab.SAMPLES[st.session_state.preset]


if "sentence" not in st.session_state:
    st.session_state.sentence = next(iter(lab.SAMPLES.values()))

with st.sidebar:
    st.selectbox("Sample text", [*lab.SAMPLES, RANDOM], key="preset", on_change=apply_preset)
    if st.session_state.preset == RANDOM:
        st.button("Another paragraph", icon=":material/casino:", on_click=apply_preset)
    st.text_area("Edit freely", key="sentence", height=140, help="Shared by every panel.")

if page.title != "Roadmap" and not (ARTIFACTS / "data" / "clean" / "train.txt").exists():
    st.error(
        "No cleaned corpus found. Run `python -m booklm.download` and "
        "`python -m booklm.tokenize_own` first.",
        icon=":material/error:",
    )
    st.stop()

st.title(page.title, icon=page.icon)
page.run()
