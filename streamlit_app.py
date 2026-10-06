import streamlit as st

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

with st.sidebar:
    st.text_area(
        "Sample sentence",
        value='Holmes looked at me and said, "Elementary, my dear Watson."',
        key="sentence",
        help="Shared by every panel.",
    )

if page.title != "Roadmap" and not (ARTIFACTS / "data" / "clean" / "train.txt").exists():
    st.error(
        "No cleaned corpus found. Run `python -m booklm.download` and "
        "`python -m booklm.tokenize_own` first.",
        icon=":material/error:",
    )
    st.stop()

st.title(page.title, icon=page.icon)
page.run()
