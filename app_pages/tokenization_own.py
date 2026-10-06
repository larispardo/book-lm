import altair as alt
import pandas as pd
import streamlit as st

from app_pages._cache import DEFAULT_VOCAB, candidates, stats, tokenizer, training_run
from booklm import tokenizer_lab as lab

sentence = st.session_state.sentence

view = st.segmented_control(
    "View",
    [
        ":material/footprint: Step by step",
        ":material/straighten: Vocabulary size",
        ":material/search: Vocab browser",
    ],
    default=":material/footprint: Step by step",
    key="own_view",
    label_visibility="collapsed",
)

# ── step by step ─────────────────────────────────────────────────────────────────────────
if view and "Step by step" in view:
    st.caption(
        "BPE starts from the 256 possible bytes and greedily merges the most frequent "
        "adjacent pair in the training text, one pair per step. Slide through the merges "
        "and watch the sentence compress."
    )
    run = training_run(DEFAULT_VOCAB)
    bpe = tokenizer(DEFAULT_VOCAB)
    total = len(run.merges)
    stops = sorted(
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 500}
        | {750, 1000, 1500, 2000, 3000, 4000, 5000, 6000, 7000, total}
    )
    step = st.select_slider("Merges applied", stops, value=20, key="own_step")

    tokens = lab.tokens_at_step(run.merges, step, sentence)
    with st.container(horizontal=True):
        st.metric("Vocabulary so far", f"{256 + step:,}", border=True)
        st.metric("Tokens in sentence", len(tokens), border=True)
        n_bytes = len(sentence.encode("utf-8"))
        st.metric("Bytes per token", f"{n_bytes / max(1, len(tokens)):.2f}", border=True)
    st.markdown(lab.chips_markdown([t for _, t in tokens]))

    if step:
        left, right = (bpe.decode([i]) for i in run.merges[step - 1])
        st.info(
            f"Merge #{step} created id **{255 + step}**: `{lab.visible(left)}` + "
            f"`{lab.visible(right)}` → `{lab.visible(left + right)}`. It was chosen because "
            f"that pair occurred **{run.counts[step - 1]:,}** times at that moment, more than "
            "any other pair.",
            icon=":material/merge:",
        )

    left_col, right_col = st.columns([3, 2])
    with left_col:
        st.subheader("How often the chosen pair occurred", divider="gray")
        curve = pd.DataFrame({"merge": range(1, total + 1), "count": run.counts})
        chart = (
            alt.Chart(curve)
            .mark_line()
            .encode(
                x=alt.X("merge", title="merge number"),
                y=alt.Y("count", scale=alt.Scale(type="log"), title="occurrences (log)"),
            )
        )
        rule = alt.Chart(pd.DataFrame({"merge": [max(step, 1)]})).mark_rule(color="red")
        st.altair_chart(
            chart + rule.encode(x="merge"),
            alt="Frequency of each chosen pair, falling steeply as merges progress",
        )
        st.caption(
            "Early merges glue very common pairs (spaces + letters); late merges promote "
            "words seen a few dozen times. Training stops at the vocabulary budget."
        )
    with right_col:
        st.subheader("What the next step chose from", divider="gray")
        if step < total:
            rows = [
                {
                    "pair": f"{lab.visible(bpe.decode([a]))} + {lab.visible(bpe.decode([b]))}",
                    "count": count,
                    "chosen": (a, b) == run.merges[step],
                }
                for (a, b), count in candidates(step)
            ]
            st.dataframe(
                pd.DataFrame(rows),
                hide_index=True,
                column_config={"chosen": st.column_config.CheckboxColumn("chosen")},
                alt="Top candidate pairs and their counts at the next merge",
            )
        else:
            st.caption("Vocabulary budget reached: no more merges.")

    st.subheader("Latest merges", divider="gray")
    recent = range(max(0, step - 12), step)
    st.dataframe(
        pd.DataFrame(
            {
                "id": [256 + i for i in recent],
                "left": [lab.visible(bpe.decode([run.merges[i][0]])) for i in recent],
                "right": [lab.visible(bpe.decode([run.merges[i][1]])) for i in recent],
                "new token": [lab.visible(bpe.decode([256 + i])) for i in recent],
                "count when chosen": [run.counts[i] for i in recent],
            }
        ).iloc[::-1],
        hide_index=True,
        alt="The most recent merges up to the selected step",
    )

# ── vocabulary size ──────────────────────────────────────────────────────────────────────
elif view and "Vocabulary size" in view:
    st.caption(
        "Same algorithm, different budgets. A bigger vocabulary means longer tokens and "
        "shorter sequences, but a bigger embedding table and rarer tokens to learn."
    )
    sizes = st.pills(
        "Vocabulary sizes",
        [1024, 2048, 4096, 8192, 16384, 32768],
        selection_mode="multi",
        default=[4096, 8192, 16384],
        format_func=lambda v: f"{v // 1024}k",
        key="own_sizes",
    )
    if not sizes:
        st.stop()
    rows = []
    for size in sorted(sizes):
        s = stats(size)
        rows.append(
            {
                "vocab": f"{size // 1024}k",
                "merges learned": s.merges_learned,
                "train bytes/token": s.train_bytes_per_token,
                "val bytes/token": s.val_bytes_per_token,
                "val gap %": s.generalization_gap,
                f"embedding params (d={lab.D_MODEL})": s.embedding_params,
            }
        )
    table = pd.DataFrame(rows)
    st.dataframe(
        table,
        hide_index=True,
        column_config={
            "train bytes/token": st.column_config.NumberColumn(format="%.2f"),
            "val bytes/token": st.column_config.NumberColumn(format="%.2f"),
            "val gap %": st.column_config.NumberColumn(format="%.1f"),
            f"embedding params (d={lab.D_MODEL})": st.column_config.NumberColumn(format="compact"),
        },
        alt="Compression and cost for each vocabulary size",
    )
    long = table.melt(
        id_vars="vocab",
        value_vars=["train bytes/token", "val bytes/token"],
        var_name="split",
        value_name="bytes per token",
    )
    st.altair_chart(
        alt.Chart(long)
        .mark_line(point=True)
        .encode(
            x=alt.X("vocab", sort=None),
            y=alt.Y("bytes per token", scale=alt.Scale(zero=False)),
            color="split",
        ),
        alt="Bytes per token rises with vocabulary size, with diminishing returns",
    )
    st.caption(
        "Compression gains shrink as the vocabulary grows, while the train/val gap widens: "
        "big vocabularies start memorising words that only appear in the training text."
    )

    st.subheader("The sample sentence at each size", divider="gray")
    for size in sorted(sizes):
        pieces = lab.pieces(tokenizer(size), sentence)
        st.markdown(f"**{size // 1024}k** · {len(pieces)} tokens")
        st.markdown(lab.chips_markdown([t for _, t in pieces]))

# ── vocab browser (exercise) ─────────────────────────────────────────────────────────────
elif view and "Vocab browser" in view:
    bpe = tokenizer(DEFAULT_VOCAB)
    query = st.text_input("Find tokens containing", value="olmes", key="browser_query")
    try:
        tree = lab.merge_tree(bpe, bpe.encode(" Holmes")[0])
    except NotImplementedError:
        st.warning(
            "**Your exercise.** Implement `merge_tree` in `booklm/tokenizer_lab.py`, then "
            "remove the `xfail` marker in `tests/test_tokenizer_lab.py`. Ideas for this "
            "panel once it works: a searchable vocab table (id, token, bytes, parents, count "
            "when chosen) and, for a selected token, its full merge tree.",
            icon=":material/school:",
        )
        st.stop()
    st.json(tree)
