import altair as alt
import pandas as pd
import streamlit as st

from app_pages._cache import (
    DEFAULT_VOCAB,
    SPECIALS,
    candidates,
    frequency_words,
    full_run,
    max_vocab,
    stats,
    tokenizer,
    word_counts,
)
from booklm import tokenizer_lab as lab

sentence = st.session_state.sentence
BUDGETS = {"4k": 4096, "8k": 8192, "16k": 16384}

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
    run = full_run()
    bpe = tokenizer(max_vocab())  # ids are the same in every prefix, so one decoder fits all
    total = len(run.merges)
    st.caption(
        "BPE starts from the 256 possible bytes and greedily merges the most frequent "
        f"adjacent pair in the training text, one pair per step. On this corpus it can merge "
        f"{total:,} times before no pair occurs twice. Choosing a vocabulary size just means "
        "stopping early: the 8k tokenizer is exactly the first 7,935 merges."
    )
    stops = sorted(
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 500}
        | {750, 1000, 1500, 2000, 3000, 5000, 10000, 15000, total}
        | {size - 256 - SPECIALS for size in BUDGETS.values()}
    )
    step = st.select_slider(
        "Merges applied",
        stops,
        value=20,
        key="own_step",
        format_func=lambda s: next(
            (f"{s:,} ({name} budget)" for name, v in BUDGETS.items() if s == v - 256 - SPECIALS),
            f"{s:,}",
        ),
    )

    tokens = lab.tokens_at_step(run.merges, step, sentence)
    with st.container(horizontal=True):
        st.metric("Vocabulary so far", f"{256 + step + SPECIALS:,}", border=True)
        st.metric("Tokens in sample", len(tokens), border=True)
        n_bytes = len(sentence.encode("utf-8"))
        st.metric("Bytes per token", f"{n_bytes / max(1, len(tokens)):.2f}", border=True)
    st.markdown(lab.chips_markdown([t for _, t in tokens]))

    if step:
        left, right = (bpe.decode([i]) for i in run.merges[step - 1])
        st.info(
            f"Merge #{step:,} created id **{255 + step}**: `{lab.visible(left)}` + "
            f"`{lab.visible(right)}` → `{lab.visible(left + right)}`. It was chosen because "
            f"that pair occurred **{run.counts[step - 1]:,}** times at that moment, more than "
            "any other pair.",
            icon=":material/merge:",
        )

    left_col, right_col = st.columns([3, 2])
    with left_col:
        st.subheader("How often the chosen pair occurred", divider="gray")
        curve = pd.DataFrame({"merge": range(1, total + 1), "count": run.counts})
        line = (
            alt.Chart(curve)
            .mark_line()
            .encode(
                x=alt.X("merge", title="merge number"),
                y=alt.Y("count", scale=alt.Scale(type="log"), title="occurrences (log)"),
            )
        )
        budgets = pd.DataFrame(
            {"merge": [v - 256 - SPECIALS for v in BUDGETS.values()], "budget": list(BUDGETS)}
        )
        budget_rules = (
            alt.Chart(budgets)
            .mark_rule(strokeDash=[4, 4], color="gray")
            .encode(x="merge", tooltip=["budget"])
        )
        budget_labels = (
            alt.Chart(budgets)
            .mark_text(align="left", dx=3, dy=-4, color="gray")
            .encode(x="merge", y=alt.value(8), text="budget")
        )
        here = (
            alt.Chart(pd.DataFrame({"merge": [max(step, 1)]}))
            .mark_rule(color="red")
            .encode(x="merge")
        )
        st.altair_chart(
            line + budget_rules + budget_labels + here,
            alt="Frequency of each chosen pair, falling steeply; dashed lines mark vocab budgets",
        )
        st.caption(
            "Early merges glue very common pairs (space + letter). By the 8k budget a merge is "
            "worth only a few dozen occurrences; at the end, pairs seen just twice."
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
            st.caption("No pair occurs twice any more: training is over.")

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
    biggest = max_vocab()
    st.caption(
        "Same training run, cut at different points. Bigger vocabularies mean longer tokens "
        "and shorter sequences, but a bigger embedding table and rarer tokens to learn. "
        f"This corpus caps out at {biggest:,}: after that no pair occurs twice."
    )
    options = [512, 1024, 2048, 4096, 8192, 16384, biggest]

    def size_label(v: int) -> str:
        return f"max ({v / 1024:.1f}k)" if v == biggest else f"{v / 1024:g}k"

    sizes = st.pills(
        "Vocabulary sizes",
        options,
        selection_mode="multi",
        default=[1024, 4096, 8192, 16384, biggest],
        format_func=size_label,
        key="own_sizes",
    )
    if not sizes:
        st.stop()
    sizes = sorted(sizes)

    rows = []
    for size in sizes:
        s = stats(size)
        rows.append(
            {
                "vocab": size_label(size),
                "vocab size": s.vocab_size,
                "train bytes/token": s.train_bytes_per_token,
                "val bytes/token": s.val_bytes_per_token,
                "val gap %": s.generalization_gap,
                "tokens in sample": len(tokenizer(size).encode(sentence)),
                "embedding params": s.embedding_params,
            }
        )
    table = pd.DataFrame(rows)

    chart_col, side_col = st.columns([3, 1], gap="large")
    with side_col:
        metric = st.radio(
            "Plot",
            ["Bytes per token", "Val gap %", "Tokens in sample", "Embedding params"],
            key="own_metric",
        )
        st.caption(
            {
                "Bytes per token": "Higher = each token carries more text. Watch val fall "
                "behind train as the vocabulary grows.",
                "Val gap %": "How much worse held-out text compresses. Growth means late "
                "merges memorise training-only words.",
                "Tokens in sample": "Sequence length the model would see for the sample text "
                "in the sidebar.",
                "Embedding params": f"vocab × {lab.D_MODEL}: the first layer of the model, "
                "paid for every token in the vocabulary.",
            }[metric]
        )
    columns = {
        "Bytes per token": ["train bytes/token", "val bytes/token"],
        "Val gap %": ["val gap %"],
        "Tokens in sample": ["tokens in sample"],
        "Embedding params": ["embedding params"],
    }[metric]
    long = table.melt(
        id_vars=["vocab", "vocab size"], value_vars=columns, var_name="series", value_name="value"
    )
    with chart_col:
        st.altair_chart(
            alt.Chart(long)
            .mark_line(point=alt.OverlayMarkDef(size=60))
            .encode(
                x=alt.X(
                    "vocab size",
                    scale=alt.Scale(type="log", base=2),
                    axis=alt.Axis(values=list(table["vocab size"]), format="~s"),
                    title="vocabulary size (log scale)",
                ),
                y=alt.Y("value", scale=alt.Scale(zero=False, nice=True), title=metric),
                color=alt.Color("series", legend=alt.Legend(orient="top", title=None)),
                tooltip=["vocab", "series", alt.Tooltip("value", format=",.2f")],
            )
            .properties(height=420),
            alt=f"{metric} for each selected vocabulary size",
        )
    st.dataframe(
        table.drop(columns="vocab size"),
        hide_index=True,
        column_config={
            "train bytes/token": st.column_config.NumberColumn(format="%.2f"),
            "val bytes/token": st.column_config.NumberColumn(format="%.2f"),
            "val gap %": st.column_config.NumberColumn(format="%.1f"),
            "embedding params": st.column_config.NumberColumn(
                f"embedding params (d={lab.D_MODEL})", format="compact"
            ),
        },
        alt="Compression and cost for each vocabulary size",
    )

    st.subheader("Words across frequency bands", divider="gray")
    st.caption(
        "Words picked from the training text at every frequency level, from thousands of "
        "occurrences down to one. Read across a row to see the size at which a word becomes "
        "a single token; rare and unseen words stay in pieces."
    )
    extra = st.text_input(
        "Add your own words (comma separated)",
        value="smartphone, Kubernetes, ciudad",
        key="own_extra_words",
    )
    counts = word_counts()
    own = [" " + w.strip() for w in extra.split(",") if w.strip()]
    words = list(frequency_words()) + [(w, counts.get(w, 0)) for w in own]
    word_rows = []
    for word, count in words:
        row = {"word": word.strip(), "times in train": count}
        for size in sizes:
            parts = [t for _, t in lab.pieces(tokenizer(size), word)]
            row[size_label(size)] = " | ".join(lab.visible(t) for t in parts)
        word_rows.append(row)
    st.dataframe(
        pd.DataFrame(word_rows),
        hide_index=True,
        height=36 * (len(word_rows) + 1) + 3,
        column_config={"times in train": st.column_config.NumberColumn(format="%d")},
        alt="How words at each frequency level split under each vocabulary size",
    )

    st.subheader("The sample at each size", divider="gray")
    for size in sizes:
        pieces = lab.pieces(tokenizer(size), sentence)
        st.markdown(f"**{size_label(size)}** · {len(pieces)} tokens")
        st.markdown(lab.chips_markdown([t for _, t in pieces]))

# ── vocab browser (exercise) ─────────────────────────────────────────────────────────────
elif view and "Vocab browser" in view:
    bpe = tokenizer(DEFAULT_VOCAB)
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
