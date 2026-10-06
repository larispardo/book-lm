import altair as alt
import pandas as pd
import streamlit as st

from app_pages._cache import DEFAULT_VOCAB, gpt_model, run_log, tokenizer
from booklm import pretrain_lab as lab
from booklm.tokenizer_lab import visible

runs = lab.list_runs()
if not runs:
    st.info(
        "No training runs yet. Start one with "
        "`python -m booklm.pretrain --preset tiny --steps 300`.",
        icon=":material/model_training:",
    )
    st.stop()

with st.container(horizontal=True, vertical_alignment="bottom"):
    chosen = st.pills(
        "Runs", runs, selection_mode="multi", default=runs, key="pt_runs", width="stretch"
    )
    st.button("Refresh", icon=":material/refresh:", help="Re-read logs of runs in progress")
if not chosen:
    st.stop()
loaded = {name: run_log(name) for name in chosen}

with st.container(horizontal=True):
    for run in loaded.values():
        best = run.best
        st.metric(
            run.name.removeprefix("holmes-gpt-"),
            "—" if best is None else f"{best.val_loss:.2f}",
            help="Best validation loss (nats per token)",
            border=True,
            chart_data=None if run.evals.empty else run.evals["val_loss"].tolist(),
            chart_type="line",
            delta=None
            if best is None
            else f"ppl {lab.perplexity(best.val_loss):.0f} · step {int(best.step)}",
            delta_color="off",
        )

view = st.segmented_control(
    "View",
    [
        ":material/show_chart: Loss curves",
        ":material/history_edu: Samples over training",
        ":material/grid_on: Attention maps",
    ],
    default=":material/show_chart: Loss curves",
    key="pt_view",
    label_visibility="collapsed",
)

# ── loss curves ──────────────────────────────────────────────────────────────────────────
if view and "Loss curves" in view:
    axis = st.segmented_control("X axis", ["steps", "epochs"], default="steps", key="pt_axis")
    frames = []
    for run in loaded.values():
        tr = run.train.assign(split="train", loss=run.train["train_loss"])
        ev = run.evals.assign(split="val", loss=run.evals["val_loss"])
        for df in (tr, ev):
            df = df.assign(run=run.name.removeprefix("holmes-gpt-"))
            df["x"] = run.epochs(df["step"]) if axis == "epochs" else df["step"]
            frames.append(df[["run", "split", "x", "step", "loss"]])
    data = pd.concat(frames)
    base = alt.Chart(data).encode(
        x=alt.X("x", title=axis or "steps"),
        y=alt.Y("loss", title="cross-entropy loss (nats)", scale=alt.Scale(zero=False)),
        color=alt.Color("run", legend=alt.Legend(orient="top", title=None)),
        strokeDash=alt.StrokeDash("split", legend=alt.Legend(orient="top", title=None)),
        tooltip=["run", "split", "step", alt.Tooltip("loss", format=".3f")],
    )
    train_line = base.transform_filter(alt.datum.split == "train").mark_line(opacity=0.45)
    val_line = base.transform_filter(alt.datum.split == "val").mark_line(point=True)
    st.altair_chart(
        (train_line + val_line).properties(height=420),
        alt="Training and validation loss over time for each selected run",
    )
    st.caption(
        "Solid = train, dashed with dots = held-out val. While the two move together the "
        "model is learning patterns that generalise; when train keeps falling but val "
        "flattens or rises, it is memorising the training text (overfitting). A loss of "
        "9.0 is random guessing among 8,192 tokens."
    )

# ── samples ──────────────────────────────────────────────────────────────────────────────
elif view and "Samples" in view:
    name = st.selectbox("Run", chosen, key="pt_sample_run")
    evals = loaded[name].evals
    if evals.empty:
        st.stop()
    steps = evals["step"].tolist()
    step = st.select_slider("Checkpoint", steps, value=steps[-1], key="pt_sample_step")
    row = evals[evals["step"] == step].iloc[0]
    st.metric("Val loss at this step", f"{row.val_loss:.2f}", border=True)
    with st.container(border=True):
        st.text(row["sample"])
    st.caption(
        "40 tokens generated after the prompt 'Holmes' (temperature 0.8, top-k 50) at each "
        "evaluation. Watch it learn word frequency, then punctuation and dialogue, then "
        "short grammatical phrases."
    )
    with st.expander("All samples of this run", icon=":material/list:"):
        st.dataframe(
            evals[["step", "val_loss", "sample"]],
            hide_index=True,
            column_config={"val_loss": st.column_config.NumberColumn(format="%.2f")},
            alt="Every evaluation sample of the selected run",
        )

# ── attention maps ───────────────────────────────────────────────────────────────────────
elif view and "Attention" in view:
    name = st.selectbox("Run", chosen, key="pt_attn_run")
    model, ckpt_step = gpt_model(name)
    bpe = tokenizer(DEFAULT_VOCAB)
    ids = bpe.encode(st.session_state.sentence)[:32]
    tokens = [f"{i:02d} {visible(bpe.decode([t]))}" for i, t in enumerate(ids)]
    maps = lab.attention_maps(model, ids)  # (layers, heads, T, T)
    n_layers, n_heads = maps.shape[:2]

    c1, c2 = st.columns(2)
    layer = c1.segmented_control(
        "Layer", list(range(n_layers)), default=0, key="pt_layer", format_func=lambda i: f"L{i}"
    )
    head = c2.segmented_control(
        "Head",
        ["all", *range(n_heads)],
        default="all",
        key="pt_head",
        format_func=lambda h: h if h == "all" else f"H{h}",
    )
    layer = layer or 0
    st.caption(
        f"Best checkpoint (step {ckpt_step}) reading the first {len(ids)} tokens of the "
        "sidebar text. Each row is a token looking back: darker cells are where its "
        "attention goes. The upper triangle is empty because the future is masked."
    )

    def frame(h: int) -> pd.DataFrame:
        w = maps[layer, h]
        return pd.DataFrame(
            [
                {"query": tokens[q], "key": tokens[k], "weight": float(w[q, k]), "head": f"H{h}"}
                for q in range(len(ids))
                for k in range(q + 1)
            ]
        )

    heads = range(n_heads) if head in (None, "all") else [head]
    data = pd.concat(frame(h) for h in heads)
    heat = (
        alt.Chart(data)
        .mark_rect()
        .encode(
            x=alt.X("key:O", sort=tokens, title=None, axis=alt.Axis(labelAngle=-60)),
            y=alt.Y("query:O", sort=tokens, title=None),
            color=alt.Color("weight:Q", scale=alt.Scale(scheme="blues", domain=[0, 1])),
            tooltip=["query", "key", alt.Tooltip("weight", format=".2f")],
        )
    )
    if len(heads) > 1:
        size = 240 if n_heads <= 4 else 200
        chart = heat.properties(width=size, height=size).facet(
            facet=alt.Facet("head:N", title=None), columns=min(n_heads, 3)
        )
    else:
        chart = heat.properties(height=560)
    st.altair_chart(chart, alt=f"Attention weights of layer {layer}")
