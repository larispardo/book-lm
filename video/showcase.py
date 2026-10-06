"""Manim showcase of BookLM. Every number on screen comes from video/data (see export_traces.py).

Render:  manim -ql video/showcase.py Showcase      (480p preview)
         manim -qh video/showcase.py Showcase      (1080p)
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import numpy as np
from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UL,
    UP,
    Arrow,
    Axes,
    Create,
    DashedLine,
    FadeIn,
    FadeOut,
    GrowFromEdge,
    Indicate,
    LaggedStart,
    Line,
    ManimColor,
    RoundedRectangle,
    Scene,
    Text,
    Transform,
    ValueTracker,
    VGroup,
    Write,
    always_redraw,
    config,
    interpolate_color,
)

DATA = Path(__file__).parent / "data"
config.background_color = "#0f1117"

MONO = "Menlo"
FG = ManimColor("#e5e7eb")
MUTED = ManimColor("#6b7280")
ACCENT = ManimColor("#f472b6")
LOW = ManimColor("#ef4444")
HIGH = ManimColor("#22c55e")
MODEL_COLORS = {
    "holmes-scratch": ManimColor("#f59e0b"),
    "smollm2-base": ManimColor("#60a5fa"),
    "holmes-lora": ManimColor("#34d399"),
}
MODEL_BLURB = {
    "holmes-scratch": "random init → books only",
    "smollm2-base": "pretrained on ~2T web tokens",
    "holmes-lora": "pretrained + LoRA on books",
}


def vis(token: str) -> str:
    """Make whitespace visible: spaces as ·, newlines as ↵."""
    return token.replace("\n", "↵").replace(" ", "·") or "∅"


def prob_color(p: float) -> ManimColor:
    return interpolate_color(LOW, HIGH, min(1.0, max(0.0, p)) ** 0.5)


def softmax(logits: np.ndarray, t: float) -> np.ndarray:
    z = logits / max(t, 1e-4)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def entropy_bits(p: np.ndarray) -> float:
    nz = p[p > 0]
    return float(-(nz * np.log2(nz)).sum())


def label(text: str, size: int = 28, color=FG, mono: bool = False) -> Text:
    return Text(text, font_size=size, color=color, font=MONO if mono else "")


def chip(token: str, color=FG, fill=None, size: int = 26) -> VGroup:
    t = label(vis(token), size, color=FG, mono=True)
    box = RoundedRectangle(
        corner_radius=0.08, width=t.width + 0.25, height=t.height + 0.25, stroke_color=color
    )
    if fill is not None:
        box.set_fill(fill, opacity=0.35)
    return VGroup(box, t)


def hbars(tokens, probs, width=6.0, scale=1.0, kept=None, highlight=None, size=24) -> VGroup:
    """Horizontal bar chart: token label | bar | percentage. Bars share a fixed scale."""
    rows = VGroup()
    for i, (tok, p) in enumerate(zip(tokens, probs, strict=True)):
        alive = kept is None or kept[i]
        color = ACCENT if i == highlight else (prob_color(p) if alive else MUTED)
        name = label(vis(tok), size, color=FG if alive else MUTED, mono=True)
        bar = RoundedRectangle(
            corner_radius=0.04,
            width=max(0.02, p / scale * width),
            height=0.32,
            stroke_width=0,
            fill_color=color,
            fill_opacity=0.9 if alive else 0.25,
        )
        pct = label(f"{p * 100:5.1f}%", size - 4, color=FG if alive else MUTED, mono=True)
        name.move_to(np.array([-0.2, -i * 0.48, 0]), aligned_edge=RIGHT)
        bar.move_to(np.array([0, -i * 0.48, 0]), aligned_edge=LEFT)
        pct.next_to(bar, RIGHT, buff=0.15)
        rows.add(VGroup(name, bar, pct))
    return rows


class Showcase(Scene):
    def setup(self):
        self.trace = json.loads((DATA / "trace.json").read_text())
        self.logits = np.load(DATA / "logits.npy").astype(np.float64)
        self.top = np.argsort(-self.logits)[:10]
        self.top_tokens = [self.trace["token_strs"][str(i)] for i in self.top]

    def construct(self):
        self.title()
        self.models()
        self.losses()
        self.pipeline()
        self.temperature()
        self.truncation()
        self.autoregression()
        self.branching()
        self.compare()
        self.outro()

    def clear_all(self):
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.6)

    def heading(self, text: str, sub: str | None = None) -> VGroup:
        h = VGroup(label(text, 40))
        if sub:
            h.add(label(sub, 24, MUTED).next_to(h[0], DOWN, buff=0.15, aligned_edge=LEFT))
        h.to_corner(UL, buff=0.5)
        self.play(FadeIn(h, shift=DOWN * 0.2), run_time=0.6)
        return h

    # 1 ─────────────────────────────────────────────────────────────────────────────
    def title(self):
        corpus = self.trace["corpus"]
        t = label("BookLM", 80)
        s = label("a glass-box language model", 36, MUTED).next_to(t, DOWN)
        d = label(
            f"trained on the Sherlock Holmes canon · {len(corpus['books'])} books · "
            f"{corpus['tokens']:,} tokens · public domain",
            24,
            MUTED,
        ).next_to(s, DOWN, buff=0.5)
        self.play(Write(t), run_time=1.2)
        self.play(FadeIn(s, shift=UP * 0.2), FadeIn(d, shift=UP * 0.2))
        self.wait(1.5)
        self.clear_all()

    # 2 ─────────────────────────────────────────────────────────────────────────────
    def models(self):
        self.heading("Three models, one tokenizer", "lower perplexity on held-out Holmes = better")
        cards = self.trace["cards"]
        worst = max(c["val_ppl"] for c in cards.values())
        rows = VGroup()
        for i, (name, card) in enumerate(cards.items()):
            color = MODEL_COLORS.get(name, FG)
            title = label(name, 30, color, mono=True)
            blurb = label(MODEL_BLURB.get(name, card["kind"]), 20, MUTED)
            trainable = card["trainable_params"]
            params = label(
                f"{card['total_params'] / 1e6:.0f}M params · {trainable / 1e6:.1f}M trained",
                20,
                MUTED,
            )
            left = VGroup(title, blurb, params).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
            left.move_to(np.array([-3.6, 1.2 - i * 1.6, 0]), aligned_edge=LEFT)
            left.shift(LEFT * 1.8)
            bar = RoundedRectangle(
                corner_radius=0.05,
                width=card["val_ppl"] / worst * 5.0,
                height=0.45,
                stroke_width=0,
                fill_color=color,
                fill_opacity=0.85,
            ).move_to(np.array([0.4, 1.2 - i * 1.6, 0]), aligned_edge=LEFT)
            value = label(f"ppl {card['val_ppl']:.1f}", 24, FG, mono=True).next_to(bar, RIGHT)
            rows.add(VGroup(left, bar, value))
        for row in rows:
            self.play(FadeIn(row[0]), GrowFromEdge(row[1], LEFT), FadeIn(row[2]), run_time=0.8)
        self.wait(2.5)
        self.clear_all()

    # 3 ─────────────────────────────────────────────────────────────────────────────
    def losses(self):
        self.heading("Training curves", "dots = train batches · line = held-out validation")
        cards = self.trace["cards"]
        panels = VGroup()
        specs = [
            ("holmes-scratch", "from scratch"),
            ("holmes-lora", "LoRA on SmolLM2-135M"),
        ]
        for col, (name, title) in enumerate(specs):
            if name not in cards:
                continue
            hist = cards[name]["history"]
            steps = max(h["step"] for h in hist)
            losses = [h[k] for h in hist for k in ("train_loss", "val_loss") if h[k] is not None]
            axes = Axes(
                x_range=[0, steps, steps // 4],
                y_range=[0, max(losses) * 1.05, 2],
                x_length=5.2,
                y_length=3.6,
                tips=False,
                axis_config={"color": MUTED, "include_numbers": True, "font_size": 18},
            ).move_to(np.array([-3.4 + col * 6.8, -0.6, 0]))
            color = MODEL_COLORS[name]
            dots = VGroup(
                *[
                    axes.plot_line_graph(
                        [h["step"]],
                        [h["train_loss"]],
                        vertex_dot_radius=0.025,
                        line_color=color,
                        vertex_dot_style={"fill_color": color, "fill_opacity": 0.4},
                    )
                    for h in hist
                    if h["train_loss"] is not None
                ]
            )
            val = [(h["step"], h["val_loss"]) for h in hist if h["val_loss"] is not None]
            val_line = axes.plot_line_graph(
                [v[0] for v in val],
                [v[1] for v in val],
                line_color=FG,
                vertex_dot_radius=0.04,
                vertex_dot_style={"fill_color": FG},
            )
            caption = label(title, 26, color).next_to(axes, UP)
            panel = VGroup(axes, caption, dots, val_line)
            if name == "holmes-lora" and "smollm2-base" in cards:
                base = cards["smollm2-base"]["val_loss"]
                ref = DashedLine(
                    axes.c2p(0, base), axes.c2p(steps, base), color=MODEL_COLORS["smollm2-base"]
                )
                note = label("base model, no finetune", 18, MODEL_COLORS["smollm2-base"])
                panel.add(ref, note.next_to(ref, UP, buff=0.08).align_to(ref, RIGHT))
            best = min(v[1] for v in val)
            panel.add(
                label(f"best val loss {best:.2f}", 22, FG, mono=True).next_to(axes, DOWN, buff=0.5)
            )
            panels.add(panel)
        for panel in panels:
            self.play(Create(panel[0]), FadeIn(panel[1]), run_time=0.8)
            self.play(LaggedStart(*[FadeIn(d) for d in panel[2]], lag_ratio=0.02), run_time=1.2)
            self.play(Create(panel[3]), *[FadeIn(m) for m in panel[4:]], run_time=1.0)
        self.wait(2.5)
        self.clear_all()

    # 4 ─────────────────────────────────────────────────────────────────────────────
    def pipeline(self):
        self.heading("One step of generation", f"model: {self.trace['focus_model']}")
        chips = VGroup(*[chip(t, MUTED) for t in self.trace["prompt_tokens"]]).arrange(
            RIGHT, buff=0.1
        )
        chips.move_to(UP * 1.6)
        cap = label("prompt → tokens", 22, MUTED).next_to(chips, UP)
        self.play(
            FadeIn(cap), LaggedStart(*[FadeIn(c, shift=DOWN * 0.2) for c in chips], lag_ratio=0.1)
        )

        box = RoundedRectangle(
            corner_radius=0.15, width=4.2, height=0.9, color=MODEL_COLORS["holmes-lora"]
        )
        params = self.trace["cards"][self.trace["focus_model"]]["total_params"]
        box_text = label(f"transformer · {params / 1e6:.0f}M params", 22)
        model = VGroup(box, box_text.move_to(box)).move_to(UP * 0.1)
        self.play(
            Create(Arrow(chips.get_bottom(), model.get_top(), color=MUTED, buff=0.1)), FadeIn(model)
        )

        k = 6
        logit_vals = self.logits[self.top[:k]]
        logit_rows = VGroup()
        for i, (tok, z) in enumerate(zip(self.top_tokens[:k], logit_vals, strict=True)):
            logit_rows.add(
                VGroup(
                    label(vis(tok), 22, FG, mono=True),
                    label(f"{z:6.2f}", 22, MUTED, mono=True),
                )
                .arrange(RIGHT, buff=0.4)
                .move_to(np.array([-3.2, -1.2 - i * 0.4, 0]))
            )
        logit_cap = label(f"logits: one score per token ({self.trace['vocab_size']:,})", 22, MUTED)
        logit_cap.next_to(logit_rows, UP)
        self.play(
            Create(
                Arrow(model.get_bottom(), logit_cap.get_top() + RIGHT * 0.3, color=MUTED, buff=0.1)
            ),
            FadeIn(logit_cap),
            LaggedStart(*[FadeIn(r) for r in logit_rows], lag_ratio=0.08),
        )
        probs = softmax(self.logits, 1.0)[self.top[:k]]
        bars = hbars(self.top_tokens[:k], probs, width=3.5, size=20).scale(0.85)
        bars.move_to(np.array([3.4, -2.1, 0]))
        sm = label("softmax → probabilities", 22, MUTED).next_to(bars, UP)
        self.play(
            Create(Arrow(logit_rows.get_right(), bars.get_left(), color=MUTED, buff=0.3)),
            FadeIn(sm),
            LaggedStart(*[GrowFromEdge(r, LEFT) for r in bars], lag_ratio=0.08),
        )
        self.wait(2.5)
        self.clear_all()

    # 5 ─────────────────────────────────────────────────────────────────────────────
    def temperature(self):
        self.heading("Temperature", "softmax(logits / T) over the full vocabulary")
        prompt = label(self.trace["prompt"] + " …", 26, MUTED).move_to(UP * 2.0)
        self.play(FadeIn(prompt))
        t = ValueTracker(1.0)

        def chart():
            p = softmax(self.logits, t.get_value())
            g = hbars(self.top_tokens[:8], p[self.top[:8]], width=6.0)
            return g.move_to(DOWN * 0.8 + RIGHT * 0.5)

        def readout():
            p = softmax(self.logits, t.get_value())
            return (
                VGroup(
                    label(f"T = {t.get_value():.2f}", 34, ACCENT, mono=True),
                    label(f"entropy {entropy_bits(p):5.2f} bits", 22, MUTED, mono=True),
                )
                .arrange(DOWN, aligned_edge=LEFT)
                .to_edge(RIGHT, buff=0.8)
                .shift(UP * 1.2)
            )

        bars, info = always_redraw(chart), always_redraw(readout)
        self.play(FadeIn(bars), FadeIn(info))
        self.wait(0.8)
        for target, note in [
            (0.3, "T < 1: sharper, safer"),
            (2.0, "T > 1: flatter, wilder"),
            (1.0, ""),
        ]:
            caption = label(note, 26, ACCENT).to_edge(DOWN, buff=0.4)
            if note:
                self.play(FadeIn(caption), run_time=0.3)
            self.play(t.animate.set_value(target), run_time=2.2)
            self.wait(0.6)
            if note:
                self.play(FadeOut(caption), run_time=0.3)
        self.wait(0.5)
        self.clear_all()

    # 6 ─────────────────────────────────────────────────────────────────────────────
    def truncation(self):
        self.heading("Truncation: who is allowed to be sampled", "then survivors are renormalised")
        n = 10
        p_all = softmax(self.logits, 1.0)
        p = p_all[self.top[:n]]
        scale = 1.0
        bars = hbars(self.top_tokens[:n], p, scale=scale, size=22).move_to(DOWN * 0.6 + RIGHT * 0.3)
        self.play(FadeIn(bars))

        # Full-vocab keep masks, same rules as booklm.sampling.
        order = np.argsort(-p_all)
        rank = np.empty_like(order)
        rank[order] = np.arange(len(order))
        before = (np.cumsum(p_all[order]) - p_all[order])[rank]  # mass ranked above each token
        rules = [
            ("top-k = 3", rank < 3, "keep the 3 most likely tokens"),
            ("top-p = 0.9", before < 0.9, "keep the smallest set reaching 90% of the mass"),
            (
                "min-p = 0.1",
                p_all >= 0.1 * p_all.max(),
                "keep tokens ≥ 10% as likely as the top one",
            ),
        ]
        for name, mask, explain in rules:
            explain += f"  →  {int(mask.sum()):,} survive"
            kept = mask[self.top[:n]]
            # Renormalise over every survivor in the vocab, not only the ones on screen.
            renorm = np.where(kept, p, 0.0) / p_all[mask].sum()
            tag = (
                VGroup(label(name, 32, ACCENT, mono=True), label(explain, 22, MUTED))
                .arrange(DOWN, aligned_edge=LEFT)
                .to_edge(DOWN, buff=0.35)
            )
            self.play(FadeIn(tag))
            greyed = hbars(self.top_tokens[:n], p, scale=scale, kept=kept, size=22).move_to(bars)
            greyed.align_to(bars, LEFT)
            self.play(Transform(bars, greyed), run_time=0.8)
            grown = hbars(self.top_tokens[:n], renorm, scale=scale, kept=kept, size=22).align_to(
                bars, LEFT
            )
            grown.align_to(bars, UP)
            self.play(Transform(bars, grown), run_time=1.0)
            self.wait(1.2)
            reset = (
                hbars(self.top_tokens[:n], p, scale=scale, size=22)
                .align_to(bars, LEFT)
                .align_to(bars, UP)
            )
            self.play(Transform(bars, reset), FadeOut(tag), run_time=0.6)
        self.clear_all()

    # 7 ─────────────────────────────────────────────────────────────────────────────
    def autoregression(self):
        self.heading(
            "Autoregression: sample, append, repeat",
            "T 0.8 · top-p 0.95 · chip colour = model's probability",
        )
        text = VGroup(*[chip(t, MUTED) for t in self.trace["prompt_tokens"]]).arrange(
            RIGHT, buff=0.08
        )
        text.scale(0.8).move_to(UP * 1.6).to_edge(LEFT, buff=0.6)
        self.play(FadeIn(text))
        chart = None
        for s in self.trace["walk"][:10]:
            cands = s["candidates"][:5]
            ids = [c["id"] for c in cands]
            hi = ids.index(s["token_id"]) if s["token_id"] in ids else None
            new_chart = hbars(
                [c["token"] for c in cands],
                [c["final_prob"] for c in cands],
                width=4.0,
                highlight=hi,
                size=22,
            ).move_to(DOWN * 1.4)
            meta = label(
                f"step {s['index'] + 1} · entropy {s['entropy_bits']:.2f} bits"
                f" · picked rank {s['rank'] + 1}",
                20,
                MUTED,
                mono=True,
            ).next_to(new_chart, DOWN, buff=0.35)
            new_chart = VGroup(new_chart, meta)
            if chart is None:
                chart = new_chart
                self.play(FadeIn(chart), run_time=0.5)
            else:
                self.play(Transform(chart, new_chart), run_time=0.5)
            c = chip(s["token"], prob_color(s["raw_prob"]), fill=prob_color(s["raw_prob"])).scale(
                0.8
            )
            c.next_to(text, RIGHT, buff=0.08)
            if c.get_right()[0] > config.frame_width / 2 - 0.4:
                c.next_to(text, DOWN, buff=0.2).align_to(text, LEFT)
                text = VGroup(c)  # start a new line
            else:
                text.add(c)
            self.play(FadeIn(c, shift=UP * 0.3), run_time=0.45)
        self.wait(1.5)
        self.clear_all()

    # 8 ─────────────────────────────────────────────────────────────────────────────
    def branching(self):
        b = self.trace["branches"]
        self.heading(
            "Branching: one different token, a different story", "both continuations are greedy"
        )
        prefix = "".join(
            [self.trace["prompt"]] + [s["token"] for s in self.trace["walk"][: b["fork_index"]]]
        )
        root = label(textwrap.shorten(prefix, 60, placeholder="…"), 26, FG).move_to(
            UP * 1.4 + LEFT * 1.0
        )

        def leaf(token, prob, steps, y, color):
            cont = "".join(s["token"] for s in steps)
            head = chip(token, color, fill=color)
            body = label(textwrap.fill(cont.replace("\n", " "), 42), 22, FG)
            p = label(f"p = {prob:.2f}", 20, MUTED, mono=True)
            g = VGroup(VGroup(head, p).arrange(RIGHT, buff=0.3), body).arrange(
                DOWN, aligned_edge=LEFT
            )
            return g.move_to(np.array([1.6, y, 0]), aligned_edge=LEFT)

        top = leaf(b["original_token"], b["original_prob"], b["original"], 0.0, HIGH)
        bottom = leaf(b["alternative_token"], b["alternative_prob"], b["alternative"], -2.0, ACCENT)
        self.play(FadeIn(root))
        for node in (top, bottom):
            edge = Line(
                root.get_bottom() + DOWN * 0.1, node[0].get_left() + LEFT * 0.1, color=MUTED
            )
            self.play(Create(edge), FadeIn(node[0]), run_time=0.6)
            self.play(Write(node[1]), run_time=1.4)
        self.play(Indicate(bottom[0], color=ACCENT))
        self.wait(2.5)
        self.clear_all()

    # 9 ─────────────────────────────────────────────────────────────────────────────
    def compare(self):
        self.heading("Same prompt, same seed, three models", f"“{self.trace['prompt']}…”")
        cols = VGroup()
        for name, text in self.trace["compare"].items():
            color = MODEL_COLORS.get(name, FG)
            head = VGroup(
                label(name, 26, color, mono=True), label(MODEL_BLURB.get(name, ""), 18, MUTED)
            )
            head.arrange(DOWN, aligned_edge=LEFT, buff=0.08)
            body = label(textwrap.fill(" ".join(text.split()), 30), 20, FG)
            cols.add(VGroup(head, body).arrange(DOWN, aligned_edge=LEFT, buff=0.3))
        cols.arrange(RIGHT, buff=0.6, aligned_edge=UP).move_to(DOWN * 0.5)
        if cols.width > config.frame_width - 0.8:
            cols.scale_to_fit_width(config.frame_width - 0.8)
        for col in cols:
            self.play(FadeIn(col[0]), Write(col[1]), run_time=1.6)
        self.wait(4)
        self.clear_all()

    # 10 ────────────────────────────────────────────────────────────────────────────
    def outro(self):
        lines = VGroup(
            label("Explore it yourself", 44),
            label("make serve  →  http://localhost:8000", 28, ACCENT, mono=True),
            label("docker compose up", 28, ACCENT, mono=True),
            label(
                "PyTorch · Transformers · PEFT · FastAPI · Manim · SmolLM2 · Project Gutenberg",
                20,
                MUTED,
            ),
        ).arrange(DOWN, buff=0.4)
        self.play(LaggedStart(*[FadeIn(m, shift=UP * 0.2) for m in lines], lag_ratio=0.25))
        self.wait(3)
