"""Render the checked-in training diagrams with optional Matplotlib.

Run from the repository root:
    python tools/render_training_diagrams.py --preview-dir /tmp/training-diagrams
Matplotlib is needed only to regenerate the documentation figures.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

COLORS = {
    "data": ("#e8f1ff", "#2765a6"),
    "bc": ("#e7f5ef", "#237456"),
    "rl": ("#fff2df", "#ab660c"),
    "eval": ("#f0eafe", "#7153a1"),
    "neutral": ("#f1f4f8", "#516174"),
}
matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "svg.fonttype": "path",
    "svg.hashsalt": "route-rl-training-method-v1",
})


def canvas(title, subtitle, *, height=10):
    fig, ax = plt.subplots(figsize=(16, height), facecolor="white")
    ax.set_xlim(0, 16)
    ax.set_ylim(0, height)
    ax.axis("off")
    ax.text(.45, height - .48, title, fontsize=23, fontweight="bold", color="#182638")
    ax.text(.45, height - .89, subtitle, fontsize=11, color="#516174")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    return fig, ax


def box(ax, x, y, width, height, title, body, *, color="neutral", size=11):
    fill, edge = COLORS[color]
    ax.add_patch(FancyBboxPatch((x, y), width, height,
                               boxstyle="round,pad=0.025,rounding_size=.12",
                               facecolor=fill, edgecolor=edge, linewidth=1.4, zorder=2))
    ax.text(x + .17, y + height - .25, title, va="top", fontsize=size + 1.5,
            color=edge, fontweight="bold", zorder=3)
    ax.text(x + .17, y + height - .67, body, va="top", fontsize=size,
            color="#24364b", linespacing=1.5, zorder=3)


def arrow(ax, start, end, *, label=None, curve=0, color="#6b7889", dashed=False):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=17,
                                connectionstyle=f"arc3,rad={curve}", color=color,
                                linewidth=1.6, linestyle="--" if dashed else "-", zorder=1))
    if label:
        ax.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + .13, label,
                ha="center", fontsize=9.5, color=color, zorder=4,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 2})


def overview():
    fig, ax = canvas("From public replays to a reviewed candidate",
                     "Owned pipeline v0.7.0 | Training objective: terminal win / draw / loss = 1 / 0.5 / 0",
                     height=10)
    xs = [.5, 4.3, 8.1, 11.9]
    cards = [
        ("1  Collect replays", "Official Kaggle API\nFrozen teacher snapshot\nActual seats and checksums", "data"),
        ("2  Validate & encode", "720 states / 719 decisions\nWhole-game train / holdout\nFeatures + executed labels", "data"),
        ("3  Behavior cloning", "CE + entropy + initial KL\nStreaming trajectory windows\nActor + trunk train; value fixed", "bc"),
        ("4  Freeze a BC policy", "Byte-stable checkpoint\nBind to its BC run\nFixed PPO KL teacher", "bc"),
    ]
    for x, (title, body, color) in zip(xs, cards):
        box(ax, x, 6.85, 3.35, 1.75, title, body, color=color)
    for x in xs[:-1]:
        arrow(ax, (x + 3.36, 7.7), (x + 3.77, 7.7))
    lower = [
        ("8  Confirm independently", "New seeds; same frozen files\nPaired score comparison\nCandidate only; human review", "eval"),
        ("7  Paired screening", "Candidate + baseline\nSame opponent; swap seats\nPositive terminal-score delta", "eval"),
        ("6  Rust self-play + PPO", "2 x games inference batch\nEconomic supports + masks\nGAE / clip / teacher KL / value", "rl"),
        ("5  Fit the critic", "Freeze actor + shared trunk\nTrain only the value head\nIndependent terminal-score MSE", "rl"),
    ]
    for x, (title, body, color) in zip(xs, lower):
        box(ax, x, 3.9, 3.35, 1.9, title, body, color=color)
    arrow(ax, (13.57, 6.8), (13.57, 5.88), label="frozen actor / BC teacher")
    for x in xs[:-1]:
        arrow(ax, (x + 3.77, 4.85), (x + 3.36, 4.85))
    box(ax, .5, 1.0, 5.1, 1.75, "9  Review, bundle & deploy",
        "Choose an actual saved policy\nExport inference code + required search binary\nAccepted deployments stay under manual control", color="eval")
    arrow(ax, (2.17, 3.86), (2.17, 2.83))
    box(ax, 7.2, 1.0, 8.05, 1.75, "Optional teacher cycle: separate diagnostic seeds",
        "Verified replay seed -> full planner self-play -> actual replay labels\nProtect all holdout / screening / confirmation seeds\nMix demonstrations -> fresh BC -> fresh critic -> fresh PPO", color="data")
    arrow(ax, (9.9, 2.82), (9.9, 3.83), label="fresh learning cycle", dashed=True)
    ax.text(.5, .42,
            "Optional final-day controller: explicit day-29 handoff; search actions do not enter actor loss.",
            fontsize=11, color="#516174")
    return fig


def architecture():
    fig, ax = canvas("Shared entity Transformer: actor + critic",
                     "Current bootstrap: 6 blocks, width 256 | Optional 10m preset: 12 blocks, same width",
                     height=11)
    box(ax, .45, 7.65, 4.0, 2.05, "A  Public observation + history",
        "Own / opponent board and units\nPublic market, clock and money\nRule-based inventory tracker\nNo private opponent inventory", color="data")
    box(ax, .45, 3.8, 4.0, 3.15, "B  Fixed entity representation",
        "1 GLOBAL + 200 CELL\nUp to 40 UNIT + 12 PRODUCT\n1 MEMORY + 10 MARKET_SLOT\n264 tokens x 124 features\nAbsent unit slots use token masks", color="data", size=12)
    arrow(ax, (2.45, 7.57), (2.45, 7.04))
    box(ax, 5.05, 7.45, 5.05, 2.25, "C  Six typed input adapters",
        "Select each entity type's features\nLinear projection -> width 256\nInput LayerNorm + token mask\nMEMORY is tracker state, not an RNN", color="bc")
    arrow(ax, (4.5, 5.4), (5.0, 8.2), curve=.15)
    box(ax, 5.05, 3.15, 5.05, 3.5, "D  Pre-LN residual blocks x 6",
        "LayerNorm -> 8-head self-attention\nSame-farm 2D RoPE (16 dims)\nResidual connection\nLayerNorm -> FFN 256 -> 1024 -> 256\nGELU + residual connection\nFinal LayerNorm", color="bc", size=12)
    arrow(ax, (7.57, 7.37), (7.57, 6.73))
    heads = [
        (8.0, "E  Unit action head", "Own unit embeddings\n20 slots x 500 logits", "bc"),
        (5.15, "F  Market + SELL heads", "10 market-slot embeddings\nLegacy 1,075 + SELL 9 x 100\nCompose -> 10 x 1,903 logits", "bc"),
        (2.3, "G  Critic value head", "Global token -> MLP 256 -> 1\nPublic cash / time input term\nSigmoid -> expected score [0,1]", "rl"),
    ]
    for y, title, body, color in heads:
        box(ax, 10.85, y, 4.65, 1.9, title, body, color=color)
        arrow(ax, (10.15, 4.9), (10.8, y + .9), curve=.08 if y > 6 else -.08)
    box(ax, .45, .65, 9.65, 1.65, "What changes during each training stage?",
        "BC: actor + trunk update; value fixed   |   Critic warmup: value only\nPPO: actor + trunk + value update; the BC KL teacher stays frozen",
        color="neutral", size=11)
    ax.text(10.9, .9, "Global attention still connects both farms.\nPartitioned RoPE is not sparse attention.\nAction supports are applied after logits.",
            fontsize=10.5, color="#516174", linespacing=1.6)
    return fig


def rollout():
    fig, ax = canvas("One collected decision, one consistent PPO update",
                     "Store the action actually executed, its conditional support, and only the neural probability factors",
                     height=9)
    xs = [.45, 4.35, 8.25, 12.15]
    stages = [
        ("1  Observe in Rust", "Independent seat histories\nFeatures + legal resources\n2 x games -> one JAX call", "data"),
        ("2  Sample a prefix", "Unit slots, then market slots\nLegal + economic supports\nForced SELL / DROP marked", "bc"),
        ("3  Execute & record", "Actual request -> next state\nIDs, supports, policy masks\nOld joint log probability", "data"),
        ("4  Finish the game", "719 decisions per seat\nTerminal score 1 / 0.5 / 0\nGAE -> advantages + returns", "rl"),
    ]
    for x, (title, body, color) in zip(xs, stages):
        box(ax, x, 5.35, 3.35, 1.95, title, body, color=color, size=10.5)
    for x in xs[:-1]:
        arrow(ax, (x + 3.4, 6.3), (x + 3.84, 6.3))
    box(ax, .45, 1.45, 4.5, 2.65, "6  Neural policy loss",
        "Recompute on stored supports\nratio = exp(new logp - old logp)\nClipped advantage objective\nFrozen teacher KL + entropy\nOnly policy-mask=True factors", color="bc", size=12)
    box(ax, 5.7, 1.45, 4.5, 2.65, "5  Same recorded batch",
        "Check execution / controller ID\nCheck old log-probability parity\nNormalize neural-row advantages\nKeep critic rows even if external\nNo probability for patched actions", color="neutral", size=11.5)
    box(ax, 10.95, 1.45, 4.5, 2.65, "7  Critic loss",
        "Sigmoid value predicts score\nHuber(value, GAE return)\nReal terminal outcome retained\nForced / search rows included\nNo cash auxiliary reward", color="rl", size=12)
    arrow(ax, (13.83, 5.27), (9.7, 4.18), curve=-.1)
    arrow(ax, (5.65, 2.78), (5.0, 2.78))
    arrow(ax, (10.25, 2.78), (10.9, 2.78))
    ax.text(.5, .6,
            "Final-day search and fully forced rows: actor excluded, critic included; neural fallback is explicitly recorded.",
            fontsize=11, color="#516174")
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "docs/images")
    parser.add_argument("--preview-dir", type=Path)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.preview_dir:
        args.preview_dir.mkdir(parents=True, exist_ok=True)
    for name, make in [("training_pipeline", overview), ("model_architecture", architecture),
                       ("ppo_execution", rollout)]:
        fig = make()
        fig.savefig(args.out_dir / f"{name}.svg", metadata={"Date": None})
        if args.preview_dir:
            fig.savefig(args.preview_dir / f"{name}.png", dpi=110)
        plt.close(fig)
        print(args.out_dir / f"{name}.svg")


if __name__ == "__main__":
    main()
