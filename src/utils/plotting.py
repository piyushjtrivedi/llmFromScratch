import time

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for Gradio and servers
import matplotlib.pyplot as plt
import numpy as np


def plot_loss_curves(metrics: dict):
    """
    Return a matplotlib Figure from a saved metrics dict.

    Renders a 2×3 grid:
      top-left:     Train / Val loss curves with epoch markers
      top-centre:   Val − Train gap (overfitting monitor)
      top-right:    Learning rate schedule
      bottom-left:  Gradient norm (pre-clip) with clipped region shaded
      bottom-centre: Peak memory usage with GPU utilisation %
      bottom-right: Throughput (tokens/sec) or step-time series

    Every panel with sufficient data also renders a self-diagnosing annotation
    that flags common problems (overfitting, constant clipping, low GPU
    utilisation, low throughput) and suggests a concrete fix.

    Backward-compatible: metric JSONs that pre-date the new keys
    (tokens_per_sec, grad_clip_norm, gpu_memory_total_gb, step_times_sec)
    still produce a valid figure — panels without data show a 'No data'
    placeholder.

    Required keys : train_losses, val_losses, tokens_seen, model_name
    Optional keys : learning_rates, grad_norms, peak_memory_gb,
                    tokens_per_sec, grad_clip_norm, gpu_memory_total_gb,
                    step_times_sec, num_epochs, batch_size,
                    gradient_accumulation_steps, learning_rate,
                    execution_time_minutes
    """
    # ── Unpack metrics ────────────────────────────────────────────────────────
    tokens_seen    = metrics["tokens_seen"]
    train_losses   = metrics["train_losses"]
    val_losses     = metrics["val_losses"]
    model_name     = metrics.get("model_name", "")

    learning_rates = metrics.get("learning_rates", [])
    grad_norms     = metrics.get("grad_norms", [])
    peak_memory_gb = metrics.get("peak_memory_gb", [])
    step_times     = metrics.get("step_times_sec", [])

    clip_ceiling       = metrics.get("grad_clip_norm", 0.5)
    gpu_total_gb       = metrics.get("gpu_memory_total_gb")        # None on MPS/CPU
    tokens_per_sec     = metrics.get("tokens_per_sec")
    num_epochs         = metrics.get("num_epochs")
    batch_size         = metrics.get("batch_size")
    grad_accum         = metrics.get("gradient_accumulation_steps")
    peak_lr            = metrics.get("learning_rate")
    exec_time          = metrics.get("execution_time_minutes")

    # ── Derived observations ─────────────────────────────────────────────────
    gap_series = [v - t for t, v in zip(train_losses, val_losses)]

    # Overfitting: gap in last quarter vs gap in first quarter
    q = max(1, len(gap_series) // 4)
    gap_early = sum(gap_series[:q]) / q
    gap_late  = sum(gap_series[-q:]) / q
    gap_widening = gap_late > gap_early * 1.5 and gap_late > 0.15

    # Constant clipping: fraction of steps where pre-clip norm > ceiling
    if grad_norms:
        clipped_frac = sum(1 for g in grad_norms if g > clip_ceiling) / len(grad_norms)
    else:
        clipped_frac = 0.0

    # Val loss plateau: range over last 10 eval steps < 0.02
    val_plateau = (
        len(val_losses) > 10
        and (max(val_losses[-10:]) - min(val_losses[-10:])) < 0.02
    )

    # GPU under-utilisation
    low_gpu = (
        gpu_total_gb is not None
        and peak_memory_gb
        and (max(peak_memory_gb) / gpu_total_gb) < 0.4
    )

    # Low throughput: heuristic baseline of 200 tok/s as minimum for any GPU
    low_throughput = tokens_per_sec is not None and tokens_per_sec < 200

    # Epoch boundary x-positions (vertical guide lines)
    epoch_boundaries = []
    if num_epochs and num_epochs > 1:
        step = len(tokens_seen) // num_epochs
        for e in range(1, num_epochs):
            idx = e * step
            if idx < len(tokens_seen):
                epoch_boundaries.append(tokens_seen[idx])

    # ── Figure setup ─────────────────────────────────────────────────────────
    BG       = "#0f1117"
    PANEL_BG = "#1a1d27"
    GRID_COL = "#2d3148"
    TICK_COL = "#9ca3af"
    WARN_COL = "#fbbf24"
    WARN_BOX = dict(boxstyle="round,pad=0.3", facecolor="#292524", alpha=0.88)

    fig = plt.figure(figsize=(17, 11))
    fig.patch.set_facecolor(BG)

    # Build run-summary subtitle from whichever keys are present
    parts = []
    if batch_size and grad_accum:
        parts.append(f"eff. batch = {batch_size * grad_accum}")
    if peak_lr:
        parts.append(f"lr = {peak_lr:.0e}")
    if num_epochs:
        parts.append(f"{num_epochs} epochs")
    if exec_time:
        parts.append(f"{exec_time:.1f} min")
    if tokens_per_sec:
        parts.append(f"{tokens_per_sec:.0f} tok/s")
    subtitle = "  ·  ".join(parts)

    fig.suptitle(
        f"Training diagnostics — {model_name}\n{subtitle}",
        fontsize=13, color="white", y=0.98, fontfamily="monospace",
    )

    axes = fig.subplots(2, 3)

    for ax in axes.flat:
        ax.set_facecolor(PANEL_BG)
        ax.tick_params(colors=TICK_COL, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(GRID_COL)
        ax.grid(True, alpha=0.15, color="#4b5563")

    def _style_ax(ax, title, xlabel="Tokens seen", ylabel=None):
        ax.set_title(title, color="white", fontsize=10, pad=6)
        ax.set_xlabel(xlabel, color=TICK_COL, fontsize=8)
        if ylabel:
            ax.set_ylabel(ylabel, color=TICK_COL, fontsize=8)

    def _epoch_lines(ax):
        for xb in epoch_boundaries:
            ax.axvline(x=xb, color="#6b7280", lw=0.8, ls=":", alpha=0.6)

    def _end_label(ax, x, y, text, color, dy=6):
        ax.annotate(
            text, xy=(x, y), xytext=(-45, dy),
            textcoords="offset points",
            color=color, fontsize=8, fontweight="bold",
        )

    def _warn(ax, msg, y_frac=0.85):
        ax.text(
            0.03, y_frac, msg, transform=ax.transAxes,
            color=WARN_COL, fontsize=7, bbox=WARN_BOX,
            verticalalignment="top",
        )

    # ── Panel 1 — Loss curves ────────────────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(tokens_seen, train_losses, label="Train", color="#60a5fa", lw=2)
    ax.plot(tokens_seen, val_losses,   label="Val",   color="#fb923c", lw=2, ls="--")
    _epoch_lines(ax)
    _style_ax(ax, "Loss", ylabel="Cross-entropy loss")
    ax.legend(fontsize=8, facecolor=GRID_COL, labelcolor="white", framealpha=0.8)
    _end_label(ax, tokens_seen[-1], train_losses[-1], f"{train_losses[-1]:.3f}", "#60a5fa")
    _end_label(ax, tokens_seen[-1], val_losses[-1],   f"{val_losses[-1]:.3f}",   "#fb923c", dy=-12)
    if gap_widening:
        _warn(ax, "⚠ overfitting gap widening\n→ try more data or early stopping")
    if val_plateau:
        _warn(ax, "⚠ val loss plateaued\n→ consider stopping here", y_frac=0.15)

    # ── Panel 2 — Val−Train gap ──────────────────────────────────────────────
    ax = axes[0, 1]
    ax.plot(tokens_seen, gap_series, color="#a78bfa", lw=2)
    ax.axhline(y=0, color="#6b7280", lw=0.8, ls="--", alpha=0.5)
    ax.fill_between(
        tokens_seen, gap_series, 0,
        where=[g > 0 for g in gap_series],
        color="#a78bfa", alpha=0.15,
    )
    _epoch_lines(ax)
    _style_ax(ax, "Val − Train Gap  (Overfitting Monitor)", ylabel="Loss gap")
    _end_label(ax, tokens_seen[-1], gap_series[-1], f"gap={gap_series[-1]:.3f}", "#a78bfa")
    if gap_widening:
        _warn(ax, f"⚠ late gap ({gap_late:.3f}) > 1.5× early gap ({gap_early:.3f})\n"
                  "→ dataset likely too small (try Dolly-15k)")

    # ── Panel 3 — Learning rate ──────────────────────────────────────────────
    ax = axes[0, 2]
    if learning_rates:
        ax.plot(tokens_seen, learning_rates, color="#34d399", lw=2)
        _epoch_lines(ax)
        _style_ax(ax, "Learning Rate Schedule", ylabel="LR")
        ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        ax.yaxis.offsetText.set_color(TICK_COL)
        _end_label(ax, tokens_seen[-1], learning_rates[-1],
                   f"{learning_rates[-1]:.1e}", "#34d399")
        # Flag if LR never warmed up (flat line = warmup may not have fired)
        lr_range = max(learning_rates) - min(learning_rates)
        if lr_range < max(learning_rates) * 0.05:
            _warn(ax, "⚠ LR barely changed\n→ check scheduler is stepping")
    else:
        _no_data(ax, "Learning Rate Schedule")

    # ── Panel 4 — Gradient norm ──────────────────────────────────────────────
    ax = axes[1, 0]
    if grad_norms:
        ax.plot(tokens_seen, grad_norms, color="#f87171", lw=1.5,
                alpha=0.9, label="Grad norm (pre-clip)")
        ax.axhline(y=clip_ceiling, color="#f87171", lw=1, ls=":",
                   alpha=0.5, label=f"Clip ceiling ({clip_ceiling})")
        # Shade steps where clipping was active
        ax.fill_between(
            tokens_seen, grad_norms, clip_ceiling,
            where=[g > clip_ceiling for g in grad_norms],
            color="#f87171", alpha=0.18, label="Clipped",
        )
        _style_ax(ax, "Gradient Norm  (pre-clip)", ylabel="L2 norm")
        ax.legend(fontsize=7, facecolor=GRID_COL, labelcolor="white", framealpha=0.8)
        if clipped_frac > 0.8:
            _warn(ax,
                  f"⚠ clipped on {clipped_frac*100:.0f}% of steps\n"
                  "→ try halving the learning rate")
        elif clipped_frac > 0.3:
            _warn(ax,
                  f"⚠ clipped on {clipped_frac*100:.0f}% of steps\n"
                  "→ consider a slightly lower LR")
    else:
        _no_data(ax, "Gradient Norm")

    # ── Panel 5 — Peak memory ────────────────────────────────────────────────
    ax = axes[1, 1]
    if peak_memory_gb and any(v > 0 for v in peak_memory_gb):
        ax.plot(tokens_seen, peak_memory_gb, color="#818cf8", lw=2)
        if gpu_total_gb:
            ax.axhline(y=gpu_total_gb, color="#818cf8", lw=1, ls=":",
                       alpha=0.4, label=f"GPU total ({gpu_total_gb:.0f} GB)")
            util_pct = max(peak_memory_gb) / gpu_total_gb * 100
            util_msg = f"Peak GPU util: {util_pct:.0f}%"
            if low_gpu:
                _warn(ax, f"⚠ {util_msg} — GPU under-utilised\n"
                          "→ increase --batch-size")
            else:
                ax.text(0.03, 0.88, util_msg, transform=ax.transAxes,
                        color="#818cf8", fontsize=7, bbox=WARN_BOX,
                        verticalalignment="top")
            ax.legend(fontsize=7, facecolor=GRID_COL, labelcolor="white",
                      framealpha=0.8)
        _style_ax(ax, "Peak Memory Usage", ylabel="GB")
        _end_label(ax, tokens_seen[-1], peak_memory_gb[-1],
                   f"{peak_memory_gb[-1]:.2f} GB", "#818cf8")
    else:
        _no_data(ax, "Peak Memory Usage")

    # ── Panel 6 — Throughput ─────────────────────────────────────────────────
    ax = axes[1, 2]
    if step_times:
        # step_times_sec has one entry per optimizer step; tokens_seen has one entry
        # per eval point (every eval_freq steps). Using step index as x-axis keeps
        # alignment correct — pairing against tokens_seen would discard ~80% of data.
        step_indices = list(range(len(step_times)))
        ax.plot(step_indices, step_times, color="#fbbf24", lw=1.5, alpha=0.9)
        mean_t = float(np.mean(step_times))
        ax.axhline(y=mean_t, color="#fbbf24", lw=1, ls="--",
                   alpha=0.5, label=f"mean = {mean_t:.1f}s")
        _style_ax(ax, "Step Time", xlabel="Optimizer step", ylabel="Seconds / optimizer step")
        ax.legend(fontsize=7, facecolor=GRID_COL, labelcolor="white", framealpha=0.8)
        if low_throughput:
            _warn(ax, f"⚠ {tokens_per_sec:.0f} tok/s — expected 400+\n"
                      "→ increase --batch-size")
    elif tokens_per_sec is not None:
        # Summary card when per-step times weren't recorded
        colour = WARN_COL if low_throughput else "#34d399"
        ax.text(0.5, 0.52, f"{tokens_per_sec:.1f}",
                transform=ax.transAxes, ha="center", va="center",
                color=colour, fontsize=36, fontweight="bold",
                fontfamily="monospace")
        ax.text(0.5, 0.36, "tokens / sec",
                transform=ax.transAxes, ha="center", va="center",
                color=TICK_COL, fontsize=10)
        total_tok = tokens_seen[-1] if tokens_seen else None
        detail_parts = []
        if total_tok:
            detail_parts.append(f"{total_tok:,} tokens total")
        if exec_time:
            detail_parts.append(f"{exec_time:.1f} min")
        if detail_parts:
            ax.text(0.5, 0.24, "  ·  ".join(detail_parts),
                    transform=ax.transAxes, ha="center", va="center",
                    color=TICK_COL, fontsize=8)
        if low_throughput:
            ax.text(0.5, 0.10,
                    "⚠ low — increase --batch-size",
                    transform=ax.transAxes, ha="center", va="center",
                    color=WARN_COL, fontsize=8,
                    bbox=WARN_BOX)
        _style_ax(ax, "Throughput")
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        _no_data(ax, "Throughput")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


def _no_data(ax, title: str) -> None:
    """Render a placeholder panel when data is absent."""
    ax.set_title(title, color="white", fontsize=10, pad=6)
    ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
            ha="center", va="center", color="#6b7280", fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#2d3148")