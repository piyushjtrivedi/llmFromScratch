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
    lora_rank          = metrics.get("lora_rank")
    lora_alpha         = metrics.get("lora_alpha")
    bertscore_f1       = metrics.get("bertscore_f1")
    rouge_l            = metrics.get("rougeL")

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
    if lora_rank is not None:
        alpha_str = f"  α={lora_alpha:.0f}" if lora_alpha is not None else ""
        parts.append(f"LoRA r={lora_rank}{alpha_str}")
    if exec_time:
        parts.append(f"{exec_time:.1f} min")
    if tokens_per_sec:
        parts.append(f"{tokens_per_sec:.0f} tok/s")
    subtitle = "  ·  ".join(parts)

    eval_parts = []
    if bertscore_f1 is not None: eval_parts.append(f"BERTScore F1={bertscore_f1:.3f}")
    if rouge_l      is not None: eval_parts.append(f"ROUGE-L={rouge_l:.3f}")
    eval_line  = ("  ·  ".join(eval_parts)) if eval_parts else ""
    title_body = f"Training diagnostics — {model_name}\n{subtitle}"
    if eval_line:
        title_body += f"\n{eval_line}"

    fig.suptitle(title_body, fontsize=13, color="white", y=0.98, fontfamily="monospace")

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
        # Filter timing artifacts: epoch-boundary sample-generation time can leak
        # into the next step's measurement in older metrics (before the trainer fix).
        # Any value > 3× the median is an isolated spike, not a real slow step.
        median_t = float(np.median(step_times))
        threshold = max(median_t * 3, median_t + 2.0)
        clean_times = [t for t in step_times if t <= threshold]
        filtered = len(step_times) - len(clean_times)

        # step_times_sec has one entry per optimizer step; tokens_seen has one entry
        # per eval point (every eval_freq steps). Using step index as x-axis keeps
        # alignment correct — pairing against tokens_seen would discard ~80% of data.
        step_indices = list(range(len(clean_times)))
        ax.plot(step_indices, clean_times, color="#fbbf24", lw=1.5, alpha=0.9)
        mean_t = float(np.mean(clean_times))
        ax.axhline(y=mean_t, color="#fbbf24", lw=1, ls="--",
                   alpha=0.5, label=f"mean = {mean_t:.2f}s")
        if filtered:
            ax.text(0.97, 0.95, f"{filtered} spike(s) filtered",
                    transform=ax.transAxes, ha="right", va="top",
                    color=TICK_COL, fontsize=7)
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

    plt.tight_layout(rect=[0, 0, 1, 0.91 if eval_line else 0.94])
    return fig


def plot_comparison_table(metrics_by_model: dict) -> plt.Figure:
    """
    Render a styled master comparison table (LoRA vs Full for all models).
    Matches the screenshot layout: Model | Mode | Best Val | Perplexity |
    Best Step | Time | Tok/s | Peak Mem | GPU Util
    """
    BG       = "#0f1117"
    HDR_BG   = "#1e3a5f"
    ROW_A    = "#111827"
    ROW_B    = "#1a1d27"
    C_LORA   = "#22d3ee"
    C_FULL   = "#fb923c"
    C_GREEN  = "#4ade80"
    C_AMBER  = "#fbbf24"
    C_MUTED  = "#9ca3af"
    WHITE    = "#f9fafb"

    def _short(name: str) -> str:
        if "(" in name:
            base, size = name.split("(", 1)
            base_clean = base.strip().replace('gpt2', 'GPT2').replace('gemma3', 'Gemma3')
            return f"{base_clean} ({size.rstrip(')')})"
        return name.replace("gemma3", "Gemma3").replace("gpt2", "GPT2")

    headers   = ["Model", "Mode", "Best Val↓", "Perplexity↓",
                 "Best Step", "Time", "Tok/s", "Peak Mem", "GPU Util", "BERTScore↑"]
    col_w     = [0.12, 0.07, 0.09, 0.09, 0.10, 0.07, 0.07, 0.09, 0.08, 0.10]

    rows, meta = [], []
    for model_name, variants in metrics_by_model.items():
        for mode_key, mode_label in [("lora", "LoRA"), ("full", "Full")]:
            m = variants.get(mode_key)
            if m and m.get("val_losses"):
                best_idx  = int(np.argmin(m["val_losses"]))
                total     = len(m["val_losses"])
                best_val  = float(min(m["val_losses"]))
                perp      = float(np.exp(best_val))
                step_sym  = "✓" if best_idx == total - 1 else "△"
                mem_list  = m.get("peak_memory_gb") or []
                peak_mem  = max(mem_list) if mem_list else None
                gpu_total = m.get("gpu_memory_total_gb")
                exec_t    = m.get("execution_time_minutes")
                tps       = m.get("tokens_per_sec")

                bs_f1 = m.get("bertscore_f1")
                rows.append([
                    _short(model_name),
                    mode_label,
                    f"{best_val:.3f}",
                    f"{perp:.3f}",
                    f"{best_idx+1}/{total}  {step_sym}",
                    f"{exec_t:.1f}m" if isinstance(exec_t, (int, float)) else "—",
                    str(int(tps)) if tps else "—",
                    f"{peak_mem:.1f} GB" if peak_mem else "—",
                    f"{peak_mem/gpu_total*100:.0f}%" if (peak_mem and gpu_total) else "—",
                    f"{bs_f1:.3f}" if bs_f1 is not None else "—",
                ])
                meta.append({"mode": mode_key, "model": model_name,
                             "best_val": best_val, "step_sym": step_sym,
                             "bertscore_f1": bs_f1})
            else:
                rows.append([_short(model_name), mode_label] + ["—"] * 8)
                meta.append({"mode": mode_key, "model": model_name,
                             "best_val": None, "step_sym": None, "bertscore_f1": None})

    n_rows   = len(rows)
    fig_h    = max(3.5, 1.6 + n_rows * 0.72 + 1.0)
    fig      = plt.figure(figsize=(16, fig_h))
    fig.patch.set_facecolor(BG)

    # Build subtitle from shared config keys if they exist
    all_m    = [v[k] for v in metrics_by_model.values()
                for k in ("full", "lora") if v.get(k)]
    parts    = []
    if all_m:
        m0 = all_m[0]
        if m0.get("batch_size"):       parts.append(f"bs={m0['batch_size']}")
        if m0.get("learning_rate"):    parts.append(f"lr={m0['learning_rate']:.0e}")
        if m0.get("num_epochs"):       parts.append(f"ep={m0['num_epochs']}")
        if m0.get("gradient_accumulation_steps") and m0["gradient_accumulation_steps"] > 1:
            parts.append(f"grad_accum={m0['gradient_accumulation_steps']}")
    lora_m = next((v["lora"] for v in metrics_by_model.values()
                   if v.get("lora") and v["lora"].get("lora_rank") is not None), None)
    if lora_m:
        alpha_str = f"  α={lora_m['lora_alpha']:.0f}" if lora_m.get("lora_alpha") is not None else ""
        parts.append(f"LoRA r={lora_m['lora_rank']}{alpha_str}")
    subtitle = "  ".join(parts)

    fig.text(0.5, 0.97, "LoRA vs Full Fine-tuning — Master Comparison",
             ha="center", va="top", fontsize=13, color=WHITE, fontfamily="monospace")
    if subtitle:
        fig.text(0.5, 0.93, subtitle, ha="center", va="top",
                 fontsize=9, color=C_MUTED, fontfamily="monospace")

    ax = fig.add_subplot(111)
    ax.set_facecolor(BG)
    ax.axis("off")

    tbl = ax.table(cellText=rows, colLabels=headers,
                   colWidths=col_w, loc="upper center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 2.1)

    # ── Header styling ────────────────────────────────────────────────────────
    for c in range(len(headers)):
        cell = tbl[(0, c)]
        cell.set_facecolor(HDR_BG)
        cell.set_text_props(color=WHITE, fontweight="bold")
        cell.set_edgecolor("#2d3148")

    # ── Data row styling ──────────────────────────────────────────────────────
    # Identify which rows belong to the same model for pairwise highlighting
    model_pairs: dict[str, dict] = {}
    for r_idx, m in enumerate(meta):
        mn = m["model"]
        if mn not in model_pairs:
            model_pairs[mn] = {}
        model_pairs[mn][m["mode"]] = r_idx

    better_cells: set[tuple] = set()
    for mn, pair in model_pairs.items():
        lora_r = pair.get("lora")
        full_r = pair.get("full")
        if lora_r is not None and full_r is not None:
            lv = meta[lora_r]["best_val"]
            fv = meta[full_r]["best_val"]
            if lv is not None and fv is not None:
                better_r = lora_r if lv < fv else full_r
                better_cells.add((better_r + 1, 2))
                better_cells.add((better_r + 1, 3))
            # BERTScore F1 — higher is better (col 9)
            lb = meta[lora_r].get("bertscore_f1")
            fb = meta[full_r].get("bertscore_f1")
            if lb is not None and fb is not None:
                better_cells.add(((lora_r if lb > fb else full_r) + 1, 9))

    for r_idx, (row_data, m) in enumerate(zip(rows, meta)):
        tbl_row = r_idx + 1
        bg      = ROW_A if r_idx % 2 == 0 else ROW_B

        for c in range(len(headers)):
            cell = tbl[(tbl_row, c)]
            cell.set_facecolor(bg)
            cell.set_text_props(color=WHITE)
            cell.set_edgecolor("#1f2937")

        # Mode cell — coloured text
        mode_cell = tbl[(tbl_row, 1)]
        if m["mode"] == "lora":
            mode_cell.set_text_props(color=C_LORA, fontweight="bold")
        else:
            mode_cell.set_text_props(color=C_FULL, fontweight="bold")

        # Best Step cell — colour by ✓ / △
        step_cell = tbl[(tbl_row, 4)]
        if m["step_sym"] == "✓":
            step_cell.set_text_props(color=C_LORA)
        elif m["step_sym"] == "△":
            step_cell.set_text_props(color=C_AMBER)

        # Best Val, Perplexity, BERTScore — highlight the winner
        for c in (2, 3, 9):
            if (tbl_row, c) in better_cells:
                tbl[(tbl_row, c)].set_text_props(color=C_GREEN, fontweight="bold")

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_y = 0.03
    legend_items = [
        ("■", C_LORA,  "LoRA"),
        ("■", C_FULL,  "Full Fine-tuning"),
        ("✓", C_LORA,  "still learning — could train longer"),
        ("△", C_AMBER, "early stopping recommended"),
        ("●", C_GREEN, "better result"),
    ]
    x = 0.04
    for sym, col, label in legend_items:
        fig.text(x, legend_y, sym, color=col, fontsize=10, va="bottom")
        fig.text(x + 0.018, legend_y, label, color=C_MUTED, fontsize=8, va="bottom")
        x += 0.18

    plt.tight_layout(rect=[0, 0.06, 1, 0.90])
    return fig


def plot_comparison(metrics_by_model: dict) -> plt.Figure:
    """
    Side-by-side LoRA vs Full comparison across models.

    Args:
        metrics_by_model: {
            "gpt2-small (124M)": {"full": dict_or_None, "lora": dict_or_None},
            "gemma3-1b":         {"full": dict_or_None, "lora": dict_or_None},
            ...
        }

    Layout:
        One row per model  — left panel = Full, right panel = LoRA loss curves
        Final row          — master comparison table (all models × variants)
    """
    BG        = "#0f1117"
    PANEL_BG  = "#1a1d27"
    GRID_COL  = "#2d3148"
    TICK_COL  = "#9ca3af"
    C_TRAIN   = "#60a5fa"
    C_VAL     = "#fb923c"
    C_WIN_BG   = "#0d2818"   # subtle dark-green tint for winning cell
    C_WIN_TEXT = "#4ade80"   # bright green text for winning value

    models = list(metrics_by_model.keys())
    n      = len(models)

    fig = plt.figure(figsize=(16, 4 * n + 3))
    fig.patch.set_facecolor(BG)
    fig.suptitle("Model Comparison — LoRA vs Full Fine-tune",
                 fontsize=14, color="white", y=0.99)

    # Build a subtitle from the first available metrics (shared config)
    _sub_parts = []
    _any_m = next((v[k] for v in metrics_by_model.values()
                   for k in ("full", "lora") if v.get(k)), None)
    if _any_m:
        if _any_m.get("learning_rate"):  _sub_parts.append(f"lr={_any_m['learning_rate']:.0e}")
        if _any_m.get("num_epochs"):     _sub_parts.append(f"ep={_any_m['num_epochs']}")
        if _any_m.get("batch_size"):     _sub_parts.append(f"bs={_any_m['batch_size']}")
    _lora_m = next((v["lora"] for v in metrics_by_model.values()
                    if v.get("lora") and v["lora"].get("lora_rank") is not None), None)
    if _lora_m:
        _alpha_str = f"  α={_lora_m['lora_alpha']:.0f}" if _lora_m.get("lora_alpha") is not None else ""
        _sub_parts.append(f"LoRA r={_lora_m['lora_rank']}{_alpha_str}")
    if _sub_parts:
        fig.text(0.5, 0.965, "  ·  ".join(_sub_parts),
                 ha="center", va="top", fontsize=9, color="#9ca3af", fontfamily="monospace")

    gs = fig.add_gridspec(n + 1, 2,
                          height_ratios=[1.0] * n + [0.80 + 0.15 * n],
                          hspace=0.55, wspace=0.28)

    def _prep_ax(ax, title):
        ax.set_facecolor(PANEL_BG)
        ax.tick_params(colors=TICK_COL, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(GRID_COL)
        ax.grid(True, alpha=0.15, color="#4b5563")
        ax.set_title(title, color="white", fontsize=9, pad=5)
        ax.set_xlabel("Tokens seen", color=TICK_COL, fontsize=8)
        ax.set_ylabel("Loss", color=TICK_COL, fontsize=8)

    def _loss_panel(ax, m, title):
        if m is None:
            ax.set_facecolor(PANEL_BG)
            for spine in ax.spines.values():
                spine.set_color(GRID_COL)
            ax.set_title(title, color="white", fontsize=9, pad=5)
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center", color="#6b7280", fontsize=11)
            ax.set_xticks([]); ax.set_yticks([])
            return
        _prep_ax(ax, title)
        ts = m["tokens_seen"]
        ax.plot(ts, m["train_losses"], color=C_TRAIN, lw=2, label="Train")
        ax.plot(ts, m["val_losses"],   color=C_VAL,   lw=2, ls="--", label="Val")
        ax.legend(fontsize=7, facecolor=GRID_COL, labelcolor="white", framealpha=0.7)
        final_val = m["val_losses"][-1]
        ax.annotate(f"val={final_val:.3f}", xy=(ts[-1], final_val),
                    xytext=(-38, 6), textcoords="offset points",
                    color=C_VAL, fontsize=7)

    # ── Loss curve rows ───────────────────────────────────────────────────────
    for row, model_name in enumerate(models):
        variants = metrics_by_model[model_name]
        short    = model_name.split(" ")[0] if "(" in model_name else model_name
        _loss_panel(fig.add_subplot(gs[row, 0]), variants.get("full"),
                    f"{short} — Full fine-tune")
        _loss_panel(fig.add_subplot(gs[row, 1]), variants.get("lora"),
                    f"{short} — LoRA")

    # ── Master comparison table ───────────────────────────────────────────────
    ax_t = fig.add_subplot(gs[n, :])
    ax_t.set_facecolor(PANEL_BG)
    ax_t.set_title("Master Comparison", color="white", fontsize=10, pad=6)
    ax_t.axis("off")

    C_LORA_T  = "#22d3ee"
    C_FULL_T  = "#fb923c"
    C_AMBER_T = "#fbbf24"

    headers  = ["Model", "Variant", "Final Val↓", "Best Val↓", "Perplexity↓",
                "Best Step", "Time", "Tok/s", "Peak Mem", "GPU Util", "BERTScore↑"]
    col_w_t  = [0.10, 0.08, 0.09, 0.09, 0.09, 0.10, 0.07, 0.07, 0.09, 0.07, 0.09]

    def _extract(m):
        if m is None:
            return ["—"] * 9
        best_idx  = int(np.argmin(m["val_losses"]))
        total     = len(m["val_losses"])
        best_val  = float(min(m["val_losses"]))
        step_sym  = "✓" if best_idx == total - 1 else "△"
        mem_list  = m.get("peak_memory_gb") or []
        peak_mem  = max(mem_list) if mem_list else None
        gpu_total = m.get("gpu_memory_total_gb")
        exec_t    = m.get("execution_time_minutes")
        tps       = m.get("tokens_per_sec")
        bs_f1     = m.get("bertscore_f1")
        return [
            f"{m['val_losses'][-1]:.4f}",
            f"{best_val:.4f}",
            f"{np.exp(best_val):.3f}",
            f"{best_idx+1}/{total}  {step_sym}",
            f"{exec_t:.1f}m" if isinstance(exec_t, (int, float)) else "—",
            str(int(tps)) if tps else "—",
            f"{peak_mem:.1f} GB" if peak_mem else "—",
            f"{peak_mem/gpu_total*100:.0f}%" if (peak_mem and gpu_total) else "—",
            f"{bs_f1:.3f}" if bs_f1 is not None else "—",
        ]

    def _num(s):
        try:
            return float(s.replace(" GB", "").replace("%", "").rstrip("m"))
        except (ValueError, AttributeError):
            return None

    rows_data = []
    for model_name in models:
        short = model_name.split(" ")[0] if "(" in model_name else model_name
        v     = metrics_by_model[model_name]
        rows_data.append([short, "Full"] + _extract(v.get("full")))
        rows_data.append([short, "LoRA"] + _extract(v.get("lora")))

    tbl = ax_t.table(
        cellText=rows_data,
        colLabels=headers,
        colWidths=col_w_t,
        loc="upper center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 1.6)

    # Header styling
    for col in range(len(headers)):
        cell = tbl[(0, col)]
        cell.set_facecolor("#1e3a5f")
        cell.set_text_props(color="white", fontweight="bold")
        cell.set_edgecolor("#2d3148")

    # col indices: 0=Model 1=Variant 2=FinalVal 3=BestVal 4=Perplexity
    #              5=BestStep 6=Time 7=Tok/s 8=PeakMem 9=GPUUtil 10=BERTScore
    metric_cols  = [2, 3, 4, 6, 7, 8, 10]
    lower_better = {2, 3, 4, 6, 8}      # losses, perplexity, time, memory
    higher_better = {7, 10}             # tok/s, BERTScore

    for pair_idx in range(len(models)):
        full_row = pair_idx * 2 + 1   # 1-indexed (0 = header)
        lora_row = pair_idx * 2 + 2

        for data_row in (full_row, lora_row):
            for col in range(len(headers)):
                cell = tbl[(data_row, col)]
                cell.set_facecolor("#111827" if data_row % 2 else PANEL_BG)
                cell.set_text_props(color="white", fontweight="normal")
                cell.set_edgecolor("#1f2937")

        # Mode cell colour
        tbl[(full_row, 1)].set_text_props(color=C_FULL_T, fontweight="bold")
        tbl[(lora_row, 1)].set_text_props(color=C_LORA_T, fontweight="bold")

        # Best Step symbol colour
        for data_row in (full_row, lora_row):
            step_val = rows_data[data_row - 1][5]
            if "✓" in step_val:
                tbl[(data_row, 5)].set_text_props(color=C_LORA_T)
            elif "△" in step_val:
                tbl[(data_row, 5)].set_text_props(color=C_AMBER_T)

        # Win/loss highlighting
        for col in metric_cols:
            fval = rows_data[full_row - 1][col]
            lval = rows_data[lora_row - 1][col]
            if fval == "—" or lval == "—":
                continue
            fv, lv = _num(fval), _num(lval)
            if fv is None or lv is None:
                continue
            f_wins = fv < lv if col in lower_better else fv > lv
            win_row = full_row if f_wins else lora_row
            tbl[(win_row, col)].set_facecolor(C_WIN_BG)
            tbl[(win_row, col)].set_text_props(color=C_WIN_TEXT, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
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