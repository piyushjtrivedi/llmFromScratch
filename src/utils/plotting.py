import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for Gradio and servers
import matplotlib.pyplot as plt


def plot_loss_curves(metrics: dict):
    """
    Return a matplotlib Figure from a saved metrics dict.

    Renders a 2×2 grid:
      top-left:     Train / Val loss
      top-right:    Learning rate schedule
      bottom-left:  Peak memory (GB)
      bottom-right: Gradient norm (pre-clip)

    The three supplementary panels are only drawn when their data is present,
    so metric JSONs from before these fields were added still produce a valid
    figure (the empty panels display a 'No data' notice).

    Required keys: train_losses, val_losses, tokens_seen, model_name
    Optional keys: learning_rates, grad_norms, peak_memory_gb
    """
    tokens_seen   = metrics["tokens_seen"]
    train_losses  = metrics["train_losses"]
    val_losses    = metrics["val_losses"]
    model_name    = metrics.get("model_name", "")

    learning_rates  = metrics.get("learning_rates", [])
    grad_norms      = metrics.get("grad_norms", [])
    peak_memory_gb  = metrics.get("peak_memory_gb", [])

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f"Training diagnostics — {model_name}", fontsize=13)

    # ── Top-left: Loss ───────────────────────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(tokens_seen, train_losses, label="Train loss",
            color="#1f77b4", linewidth=2)
    ax.plot(tokens_seen, val_losses,   label="Val loss",
            color="#ff7f0e", linewidth=2, linestyle="--")
    ax.set_xlabel("Tokens seen")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.annotate(f"{train_losses[-1]:.3f}", xy=(tokens_seen[-1], train_losses[-1]),
                xytext=(8, 0), textcoords="offset points",
                color="#1f77b4", fontsize=9)
    ax.annotate(f"{val_losses[-1]:.3f}", xy=(tokens_seen[-1], val_losses[-1]),
                xytext=(8, 0), textcoords="offset points",
                color="#ff7f0e", fontsize=9)

    # ── Top-right: Learning rate ──────────────────────────────────────────────
    ax = axes[0, 1]
    if learning_rates:
        ax.plot(tokens_seen, learning_rates, color="#2ca02c", linewidth=2)
        ax.set_xlabel("Tokens seen")
        ax.set_ylabel("Learning rate")
        ax.set_title("Learning Rate Schedule")
        ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        ax.grid(True, alpha=0.3)
        ax.annotate(f"{learning_rates[-1]:.2e}",
                    xy=(tokens_seen[-1], learning_rates[-1]),
                    xytext=(8, 0), textcoords="offset points",
                    color="#2ca02c", fontsize=9)
    else:
        _no_data(ax, "Learning Rate Schedule")

    # ── Bottom-left: Peak memory ─────────────────────────────────────────────
    ax = axes[1, 0]
    if peak_memory_gb and any(v > 0 for v in peak_memory_gb):
        ax.plot(tokens_seen, peak_memory_gb, color="#9467bd", linewidth=2)
        ax.set_xlabel("Tokens seen")
        ax.set_ylabel("Memory (GB)")
        ax.set_title("Peak Memory Usage")
        ax.grid(True, alpha=0.3)
        ax.annotate(f"{peak_memory_gb[-1]:.2f} GB",
                    xy=(tokens_seen[-1], peak_memory_gb[-1]),
                    xytext=(8, 0), textcoords="offset points",
                    color="#9467bd", fontsize=9)
    else:
        _no_data(ax, "Peak Memory Usage")

    # ── Bottom-right: Gradient norm ──────────────────────────────────────────
    ax = axes[1, 1]
    if grad_norms:
        ax.plot(tokens_seen, grad_norms, color="#d62728", linewidth=1.5,
                alpha=0.85, label="Grad norm (pre-clip)")
        # Draw the clipping ceiling so it's obvious when norms are being cut
        ax.axhline(y=0.5, color="#d62728", linewidth=1, linestyle=":",
                   alpha=0.5, label="Clip ceiling (0.5)")
        ax.set_xlabel("Tokens seen")
        ax.set_ylabel("L2 norm")
        ax.set_title("Gradient Norm (pre-clip)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    else:
        _no_data(ax, "Gradient Norm")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def _no_data(ax, title: str):
    """Fill a subplot with a placeholder when its data is absent."""
    ax.set_title(title)
    ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
            ha="center", va="center", color="grey", fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
