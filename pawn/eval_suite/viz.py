"""Seaborn/matplotlib visualization helpers for the evaluation suite."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import seaborn as sns

# Consistent style
sns.set_theme(style="darkgrid", context="notebook", palette="muted")
FIGSIZE = (10, 5)
FIGSIZE_WIDE = (14, 5)
FIGSIZE_TALL = (10, 8)


def _get_figure(fig: plt.Figure | None, ax: Axes) -> plt.Figure:
    """Return the figure, preferring an explicitly-created one over ax.figure."""
    if fig is not None:
        return fig
    assert isinstance(ax.figure, plt.Figure)
    return ax.figure


# Grid-PAWN baselines (Run 2, layer 7)
GRID_PAWN_BASELINES = {
    "piece_type": 0.902,
    "side_to_move": 0.9998,
    "is_check": 0.961,
    "castling_rights": 0.980,
    "ep_square": 0.998,
}


# ---------------------------------------------------------------------------
# Corpus & bounds plots
# ---------------------------------------------------------------------------


def plot_game_length_distribution(stats: dict, ax: Axes | None = None) -> plt.Figure:
    """Histogram of game lengths."""
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=FIGSIZE)
    assert ax is not None
    counts = stats["game_length"]["histogram_counts"]
    edges = stats["game_length"]["histogram_edges"]
    centers = [(edges[i] + edges[i + 1]) / 2 for i in range(len(counts))]
    ax.bar(centers, counts, width=(edges[1] - edges[0]) * 0.9, color=sns.color_palette()[0], alpha=0.8)
    ax.axvline(stats["game_length"]["mean"], color="red", ls="--", label=f"Mean: {stats['game_length']['mean']:.1f}")
    ax.axvline(stats["game_length"]["median"], color="orange", ls="--", label=f"Median: {stats['game_length']['median']:.0f}")
    ax.set_xlabel("Game Length (ply)")
    ax.set_ylabel("Count")
    ax.set_title("Game Length Distribution")
    ax.legend()
    return _get_figure(fig, ax)


def plot_legal_move_distribution(bounds: dict, ax: Axes | None = None) -> plt.Figure:
    """Histogram of legal move counts (K) from pre-computed histogram data."""
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=FIGSIZE)
    assert ax is not None
    k_hist = bounds["k_histogram"]
    k_vals = np.array(k_hist["values"])
    k_counts = np.array(k_hist["counts"], dtype=np.float64)
    k_density = k_counts / k_counts.sum()
    mask = k_vals <= 80
    ax.bar(k_vals[mask], k_density[mask], width=0.9, color=sns.color_palette()[1], alpha=0.8)
    ax.axvline(bounds["k_stats"]["mean"], color="red", ls="--", label=f"Mean: {bounds['k_stats']['mean']:.1f}")
    ax.set_xlabel("Legal Move Count (K)")
    ax.set_ylabel("Density")
    ax.set_title("Legal Move Count Distribution")
    ax.set_xlim(0, 80)
    ax.legend()
    return _get_figure(fig, ax)


def plot_outcome_rates(stats: dict, ax: Axes | None = None) -> plt.Figure:
    """Bar chart of outcome base rates."""
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=FIGSIZE)
    assert ax is not None
    rates = stats["outcome_rates"]
    names = list(rates.keys())
    values = [rates[n] * 100 for n in names]
    bars = ax.barh(names, values, color=sns.color_palette("Set2", len(names)))
    for bar, v in zip(bars, values):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{v:.2f}%", va="center", fontsize=9)
    ax.set_xlabel("Rate (%)")
    ax.set_title("Outcome Base Rates (1M Random Games)")
    ax.invert_yaxis()
    return _get_figure(fig, ax)


def plot_k_by_phase(bounds: dict, ax: Axes | None = None) -> plt.Figure:
    """E[1/K] by game phase."""
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=FIGSIZE)
    assert ax is not None
    phase_data = bounds["phase_bounds"]
    names = list(phase_data.keys())
    e_inv_k = [phase_data[n]["e_1_over_k"] * 100 for n in names]
    mean_k = [phase_data[n]["mean_k"] for n in names]
    labels = [n.replace("_", " ").replace("ply ", "Ply ") for n in names]

    x = np.arange(len(names))
    ax.bar(x, e_inv_k, color=sns.color_palette()[2], alpha=0.8)
    for i, (v, mk) in enumerate(zip(e_inv_k, mean_k)):
        ax.text(i, v + 0.1, f"E[1/K]={v:.2f}%\nmean K={mk:.1f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("E[1/K] (%)")
    ax.set_title("Theoretical Top-1 Accuracy Ceiling by Game Phase")
    return _get_figure(fig, ax)


def plot_prefix_histogram(sanity: dict, ax: Axes | None = None) -> plt.Figure:
    """Histogram of common prefix lengths."""
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=FIGSIZE)
    assert ax is not None
    hist = sanity["prefix_length_histogram"]
    ks = sorted(hist.keys())
    vs = [hist[k] for k in ks]
    ax.bar(ks, vs, color=sns.color_palette()[3], alpha=0.8)
    ax.set_xlabel("Common Prefix Length (moves)")
    ax.set_ylabel("Count (adjacent pairs)")
    ax.set_title(f"Adjacent-Pair Common Prefix Lengths (max={sanity['max_prefix_moves']})")
    ax.set_yscale("log")
    return _get_figure(fig, ax)


# ---------------------------------------------------------------------------
# Probe plots
# ---------------------------------------------------------------------------


def plot_probe_heatmap(
    probe_results: dict,
    title: str = "Per-Layer Probe Accuracy",
    metric_key: str = "best_accuracy",
    figsize: tuple = (12, 6),
) -> plt.Figure:
    """Heatmap of probe accuracy across layers."""
    probe_names = list(probe_results.keys())
    layer_names = list(probe_results[probe_names[0]].keys())

    data = np.array([
        [probe_results[p][l].get(metric_key, 0) for l in layer_names]
        for p in probe_names
    ])

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        data, annot=True, fmt=".3f", cmap="YlOrRd", linewidths=0.5,
        ax=ax, vmin=0.0, vmax=1.0, cbar_kws={"label": metric_key},
        xticklabels=layer_names, yticklabels=probe_names,
    )
    ax.set_title(title)
    ax.set_ylabel("Probe")
    ax.set_xlabel("Layer")
    plt.tight_layout()
    return fig


def plot_probe_comparison(
    clm_results: dict,
    pawn_baselines: dict | None = None,
    figsize: tuple = (10, 5),
) -> plt.Figure:
    """Side-by-side comparison of CLM vs grid-PAWN probe results (final layer)."""
    if pawn_baselines is None:
        pawn_baselines = GRID_PAWN_BASELINES

    # Get CLM final-layer results
    shared_probes = [p for p in pawn_baselines if p in clm_results]
    clm_vals = []
    pawn_vals = []
    for p in shared_probes:
        layers = clm_results[p]
        # Use last layer
        last_layer = list(layers.keys())[-1]
        clm_vals.append(layers[last_layer]["best_accuracy"])
        pawn_vals.append(pawn_baselines[p])

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(shared_probes))
    w = 0.35
    ax.bar(x - w / 2, [v * 100 for v in pawn_vals], w, label="Grid-PAWN (Run 2)", color=sns.color_palette()[0])
    ax.bar(x + w / 2, [v * 100 for v in clm_vals], w, label="CLM-PAWN", color=sns.color_palette()[1])
    ax.set_xticks(x)
    ax.set_xticklabels(shared_probes, rotation=30, ha="right")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Probe Accuracy: Grid-PAWN vs CLM-PAWN (Final Layer)")
    ax.legend()
    ax.set_ylim(0, 105)
    for i, (pv, cv) in enumerate(zip(pawn_vals, clm_vals)):
        ax.text(i - w / 2, pv * 100 + 0.5, f"{pv:.1%}", ha="center", fontsize=8)
        ax.text(i + w / 2, cv * 100 + 0.5, f"{cv:.1%}", ha="center", fontsize=8)
    plt.tight_layout()
    return fig


def plot_probe_layer_profile(
    probe_results: dict,
    probe_name: str,
    figsize: tuple = (10, 4),
) -> plt.Figure:
    """Line plot of a single probe's accuracy across layers."""
    layers = probe_results[probe_name]
    layer_names = list(layers.keys())
    accs = [layers[l]["best_accuracy"] for l in layer_names]

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(range(len(layer_names)), accs, "o-", color=sns.color_palette()[0], linewidth=2)
    ax.set_xticks(range(len(layer_names)))
    ax.set_xticklabels(layer_names, rotation=30, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Layer Profile: {probe_name}")
    ax.grid(True, alpha=0.3)

    # Add grid-PAWN baseline if available
    if probe_name in GRID_PAWN_BASELINES:
        ax.axhline(GRID_PAWN_BASELINES[probe_name], color="red", ls="--",
                    label=f"Grid-PAWN: {GRID_PAWN_BASELINES[probe_name]:.3f}")
        ax.legend()

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Outcome signal test plots
# ---------------------------------------------------------------------------


def plot_outcome_signal_results(
    signal_results: dict,
    base_rates: dict,
    figsize: tuple = (14, 6),
) -> plt.Figure:
    """Bar chart comparing conditioned outcome match rates vs base rates."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    for ax, condition in zip(axes, ["unmasked", "masked"]):
        names = []
        match_rates = []
        base_rate_vals = []

        for outcome_name in signal_results:
            if condition not in signal_results[outcome_name]:
                continue
            names.append(outcome_name.replace("_", "\n"))
            match_rates.append(signal_results[outcome_name][condition]["outcome_match_rate"] * 100)
            base_rate_vals.append(base_rates.get(outcome_name, 0) * 100)

        x = np.arange(len(names))
        w = 0.35
        ax.bar(x - w / 2, base_rate_vals, w, label="Base Rate", color=sns.color_palette()[0], alpha=0.7)
        ax.bar(x + w / 2, match_rates, w, label="Conditioned", color=sns.color_palette()[1], alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=8)
        ax.set_ylabel("Match Rate (%)")
        ax.set_title(f"Outcome Match Rate ({condition})")
        ax.legend()

    plt.tight_layout()
    return fig


def plot_outcome_distributions(
    signal_results: dict,
    condition: str = "masked",
    figsize: tuple = (14, 8),
) -> plt.Figure:
    """Stacked bar chart showing actual outcome distributions for each conditioned outcome."""
    outcome_names = list(signal_results.keys())
    fig, axes = plt.subplots(len(outcome_names), 1, figsize=figsize, sharex=True)
    if len(outcome_names) == 1:
        axes = [axes]

    all_outcomes = set()
    for oname in outcome_names:
        if condition in signal_results[oname]:
            all_outcomes.update(signal_results[oname][condition]["outcome_distribution"].keys())
    all_outcomes = sorted(all_outcomes)
    colors = dict(zip(all_outcomes, sns.color_palette("Set2", len(all_outcomes))))

    for ax, oname in zip(axes, outcome_names):
        if condition not in signal_results[oname]:
            continue
        dist = signal_results[oname][condition]["outcome_distribution"]
        vals = [dist.get(o, 0) * 100 for o in all_outcomes]
        bars = ax.barh(all_outcomes, vals, color=[colors[o] for o in all_outcomes])
        ax.set_title(f"Conditioned: {oname}", fontsize=10)
        ax.set_xlim(0, 100)
        ax.set_xlabel("% of games")

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Diagnostic plots
# ---------------------------------------------------------------------------


def plot_diagnostic_results(
    diag_results: dict,
    figsize: tuple = (12, 6),
) -> plt.Figure:
    """Bar chart of diagnostic position metrics."""
    cats = list(diag_results.keys())
    legal_rates = [diag_results[c]["mean_legal_rate"] * 100 for c in cats]
    pad_probs = [diag_results[c]["mean_pad_prob"] * 100 for c in cats]
    entropies = [diag_results[c]["mean_entropy"] for c in cats]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Legal move rate
    ax = axes[0]
    bars = ax.barh(cats, legal_rates, color=sns.color_palette()[0], alpha=0.8)
    ax.set_xlabel("Legal Move Rate (%)")
    ax.set_title("Sampled Legal Move Rate")
    ax.set_xlim(0, 105)

    # PAD probability
    ax = axes[1]
    ax.barh(cats, pad_probs, color=sns.color_palette()[1], alpha=0.8)
    ax.set_xlabel("PAD Probability (%)")
    ax.set_title("PAD Token Probability")

    # Entropy
    ax = axes[2]
    ax.barh(cats, entropies, color=sns.color_palette()[2], alpha=0.8)
    ax.set_xlabel("Entropy (nats)")
    ax.set_title("Distribution Entropy")

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Lichess plots
# ---------------------------------------------------------------------------


def plot_lichess_comparison(
    lichess_results: dict,
    random_metrics: dict | None = None,
    figsize: tuple = (14, 5),
) -> plt.Figure:
    """Compare metrics across Elo bands."""
    bands = sorted(lichess_results.keys())
    metrics = ["loss", "top1_accuracy", "legal_move_rate"]
    titles = ["Loss", "Top-1 Accuracy", "Legal Move Rate"]

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    for ax, metric, title in zip(axes, metrics, titles):
        vals = [lichess_results[b][metric] for b in bands]
        labels = [lichess_results[b]["elo_range"] for b in bands]
        label_strs = [f"{lo}-{hi}" for lo, hi in labels]

        ax.bar(range(len(bands)), vals, color=sns.color_palette()[0], alpha=0.8)
        ax.set_xticks(range(len(bands)))
        ax.set_xticklabels(label_strs, rotation=30, ha="right")
        ax.set_title(title)
        ax.set_xlabel("Elo Band")

        if random_metrics and metric in random_metrics:
            ax.axhline(random_metrics[metric], color="red", ls="--", label="Random games")
            ax.legend()

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Prefix continuation plot
# ---------------------------------------------------------------------------


def plot_prefix_continuation(
    prefix_results: dict,
    base_rates: dict,
    figsize: tuple = (14, 8),
) -> plt.Figure:
    """Matrix of outcome match rates by conditioned outcome × prefix length."""
    outcome_names = list(prefix_results.keys())
    if not outcome_names:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig

    # Get bucket names from first outcome
    buckets = list(prefix_results[outcome_names[0]].keys())
    n_outcomes = len(outcome_names)
    n_buckets = len(buckets)

    fig, axes = plt.subplots(n_outcomes, 1, figsize=(14, 3 * n_outcomes), sharex=True)
    if n_outcomes == 1:
        axes = [axes]

    for ax, source_outcome in zip(axes, outcome_names):
        rates_by_cond = {}
        for bucket in buckets:
            if bucket not in prefix_results[source_outcome]:
                continue
            for cond_name in prefix_results[source_outcome][bucket]:
                if cond_name not in rates_by_cond:
                    rates_by_cond[cond_name] = []
                rates_by_cond[cond_name].append(
                    prefix_results[source_outcome][bucket][cond_name]["outcome_match_rate"] * 100
                )

        for cond_name, rates in rates_by_cond.items():
            ax.plot(range(len(rates)), rates, "o-", label=cond_name, markersize=4)

        if source_outcome in base_rates:
            ax.axhline(base_rates[source_outcome] * 100, color="gray", ls=":", alpha=0.5, label="base rate")

        ax.set_title(f"Source: {source_outcome}")
        ax.set_ylabel("Match Rate (%)")
        ax.legend(fontsize=7, ncol=3)
        ax.set_xticks(range(len(buckets)))
        ax.set_xticklabels(buckets, rotation=30, ha="right", fontsize=8)

    plt.tight_layout()
    return fig
