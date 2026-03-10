"""Unit 8 -- Bar Plot Rendering."""

from pathlib import Path
from dataclasses import dataclass
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np

from gsea_tool.configuration import FisherConfig, PlotAppearanceConfig
from gsea_tool.meta_analysis import FisherResult
from gsea_tool.go_clustering import ClusteringResult


@dataclass
class BarPlotResult:
    """Metadata about the rendered bar plot figure, for notes.md consumption."""
    pdf_path: Path
    png_path: Path
    svg_path: Path
    n_bars: int
    n_mutants: int
    clustering_was_used: bool


def select_bar_data(
    fisher_result: FisherResult,
    clustering_result: ClusteringResult | None,
    top_n: int,
) -> tuple[list[str], list[float], list[int]]:
    """Select GO terms, p-values, and contributing counts for the bar plot.

    Returns:
        term_names: Display names for Y-axis labels.
        neg_log_pvalues: -log10(combined p-value) for X-axis.
        n_contributing: Number of contributing mutant lines for color encoding.
    All lists are ordered by combined p-value (most significant first).
    """
    if clustering_result is not None:
        # Clustering mode: use representatives (already sorted by combined p-value)
        if len(clustering_result.representatives) == 0:
            raise ValueError("No terms to plot: no representative GO terms from clustering")

        n = min(top_n, len(clustering_result.representatives))
        go_ids = clustering_result.representatives[:n]
        names = clustering_result.representative_names[:n]
        pvalues = clustering_result.representative_pvalues[:n]
        contrib = clustering_result.representative_n_contributing[:n]

        term_names = list(names)
        neg_log_pvalues = [-math.log10(p) if p > 0 else 300.0 for p in pvalues]
        n_contributing = list(contrib)
    else:
        # No clustering: use top N GO terms by combined p-value
        if len(fisher_result.combined_pvalues) == 0:
            raise ValueError("No terms to plot: no GO terms exist")

        # Sort GO terms by combined p-value ascending
        sorted_terms = sorted(
            fisher_result.combined_pvalues.items(),
            key=lambda x: x[1],
        )

        n = min(top_n, len(sorted_terms))
        sorted_terms = sorted_terms[:n]

        term_names = []
        neg_log_pvalues = []
        n_contributing = []

        for go_id, pval in sorted_terms:
            name = fisher_result.go_id_to_name.get(go_id, go_id)
            term_names.append(name)
            neg_log_pvalues.append(-math.log10(pval) if pval > 0 else 300.0)
            n_contributing.append(fisher_result.n_contributing.get(go_id, 0))

    return term_names, neg_log_pvalues, n_contributing


def render_bar_plot(
    fisher_result: FisherResult,
    clustering_result: ClusteringResult | None,
    fisher_config: FisherConfig,
    plot_config: PlotAppearanceConfig,
    output_dir: Path,
    output_stem: str = "figure3_meta_analysis",
) -> BarPlotResult:
    """Render the meta-analysis bar plot and save to PDF, PNG, and SVG.

    If clustering_result is provided, uses representative terms.
    If clustering_result is None, uses top N terms by combined p-value.

    Returns BarPlotResult with paths and summary counts.
    """
    # Pre-conditions
    assert output_dir.is_dir(), "Output directory must exist"
    assert fisher_config.top_n_bars > 0, "top_n_bars must be positive"

    # Select data
    term_names, neg_log_pvalues, n_contributing = select_bar_data(
        fisher_result, clustering_result, fisher_config.top_n_bars
    )

    n_bars = len(term_names)

    # Truncate long labels
    max_len = plot_config.label_max_length
    display_names = []
    for name in term_names:
        if len(name) > max_len:
            display_names.append(name[:max_len - 3] + "...")
        else:
            display_names.append(name)

    # Set up the colormap for number of contributing lines
    cmap = plt.get_cmap(plot_config.bar_colormap)
    contrib_array = np.array(n_contributing)
    vmin = contrib_array.min() if len(contrib_array) > 0 else 0
    vmax = contrib_array.max() if len(contrib_array) > 0 else 1
    if vmin == vmax:
        vmax = vmin + 1
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    colors = [cmap(norm(c)) for c in n_contributing]

    # Create figure
    fig, ax = plt.subplots(
        figsize=(plot_config.bar_figure_width, plot_config.bar_figure_height)
    )

    # Set font
    plt.rcParams["font.family"] = plot_config.font_family

    # Bars are ordered most significant at top, so reverse for bottom-to-top plotting
    y_positions = list(range(n_bars))
    # Reverse so most significant is at top
    reversed_names = list(reversed(display_names))
    reversed_values = list(reversed(neg_log_pvalues))
    reversed_colors = list(reversed(colors))
    reversed_contrib = list(reversed(n_contributing))

    bars = ax.barh(y_positions, reversed_values, color=reversed_colors, edgecolor="none")

    ax.set_yticks(y_positions)
    ax.set_yticklabels(reversed_names)
    ax.set_xlabel("-log$_{10}$(combined p-value)")

    # Significance line
    if plot_config.show_significance_line:
        sig_line = -math.log10(0.05)
        ax.axvline(x=sig_line, color="gray", linestyle="--", linewidth=1, zorder=0)

    # Annotate bars with number of contributing lines
    if plot_config.show_recurrence_annotation:
        for i, (bar, count) in enumerate(zip(bars, reversed_contrib)):
            width = bar.get_width()
            ax.text(
                width + 0.05,
                bar.get_y() + bar.get_height() / 2,
                str(count),
                ha="left",
                va="center",
                fontsize=9,
            )

    # Colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Number of contributing lines")

    # Clean, minimal styling
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    plt.tight_layout()

    # Save files
    pdf_path = output_dir / f"{output_stem}.pdf"
    png_path = output_dir / f"{output_stem}.png"
    svg_path = output_dir / f"{output_stem}.svg"

    try:
        fig.savefig(str(pdf_path), format="pdf", dpi=plot_config.dpi, bbox_inches="tight")
        fig.savefig(str(png_path), format="png", dpi=plot_config.dpi, bbox_inches="tight")
        fig.savefig(str(svg_path), format="svg", dpi=plot_config.dpi, bbox_inches="tight")
    except OSError as e:
        raise OSError(f"Cannot write figure files to {output_dir}: {e}") from e
    finally:
        plt.close(fig)

    # Post-conditions
    assert pdf_path.exists(), "PDF file must be written"
    assert png_path.exists(), "PNG file must be written"
    assert svg_path.exists(), "SVG file must be written"

    clustering_was_used = clustering_result is not None

    result = BarPlotResult(
        pdf_path=pdf_path,
        png_path=png_path,
        svg_path=svg_path,
        n_bars=n_bars,
        n_mutants=fisher_result.n_mutants,
        clustering_was_used=clustering_was_used,
    )

    assert result.n_bars > 0, "At least one bar must be plotted"
    assert result.n_bars <= fisher_config.top_n_bars, "Number of bars cannot exceed top_n_bars"

    return result
