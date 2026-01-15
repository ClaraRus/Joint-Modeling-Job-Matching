import os
import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from src.plot.constants import custom_palette, marker_dict
from src.plot.read_results import read_results_file, read_eval_results_methods
from src.plot.format_naming import format_naming
from src.plot.method_types import add_method_type
# =========================
# Results generation
# =========================
def generate_results_file(root_dir, core, k):
    metrics = [
        ["DGI", "DGI(global)", "DUP"],
        ["ndcg", "precision", "recall"],
        ["coverage", "gini"],
        [
            "compatibility_score", "compatibility_added_de", "compatibility_added_non-de",
            "compatibility_removed_de", "compatibility_removed_non-de",
            "compatibility_added_0", "compatibility_added_1",
            "compatibility_removed_0", "compatibility_removed_1"
        ]
    ]
    results = read_eval_results_methods(root_dir, core, metrics, k)
    results.to_csv(os.path.join(root_dir, f"core-{core}", f"evaluation_results_core{core}_K{k}.csv"))


# =========================
# Plot helpers
# =========================

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
from constants import (
    METHOD_PALETTE, BAR_PLOT_PALETTE, ALL_METRICS, 
    ALL_FAIRNESS_METRICS, UTILITY_METRIC_LABELS, 
    get_metric_label, FUSION_TYPES, USER_GROUPS,
    add_method_type, format_naming, get_best_params
)

# =========================
# Plot helpers
# =========================
def init_plot(plots_per_row, metrics, num_rows=None):
    num_rows = num_rows or int(np.ceil(len(metrics) / plots_per_row))
    fig, axes = plt.subplots(
        num_rows, plots_per_row,
        figsize=(3 * plots_per_row, 4 * num_rows),
        sharex=False, sharey=False
    )
    axes = axes.flatten().tolist() if isinstance(axes, np.ndarray) else [axes]
    return fig, axes

def create_legend(fig, axes):
    axes_list = np.ravel(np.atleast_1d(axes)).tolist()
    empty_axes, non_empty_axes = [], []

    for ax in axes_list:
        if not (ax.lines or ax.patches or ax.collections or ax.images or ax.containers):
            empty_axes.append(ax)
        else:
            non_empty_axes.append(ax)

    legend_ax_pos = empty_axes[-1].get_position() if empty_axes else None
    for ax in empty_axes:
        fig.delaxes(ax)

    handles, labels = [], []
    for ax in non_empty_axes:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
        if ax.get_legend():
            ax.get_legend().remove()

    # Deduplicate while preserving order
    unique = {l: h for h, l in zip(handles, labels)}
    labels_sorted = sorted(unique.keys())
    handles_sorted = [unique[l] for l in labels_sorted]

    if legend_ax_pos:
        fig.legend(
            handles_sorted, labels_sorted,
            title="Method", fontsize=20, title_fontsize=17,
            frameon=False, loc="center",
            bbox_to_anchor=(legend_ax_pos.x0 + legend_ax_pos.width / 2,
                            legend_ax_pos.y0 + legend_ax_pos.height / 2),
            bbox_transform=fig.transFigure, ncol=1,
            handletextpad=0.4, columnspacing=0.8
        )
    fig.tight_layout()

def save_fig(core, file_name, base_dir="./"):
    plt.savefig(os.path.join(base_dir, f"core-{core}/plots", file_name), bbox_inches="tight")  

# =========================
# Results plotting
# =========================
def plot_results_ax(fig, ax, results, metric_x, metric_y, params=False):
    results = results.copy()
    params_num = pd.to_numeric(results["params"], errors="coerce")

    # Line plot: regular methods
    mask_line = (params_num <= 10) & (~params_num.isna()) & (~results["method"].str.contains("CFP"))
    mask_line |= results["method"].str.contains("FA*IR", regex=False)
    df_line = results[mask_line].reset_index(drop=True)
    if not df_line.empty:
        sns.lineplot(
            data=df_line, x=metric_x, y=metric_y, hue="method",
            palette=METHOD_PALETTE, ax=ax, estimator=None, errorbar=None
        )

    # Line plot: dotted (CFP)
    mask_dotted = (params_num <= 10) & (~params_num.isna()) & results["method"].str.contains("CFP")
    df_dotted = results[mask_dotted]
    if not df_dotted.empty:
        sns.lineplot(
            data=df_dotted, x=metric_x, y=metric_y, hue="method",
            palette=METHOD_PALETTE, linestyle="dotted", ax=ax,
            estimator=None, errorbar=None
        )

    # Scatter for non-param methods
    if not params:
        mask_scatter = (
            results["method"].str.contains("2Sided - fair") & results["params"].isna()
        )
        mask_scatter |= results["method"].str.contains("TFROM|userfairness|BPR") | results["params"].isna()
        df_scatter = results[mask_scatter]
        if not df_scatter.empty:
            df_scatter = df_scatter.fillna(0.5)
            sns.scatterplot(
                data=df_scatter, x=metric_x, y=metric_y, hue="method",
                palette=METHOD_PALETTE, style="method", s=200, ax=ax
            )

    # Axis labels
    def format_label(label):
        return get_metric_label(label)
    ax.set_xlabel(format_label(metric_x), fontsize=23)
    ax.set_ylabel(format_label(metric_y), fontsize=23)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=15)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=15)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.2f}"))
    ax.tick_params(axis="x", which="major", pad=15)
    ax.tick_params(axis="y", which="major", pad=15)
    plt.tight_layout()

# =========================
# Add expanded params
# =========================
def add_params_expanded(results):
    param_values = np.arange(0, 1.1, 0.1)
    subset = results[results["params"].isna()]
    expanded = subset.loc[subset.index.repeat(len(param_values))].assign(
        params=param_values.tolist() * len(subset)
    )
    return expanded

# =========================
# Plot metric vs α
# =========================
def plot_metric_vs_alpha(root_dir, core, k=10, rank_fusion=False):
    results = pd.read_csv(os.path.join(root_dir, f"core-{core}", f"evaluation_results_core{core}_K{k}.csv"))
    results = add_method_type(results)
    results = format_naming(results)

    # Baselines and two-sided
    results_baselines = add_params_expanded(results[results["method"].isin(["BPR"])])
    results_2sided = results[results["method"].str.contains("TSF")]

    # Fusion methods
    results_rank_fusion = results[results["method_type"] == "fusion_rank"]
    results_rank_fusion_score = add_params_expanded(results[results["method_type"] == "fusion_rank_score"])

    # Fair fusion optim
    results_fair_optim = results[results["method_type"] == "fusion_learn"]
    results_fair_optim["α"] = results_fair_optim["params"]

    results_plot = pd.concat([results_2sided, results_baselines])
    if rank_fusion:
        results_plot = pd.concat([results_plot, results_rank_fusion, results_rank_fusion_score])

    # Metrics
    fair_metrics = [m for m in ALL_FAIRNESS_METRICS if "global" not in m]
    other_metrics = [m for m in ALL_METRICS if m not in fair_metrics]
    metrics = fair_metrics + other_metrics

    fig, axes = init_plot(len(fair_metrics), metrics, num_rows=2)
    results_plot["α"] = results_plot["params"]

    for i, metric in enumerate(metrics):
        df_plot = pd.concat([results_plot, results_fair_optim])
        df_plot = format_naming(df_plot)
        df_plot["α"] = pd.to_numeric(df_plot["α"], errors="coerce")
        plot_results_ax(fig, axes[i], df_plot, "α", metric, params=True)

    create_legend(fig, axes)
    file_name = f"2sided_alpha_vs_base_metric{'_rank_fusion' if rank_fusion else ''}_K{k}.pdf"
    save_fig(core, file_name)

# =========================
# Compatibility bar plots
# =========================

def plot_compatibility_diff_analysis(root_dir, core, k):
    """
    Plot compatibility added/removed per user/item group,
    for methods including fair_fusion and baselines.
    """
    results = read_results_file(root_dir, core, k)
    results = add_method_type(results)

    # Filter 2-sided non-optim methods
    mask_2sided = (
        results["method"].str.contains("2sided") &
        ~results["method"].str.contains("optim") &
        (results["method_type"] != "fusion_rank") &
        ~results["method"].str.contains("ATT") &
        (results["method_type"] != "fusion_rank_score")
    )

    # Keep all other methods except UserJob
    mask_keep = ~mask_2sided & (results["method"] != "UserJob")

    # Select fair fusion optimization methods
    mask_fair_optim = results["method"].str.contains("2sided") & results["method"].str.contains("fair_fusion_optim")
    results_fair_optim = results[mask_fair_optim]
    methods_fair_fusion_optim = [
        "2sided_fair_fusion_optim__DGI__country",
        "2sided_fair_fusion_optim__DGI__is_payed",
        "2sided_fair_fusion_optim__DUP__country"
    ]
    results_fair_optim = results_fair_optim[results_fair_optim["method"].isin(methods_fair_fusion_optim)]

    # Select best params and combine
    results = get_best_params(results, core, k)
    results = pd.concat([results[mask_keep], results_fair_optim])
    results = format_naming(results, group_flag=None)

    print("RESULTS TO PLOT")
    print(results["method"].unique())

    # --- Define metrics to plot ---
    metrics = [
        "compatibility_added_0","compatibility_added_1",
        "compatibility_added_de","compatibility_added_non-de",
        "compatibility_removed_0","compatibility_removed_1",
        "compatibility_removed_de","compatibility_removed_non-de"
    ]

    # --- Melt into long format ---
    plot_df = results.melt(
        id_vars=["method"],
        value_vars=metrics,
        var_name="metric", value_name="value"
    )

    # Extract type (added/removed) and attribute
    plot_df["type"] = plot_df["metric"].apply(lambda x: "added" if "added" in x else "removed")
    plot_df["attribute"] = plot_df["metric"].str.replace("compatibility_added_", "").str.replace("compatibility_removed_", "")

    dict_map_gr = {
        "1": "Premium",
        "0": "Non-premium",
        "de": "German",
        "non-de": "Non-German"
    }
    plot_df["attribute"] = plot_df["attribute"].apply(lambda x: dict_map_gr[str(x)])

    # --- Plotting ---
    attribute_order = ['Non-German', 'German', 'Non-premium', 'Premium']
    method_order = [
        "FA*IR(country)", "FA*IR(premium)", "FA*IR(country,premium)",
        "userfairness(country)", "userfairness(premium)",
        "PCT", "CP-Fair(country)", "CP-Fair(premium)",
        "TFROM(country)", "TFROM(premium)", "TFROM(country,premium)",
        "TSF", "TSF - ATT"
    ]
    type_colors = {"added": "#2E86AB", "removed": "#F26457"}

    plot_df["attribute"] = pd.Categorical(plot_df["attribute"], categories=attribute_order, ordered=True)
    plot_df["method"] = pd.Categorical(plot_df["method"], categories=method_order, ordered=True)

    # Separate added/removed
    removed_data = plot_df[plot_df["type"] == "removed"]
    added_data = plot_df[plot_df["type"] == "added"]

    fig, ax = plt.subplots(figsize=(10, 3))
    sns.despine(ax=ax)
    bar_width = 0.8 / len(method_order)

    # Plot added bars
    sns.barplot(
        data=added_data,
        x="attribute", y="value",
        hue="method",
        order=attribute_order,
        dodge=True,
        alpha=0.5,
        palette=custom_palette_bar_plot,
        edgecolor="black",
        ax=ax,
    )

    # Hatch pattern for removed bars
    from matplotlib.patches import Rectangle
    hatch_pattern = ".."
    methods = list(removed_data["method"].cat.categories)

    for i_attr, attr in enumerate(attribute_order):
        for j_meth, method in enumerate(methods):
            add_row = added_data[(added_data["attribute"] == attr) & (added_data["method"] == method)]
            rem_row = removed_data[(removed_data["attribute"] == attr) & (removed_data["method"] == method)]
            if add_row.empty and rem_row.empty:
                continue
            add_val = add_row["value"].values[0] if not add_row.empty else 0
            rem_val = rem_row["value"].values[0] if not rem_row.empty else 0
            color = custom_palette_bar_plot[method]
            x = i_attr + j_meth * bar_width - 0.4 + bar_width / 2 - 0.03
            rect = Rectangle(
                (x, 0),
                bar_width,
                rem_val,
                facecolor=color,
                alpha=0.5,
                hatch=hatch_pattern,
                linewidth=1.0,
                zorder=3
            )
            ax.add_patch(rect)

    # Fix x-ticks and labels
    ax.set_xticks(range(len(attribute_order)))
    ax.set_xticklabels(attribute_order)
    ax.set_xlabel("")
    ax.set_ylabel("Compatibility")

    # --- Legend ---
    import matplotlib.patches as mpatches
    type_patches = [
        mpatches.Patch(facecolor="gray", edgecolor="black", hatch=hatch_pattern, label="Removed"),
        mpatches.Patch(facecolor="gray", edgecolor="black", alpha=0.8, label="Added")
    ]
    ax.legend(handles=type_patches, title="Type", loc="upper right", frameon=True)

    # Method legend
    custom_order = [
        "TSF", "TSF - ATT", "PCT",
        "TFROM(country)", "TFROM(premium)", "TFROM(country,premium)",
        "FA*IR(country)", "FA*IR(premium)", "FA*IR(country,premium)",
        "CP-Fair(country)", "CP-Fair(premium)",
        "userfairness(country)", "userfairness(premium)"
    ]
    unique_methods = [m for m in custom_order if m in plot_df["method"].unique()]
    method_patches = [mpatches.Patch(color=custom_palette_bar_plot[m], label=m) for m in unique_methods]
    fig.legend(
        handles=method_patches,
        title="Method",
        loc="lower center",
        ncol=5,
        frameon=True,
        bbox_to_anchor=(0.5, -0.17),
        title_fontsize=13
    )

    plt.ylim(0.05, 0.4)
    plt.subplots_adjust(bottom=0.9)
    plt.tight_layout()

    file_name = f"analysis_core{core}_compatibility_overlap_K{k}.png"
    save_fig(core, file_name)


# =========================
# Main execution
# =========================
data_dir = "./DATA"
plot_group_dist_recommendations(data_dir, "20", k=10)
generate_results_file(data_dir, "20", k=10)
plot_metric_vs_alpha(data_dir, "20", k=10, rank_fusion=False)
plot_metric_vs_alpha(data_dir, "20", k=10, rank_fusion=True)
plot_compatibility_analysis(data_dir, "20", k=10)
