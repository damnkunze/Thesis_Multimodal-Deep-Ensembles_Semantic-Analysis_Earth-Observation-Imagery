import re

from plottable import ColumnDefinition, Table
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.patheffects as mpatheffects
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties

from src.base import *
from src.metrics import shannon_entropy

# ## Visualize Training

# with placeholder:
SAVE_TRAIN_PATH = PLOTS_SAVE_DIR + "train_{}.png"
SAVE_TEST_PATH = PLOTS_SAVE_DIR + "test_{}.png"
SAVE_FULL_AREA_TEST_PATH = PLOTS_SAVE_DIR + "full_area_test_{}.png"
SAVE_MODULATIONS_VIEW_PATH = PLOTS_SAVE_DIR + "modulations_view_{}.png"

if not os.path.exists(PLOTS_SAVE_DIR):
    os.makedirs(PLOTS_SAVE_DIR)

# For whole already trained Ensemble
def visualize_train_ensemble(ensemble, save=False, model_indices=None, show_title=True, highlight_best=True):
    """
    model_indices: explicit list of model indices (into ensemble.models) to
    show, eg. [0, 3, 6] - shown in the given order. Subplot titles still show
    each model's original ("Model {i+1}") number, not its position in this
    subset - same convention as visualize_ensemble_modulations().

    show_title: if False, hides the whole-grid suptitle (ensemble name +
    model count). Per-subplot titles are unaffected.

    highlight_best: if False, don't bold the lowest-validation-loss model's
    subplot title.
    """

    MAX_COLS = 1

    if model_indices is not None:
        selected_indices = list(model_indices)
    else:
        selected_indices = list(range(len(ensemble.models)))

    n = len(selected_indices)
    if n == 0:
        print("No models in ensemble.")
        return

    ncols = min(MAX_COLS, n)
    nrows = int(np.ceil(n / ncols))

    # Fixed width per column (6in, matching the ncols=3/figwidth=18 layout
    # this was originally tuned for) - a flat figwidth stayed just as wide
    # for ncols=1 as for ncols=3, which is both needlessly wide and, since
    # the right-side twin axis (DICE) is offset by a fraction of each
    # subplot's own width, made that offset's absolute size shrink relative
    # to the (unnecessarily wide) figure - scaling figwidth with ncols keeps
    # each subplot's own width (and therefore that offset) consistent
    col_width_in = 6
    figwidth = col_width_in * ncols

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(figwidth, 3 * nrows),
        squeeze=False
    )

    if show_title:
        fig.suptitle(f"{ensemble.name}\nTraining Curves for {n} Models", fontsize=16)

    legend_handles, legend_labels = None, None

    # Find model with lowest validation loss (among the selected models only)
    min_val_loss = float('inf')
    best_orig_idx = -1
    for orig_idx in selected_indices:
        stats = ensemble.models[orig_idx].train_stats
        if "train_stats" not in stats:
            val_losses = stats.get("val_losses", [])
        else:
            val_losses = stats["train_stats"][2]

        if val_losses:
            early_stopping_patience = stats.get("early_stopping_patience", EARLY_STOPPING_PATIENCE)
            stopped_epoch = len(val_losses) - early_stopping_patience
            val_loss = val_losses[stopped_epoch - 1]
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                best_orig_idx = orig_idx

    for pos in range(nrows * ncols):
        i, j = divmod(pos, ncols)
        ax1 = axes[i][j]

        # Hide unused subplot cells
        if pos >= n:
            ax1.axis("off")
            continue

        orig_idx = selected_indices[pos]
        stats = ensemble.models[orig_idx].train_stats

        # Not set with all models
        train_start_time = 0
        train_end_time = 0
        early_stopping_patience = EARLY_STOPPING_PATIENCE


        if "train_stats" not in stats: # New version
            # stats has:
            train_losses = stats["train_losses"]
            train_dcs = stats["train_dcs"]
            val_losses = stats["val_losses"]
            val_dcs = stats["val_dcs"]
            lrs = stats["lrs"]

            if "train_start_time" in stats and "train_end_time" in stats and "early_stopping_patience" in stats:
                train_start_time = stats["train_start_time"]
                train_end_time = stats["train_end_time"]
                early_stopping_patience = stats["early_stopping_patience"]

        else: # Old version
            train_losses, train_dcs, val_losses, val_dcs, lrs = stats["train_stats"]

        if not (train_losses and val_losses and lrs and train_dcs and val_dcs):
            ax1.set_title(f"Model {orig_idx + 1} (no stats)")
            ax1.axis("off")
            continue

        epochs = len(train_losses)
        epochs_list = list(range(1, epochs + 1))

        train_losses_clamped = torch.clamp(torch.tensor(train_losses), max=3)
        val_losses_clamped = torch.clamp(torch.tensor(val_losses), max=3)
        train_dcs_means = [np.mean(dc) for dc in train_dcs]
        val_dcs_means = [np.mean(dc) for dc in val_dcs]

        # Losses (left axis)
        ax1.plot(epochs_list, train_losses_clamped, label="Training Loss", color="tab:brown")
        ax1.plot(epochs_list, val_losses_clamped, label="Validation Loss", linewidth=2.5, color="tab:orange")

        # Best epoch line (early stopping)
        stopped_epoch = epochs - early_stopping_patience

        early_stop_label = "Best"
        ax1.axvline(
            x=stopped_epoch,
            linestyle="--",
            color="tab:gray",
            alpha=0.7,
            label=early_stop_label
        )

        best_val_loss = val_losses_clamped[stopped_epoch - 1]

        title = f"Model {orig_idx + 1} ({best_val_loss:.3f})"
        weight = 'bold' if (highlight_best and orig_idx == best_orig_idx) else 'normal'
        ax1.set_title(title, fontweight=weight)

        # Best (stopped_epoch) and last (epochs) ticks always shown, even if
        # close together. Other candidates (epoch 1, every 20th epoch) are
        # only kept if they don't land within min_gap of any tick already
        # kept (checked against ALL kept ticks, not just the nearest one, so
        # they can't sneak in right next to a must-keep tick either)
        must_keep_ticks = sorted({stopped_epoch, epochs})
        optional_ticks = sorted(({1} | set(range(20, epochs + 1, 20))) - set(must_keep_ticks))

        min_gap = max(2, round(epochs * 0.04))

        epoch_ticks = list(must_keep_ticks)
        for tick in optional_ticks:
            if all(abs(tick - kept) > min_gap for kept in epoch_ticks):
                epoch_ticks.append(tick)

        epoch_ticks = sorted(epoch_ticks)
        ax1.set_xticks(epoch_ticks)
        ax1.set_xticklabels(epoch_ticks)

        ax1.grid(alpha=0.3)

        # Only show left ylabel once per row (first column)
        if j == 0:
            ax1.set_ylabel("Loss")
        else:
            ax1.set_ylabel("")
            ax1.tick_params(axis="y", labelleft=False)
            ax1.set_yticks([])

        # LR (right axis)
        ax2 = ax1.twinx()
        ax2.set_yscale("log")
        ax2.plot(epochs_list, lrs, label="LR", color="tab:red")

        # DICE (second right axis)
        ax3 = ax1.twinx()
        ax3.spines["right"].set_position(("axes", 1.2))
        ax3.plot(epochs_list, train_dcs_means, label="Mean Training DICE", color="tab:green")
        ax3.plot(epochs_list, val_dcs_means, label="Mean Validation DICE", color="tab:blue")

        # Only show right ylabels once per row (last used column in this row)
        row_start = i * ncols
        row_end = min(row_start + ncols, n) - 1
        show_right_labels = (pos == row_end)

        if show_right_labels:
            ax2.set_ylabel("LR")
            ax3.set_ylabel("DICE")
            ax2.tick_params(axis="y", labelleft=False, labelright=True)
            ax3.tick_params(axis="y", labelleft=False, labelright=True)
        else:
            ax2.set_ylabel("")
            ax3.set_ylabel("")
            ax2.tick_params(axis="y", labelright=False)
            ax3.tick_params(axis="y", labelright=False)
            ax2.set_yticks([])
            ax2.set_yticklabels([])
            ax3.set_yticks([])
            ax3.set_yticklabels([])
            ax2.spines["right"].set_visible(False)
            ax3.spines["right"].set_visible(False)

        # Build one clean legend from first plotted panel only
        if legend_handles is None:
            h1, l1 = ax1.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            h3, l3 = ax3.get_legend_handles_labels()
            legend_handles = h1 + h2 + h3
            legend_labels = l1 + l2 + l3

    # Single legend above plots - needs more headroom when the (2-line)
    # suptitle is also shown above it, otherwise the two overlap
    legend_y = 0.88 if show_title else 0.97
    top_rect = 0.83 if show_title else 0.92

    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, legend_y)
    )

    # Fixed absolute space (not shrinking with figwidth) for the DICE/LR
    # twin-axis, which is offset 20% beyond its subplot's own right edge
    # (ax3.spines["right"].set_position(("axes", 1.2)) below) plus its
    # tick/axis labels - needed on every ncols, since col_width_in scaling
    # keeps each subplot's own width (and so that 20% offset) constant
    right_margin_in = 1.3
    right_rect = 1 - right_margin_in / figwidth

    fig.tight_layout(rect=[0, 0, right_rect, top_rect])
    fig.subplots_adjust(wspace=0.12)


    if save:
        save_path = SAVE_TRAIN_PATH.format(ensemble.name)
        print(f"Saving train to {save_path}")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    else:
        plt.show()


def color_swatch(ax, val):
    if val and val != "none":
        ax.add_patch(mpatches.Rectangle(
            (-0.12, -0.30), 1, 1.58,
            color=val,
            transform=ax.transAxes,
            clip_on=False,
            zorder=10,
        ))
    ax.axis("off")


def _format_dice(v):
    """0.6912 -> '.69', 1.0 -> '1.00', nan -> 'nan' (drop leading 0 to save space)"""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "nan"
    s = f"{v:.2f}"
    return s[1:] if s.startswith("0.") else s


def _get_val_loss_dice(train_stats):
    """Val loss & mean val DICE at the early-stopped/best epoch, from a model's train_stats."""
    if not train_stats:
        return np.nan, np.nan

    if "train_stats" not in train_stats:  # New version
        val_losses = train_stats.get("val_losses")
        val_dcs = train_stats.get("val_dcs")
        patience = train_stats.get("early_stopping_patience") or 0
    else:  # Old version
        _, _, val_losses, val_dcs, _ = train_stats["train_stats"]
        patience = 0

    if not val_losses or not val_dcs:
        return np.nan, np.nan

    stopped_epoch = len(val_losses) - patience
    return val_losses[stopped_epoch - 1], np.mean(val_dcs[stopped_epoch - 1])


# Rows where the smallest value is "best" (used when bolding the best value per row)
LOWER_IS_BETTER_ROWS = {"Val Loss"}


def visualize_test_ensembles(ensembles, num_top_models=None, save=False,
                              show_class_dice=False, show_average_dice=True, 
                              show_val=False, show_title=True, show_legend=True):
    if not ensembles:
        print("No ensembles to visualize.")
        return

    # Also allow to pass only one
    if not isinstance(ensembles, list):
        ensembles = [ensembles]

    Dataset_class = ensembles[0].Dataset_class

    # ── Filter ensembles without test stats ──────────────────────────────────
    valid_ensembles = []
    for ensemble in ensembles:
        has_ensemble = getattr(ensemble, "test_stats", None) is not None
        has_models   = any(getattr(m, "test_stats", None) for m in ensemble.models)
        if has_ensemble and has_models:
            valid_ensembles.append(ensemble)
        else:
            print(f"WARNING: '{ensemble.name}' skipped — missing test stats")

    if not valid_ensembles:
        print("No valid ensembles.")
        return

    # ── Class names and colors (skip class 0 = Unclassified) ─────────────────
    class_names      = Dataset_class.CLASS_NAMES[1:]
    class_colors_hex = [mcolors.to_hex(Dataset_class.CMAP_GT.colors[i])
                        for i in range(1, len(class_names) + 1)]
    n_classes        = len(class_names)

    # ── Build DataFrame ───────────────────────────────────────────────────────
    metric_labels = ["OA (%)", "AA (%)", "Kappa"]
    if show_average_dice:
        metric_labels.append("Avg DICE")
    if show_val:
        metric_labels.append("Val Loss")
        metric_labels.append("Val DICE")
    n_metric_rows = len(metric_labels)

    rows = {
        "num":   list(range(1, n_classes + 1)) + [""] * n_metric_rows,
        "class": list(class_names) + metric_labels,
        "color": class_colors_hex + ["none"] * n_metric_rows,
    }

    ensemble_groups = []

    # Custom (usually long) labels get their header rotated to fit narrow columns
    rotated_header_cols = set()
    # Skip the group title for these - a tall rotated header would collide with it
    groups_without_title = set()

    for ens_idx, ensemble in enumerate(valid_ensembles):
        available = [(i, m, m.test_stats) for i, m in enumerate(ensemble.models)
                     if getattr(m, "test_stats", None)]

        # Custom per-model labels, one per line, in ensemble.models order
        custom_labels = None
        custom_labels_path = os.path.join(ensemble.save_dir_path, ENSEMBLE_VIZ_CUSTOM_MODEL_LABELS)
        if os.path.exists(custom_labels_path):
            with open(custom_labels_path) as f:
                custom_labels = [line.strip() for line in f if line.strip()]

        if custom_labels is not None:
            # Labels are position-based, so keep original order (no accuracy sort/top-N)
            ordered = available
        else:
            ordered = sorted(available, key=lambda x: x[2]["overall_accuracy"], reverse=True)

            if num_top_models is not None:
                ordered = ordered[:num_top_models]


        def entry_label(i):
            if custom_labels is not None and i < len(custom_labels):
                return custom_labels[i]
            return f"#{i+1}"

        entries = [(entry_label(i), m, s) for i, m, s in ordered]
        entries.append(("E", None, ensemble.test_stats))  # ensemble itself has no train_stats

        if custom_labels is not None:
            rotated_header_cols.update(f"ens{ens_idx}_{label}" for label, _, _ in entries)
            groups_without_title.add(f"{ens_idx+1}. Ensemble")

        group_cols = []
        for label, model_obj, stats in entries:
            col_name = f"ens{ens_idx}_{label}"
            group_cols.append(col_name)

            dice_per_class = stats.get("dice_per_class", None)
            class_vals = []
            for c in range(n_classes):
                acc = stats["accuracy_in_classes"][c] * 100
                if show_class_dice:
                    dice_val = dice_per_class[c] if dice_per_class is not None else None
                    class_vals.append(f"{acc:.2f} | {_format_dice(dice_val)}")
                else:
                    class_vals.append(acc)

            metric_vals = [
                stats["overall_accuracy"] * 100,
                stats["average_accuracy"] * 100,
                stats.get("kappa", np.nan),
            ]
            if show_average_dice:
                metric_vals.append(stats.get("average_dice", np.nan))
            if show_val:
                val_loss, val_dice = _get_val_loss_dice(getattr(model_obj, "train_stats", None))
                metric_vals.append(val_loss)
                metric_vals.append(val_dice)

            rows[col_name] = class_vals + metric_vals

        ensemble_groups.append((f"{ens_idx+1}. Ensemble", group_cols))

    df = pd.DataFrame(rows)
    df = df.reset_index(drop=True)
    data_cols = [c for _, g in ensemble_groups for c in g]

    # ── Column definitions ────────────────────────────────────────────────────
    # width= is a relative weight, not absolute - fig_w below derives from these same weights
    NUM_COL_WIDTH = 0.2
    CLASS_COL_WIDTH = 2.0
    COLOR_COL_WIDTH = 0.15
    DATA_COL_WIDTH = 0.9 if show_class_dice else 0.45

    col_defs = [
        ColumnDefinition("num", title="#", width=NUM_COL_WIDTH),
        ColumnDefinition("class", title="Class Name   \\   Model Num.", width=CLASS_COL_WIDTH,
                         textprops={"ha": "left", "fontsize": 8}),
        ColumnDefinition("color", title="", width=COLOR_COL_WIDTH,
                         plot_fn=color_swatch),
    ]
    total_width_units = NUM_COL_WIDTH + CLASS_COL_WIDTH + COLOR_COL_WIDTH

    def cell_formatter(v):
        if isinstance(v, str):
            return v
        if v is None or np.isnan(v):
            return ""  # eg. ensemble ("E") has no val loss/dice, since it isn't trained itself
        return f"{v:.2f}"

    for ens_label, group_cols in ensemble_groups:
        show_group_title = ens_label not in groups_without_title
        for i, col in enumerate(group_cols):
            is_E_col = col.endswith("_E")
            border   = "left" if (i == 0 or is_E_col) else None
            col_defs.append(ColumnDefinition(
                name      = col,
                title     = col.split("_", 1)[1],
                group     = ens_label if show_group_title else None,
                width     = DATA_COL_WIDTH,
                formatter = cell_formatter,
                textprops = {"ha": "center", "fontsize": 8},
                border    = border,
            ))
            total_width_units += DATA_COL_WIDTH

    # ── Figure ────────────────────────────────────────────────────────────────
    n_data_cols = len(data_cols)
    # Fixed left/right chrome (unrelated to column content) - kept separate
    # from total_width_units so the table's own axes box can be positioned
    # in absolute inches below, directly tied to total_width_units, instead
    # of relying on matplotlib's default subplot margins (fragile: those
    # aren't guaranteed to be what any % of fig_w happens to assume)
    LEFT_MARGIN_IN = 0.4
    RIGHT_BUFFER_IN = 0.3

    # Rotated labels overflow upward past the header row unclipped; suptitle
    # placement below accounts for it
    has_rotated_headers = len(rotated_header_cols) > 0
    header_row_height = 1

    # Reserve right padding for the longest rotated label's overflow, or the table looks pushed left
    right_pad_in = 0.0
    if rotated_header_cols:
        longest_label = max((col.split("_", 1)[1] for col in rotated_header_cols), key=len)
        label_width_in = TextPath((0, 0), longest_label, size=8).get_extents().width / 72.0
        right_pad_in = label_width_in * 0.866 + 0.2  # cos(30°) horizontal projection + small buffer

    fig_w = LEFT_MARGIN_IN + total_width_units + RIGHT_BUFFER_IN + right_pad_in

    n_legend_lines = 3 + len(valid_ensembles)
    fig_h = 2.0 + (n_classes + n_metric_rows) * 0.28 + n_legend_lines * 0.08

    fig, (ax, ax_legend) = plt.subplots(
        2, 1,
        figsize=(fig_w, fig_h),
        gridspec_kw={"height_ratios": [1, n_legend_lines * 0.08 / fig_h]},
    )
    ax.set_axis_off()
    ax_legend.set_axis_off()

    # Table() below sets ax's data xlim to sum(column widths) == total_width_units
    # (see plottable's Table.__init__) - position ax so its box is EXACTLY
    # total_width_units inches wide, so 1 data-unit stays ~1 rendered inch and
    # the table's size responds directly to the column width= values, not to
    # whatever fraction of fig_w matplotlib's defaults happen to carve out
    ax_pos = ax.get_position()
    ax.set_position([LEFT_MARGIN_IN / fig_w, ax_pos.y0, total_width_units / fig_w, ax_pos.height])

    tab = Table(
        df,
        ax                   = ax,
        index_col            = "num",
        column_definitions   = col_defs,
        row_dividers         = True,
        row_divider_kw       = {"linewidth": 0.3, "color": "#cccccc"},
        col_label_divider    = True,
        col_label_divider_kw = {"linewidth": 1.0, "color": "black"},
        col_label_cell_kw    = {"height": header_row_height},
        column_border_kw     = {"linewidth": 0.5, "color": "black"},  # thin globally
        cell_kw              = {"linewidth": 0},
        textprops            = {"fontsize": 8},
        odd_row_color        = "#f5f5f5",
        even_row_color       = "white",
    )

    # Rotate only the header text (not body cells) so long labels fit
    if rotated_header_cols:
        header_col_order = list(df.columns)
        for col_name in rotated_header_cols:
            col_idx = header_col_order.index(col_name)
            header_cell = tab.col_label_row.cells[col_idx]
            header_cell.text.set_rotation(45)
            header_cell.text.set_ha("left")
            header_cell.text.set_va("bottom")
            # Anchor at the cell's true bottom-left corner (va alone keeps
            # plottable's default centered position); nudged for alignment
            header_cell.text.set_position((header_cell.x + 0.1, header_cell.y + 0.7))

        # These columns never get a group title, so their row-slice stays
        # blank and tall; anchor to the same corner as the rotated labels to
        # avoid a gap above
        for col_name in ("num", "class", "color"):
            col_idx = header_col_order.index(col_name)
            header_cell = tab.col_label_row.cells[col_idx]
            header_cell.text.set_ha("left")
            header_cell.text.set_va("bottom")
            header_cell.text.set_position((header_cell.x + 0.1, header_cell.y + 0.7))

    # thicken borders on first col of each ensemble group
    first_cols = [group_cols[0] for _, group_cols in ensemble_groups]
    
    # get x positions of those columns
    thick_xs = set()
    for col_name in first_cols:
        col = tab.columns[col_name]
        thick_xs.add(round(col.x, 4))

    # find all vertical Line2D objects on the axes at those x positions
    for line in ax.get_lines():
        xdata = line.get_xdata()
        if len(xdata) == 2 and xdata[0] == xdata[1]:  # vertical line
            if round(xdata[0], 4) in thick_xs:
                line.set_linewidth(2.0)

    # draw separator above OA/AA/Kappa rows
    oa_y = tab.rows[n_classes - 1].cells[0].xy[1] + tab.rows[n_classes - 1].cells[0].height
    ax.plot([ax.get_xlim()[0], ax.get_xlim()[1]], [oa_y, oa_y],
            color="black", linewidth=1.0, zorder=5, transform=ax.transData)

    # bold the best value in each row (lowest for loss rows, highest otherwise)
    for row_idx in range(len(df)):
        vals = {}
        for c in data_cols:
            v = df.iloc[row_idx][c]
            try:
                v = float(v)
                if not np.isnan(v):
                    vals[c] = v
            except (ValueError, TypeError):
                pass
        if not vals:
            continue
        lower_is_better = df.iloc[row_idx]["class"] in LOWER_IS_BETTER_ROWS
        best_val = min(vals.values()) if lower_is_better else max(vals.values())
        for col, val in vals.items():
            if abs(val - best_val) < 1e-6 and (lower_is_better or best_val > 0):
                tab.columns[col].cells[row_idx].text.set_fontweight("bold")

    title_parts = ["Accuracies (%)"]
    if show_class_dice or show_average_dice:
        title_parts.append("DICE")
    if show_val:
        title_parts.append("Val Loss/DICE")
    top_models_desc = f"Top {num_top_models}" if num_top_models is not None else "All"

    # Measure the tallest rendered label and place the title just above it
    # (no fixed y works for all label lengths)
    suptitle_y = 0.98
    if rotated_header_cols:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        header_col_order = list(df.columns)
        max_top_px = max(
            tab.col_label_row.cells[header_col_order.index(col_name)].text
               .get_window_extent(renderer=renderer).y1
            for col_name in rotated_header_cols
        )
        fig_height_px = fig.get_size_inches()[1] * fig.dpi
        # Grow above the 0.98 default as needed (never shrink below it) -
        # bbox_inches="tight" on save expands the canvas to fit, so y > 1 is fine
        suptitle_y = max(0.98, max_top_px / fig_height_px + 0.04)

    if show_title:
        fig.suptitle(
            f"Ensembles {' & '.join(title_parts)} — Each {top_models_desc} Models & Ensemble",
            fontsize=12, fontweight="bold", y=suptitle_y,
        )

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_lines = ["E: whole Ensemble", "", "Ensemble Legend:"]
    for i, ens in enumerate(valid_ensembles):
        legend_lines.append(f'  {i+1}. "{ens.name}"')

    if show_legend:
        ax_legend.text(0.0, 1.0, "\n".join(legend_lines),
                    fontsize=7, family="monospace",
                    va="top", ha="left",
                    transform=ax_legend.transAxes)

    # plt.tight_layout()

    if save:
        save_name = get_unique_save_name(PLOTS_SAVE_DIR, "ensembles_comparison", extension=".png")
        save_path = f"{PLOTS_SAVE_DIR}{save_name}"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved to {save_path}")
    else:
        plt.show()
        
# For Full Area Test Visualization
def show_full_area_test(ensemble, subsample_factor=1, save=False, show_entropy=False, show_prediction=False,
                         show_models=True, show_title=True):
    """ Show predictions on whole test dataset for ensemble as one big image

    # full_predictions:
    # from ensemble.save_dir_path + ENSEMBLE_FULL_TEST_FILENAME
    (1 + N_MODELS, N_TEST_IMAGES, 2, H, W) 
    - [class index, entropy] per pixel

    First combine all prediction tiles with respect to
    ensemble.Dataset_class.TILES_PER_ROW

    Then subsample and generate a image and save

    show_entropy: plot each pixel's entropy instead of its class prediction
    show_models: if False, only plot the ensemble's own prediction, skipping each individual model
    - visualizes uncertainty

    Sizes input mask
    # H: ca 1000px, W: ca 3000px
    """

    path = os.path.join(ensemble.save_dir_path, ENSEMBLE_FULL_TEST_FILENAME)
    if not os.path.exists(path):
        print("\n\nBEFORE THIS RUN: ensemble.test_predict_full_dataset()\n")
        raise FileNotFoundError(f"Full test predictions not found at: {path}")

    full_predictions = torch.load(path, weights_only=False, map_location='cpu')

    # Convert to tensor if it's a list or numpy array
    if isinstance(full_predictions, list):
        full_predictions = torch.from_numpy(np.array(full_predictions))
    elif isinstance(full_predictions, np.ndarray):
        full_predictions = torch.from_numpy(full_predictions)

    full_predictions = full_predictions.float()

    # (1 + N_MODELS, N_TEST_IMAGES, 2, H, W)
    print(f"Full predictions shape: {full_predictions.shape}")
    n_models_plus_ensemble, n_test_images, n_channels, h, w = full_predictions.shape

    # Reconstruct full-size images from tiles
    # Tiles are saved in raster scan order: row by row, left to right
    # We need to figure out the grid dimensions

    # Try to use TILES_PER_ROW if it matches, otherwise compute from factors
    tiles_per_row_candidate = ensemble.Dataset_class.VIS_TILES_PER_ROW

    # Check if it divides evenly or is a reasonable factor
    if n_test_images % tiles_per_row_candidate == 0:
        tiles_per_row = tiles_per_row_candidate
        tiles_per_col = n_test_images // tiles_per_row
    else:
        raise ValueError(f"Number of test images ({n_test_images}) is not divisible by TILES_PER_ROW ({tiles_per_row_candidate}). ")

    print(f"Reconstructing from {n_test_images} tiles: {tiles_per_row}W × {tiles_per_col}H = {tiles_per_row*tiles_per_col}")

    # Index 0 is the ensemble's own prediction, indices 1..N_MODELS are the individual
    # models (see Ensemble.test_predict_full_dataset(): all_preds[0] = ensemble, all_preds[i+1] = model i)
    positions_to_show = range(n_models_plus_ensemble) if show_models else [0]

    reconstructed = {}
    for pos in positions_to_show:
        tiles = full_predictions[pos]  # (N_TEST_IMAGES, 2, H, W)

        # Pad to complete rectangle
        expected_tiles = tiles_per_row * tiles_per_col
        if n_test_images < expected_tiles:
            padding = expected_tiles - n_test_images
            print(f"Padding with {padding} empty tiles to complete rectangle")
            tiles = torch.cat([tiles, torch.zeros(padding, n_channels, h, w, dtype=tiles.dtype)], dim=0)

        # Reshape to (TILES_PER_COL, TILES_PER_ROW, 2, H, W)
        # Raster scan order: index = row * tiles_per_row + col
        tiles_grid = tiles.view(tiles_per_col, tiles_per_row, n_channels, h, w)

        # Concatenate horizontally within each row (along W)
        rows = [torch.cat([tiles_grid[row, col] for col in range(tiles_per_row)], dim=2)
                for row in range(tiles_per_col)]

        # Stack rows vertically (along H)
        full_img = torch.cat(rows, dim=1)
        reconstructed[pos] = full_img

    # Max possible entropy (uniform distribution over classes) - fixes the
    # entropy colorbar's scale so it's comparable across models/ensemble
    max_entropy = np.log(ensemble.Dataset_class.NUM_CLASSES)

    # Create individual plots for each model + ensemble
    for pos in positions_to_show:
        # Subsample for visualization (every Nth pixel)
        class_pred, entropy = reconstructed[pos][:, ::subsample_factor, ::subsample_factor]  # each (H, W)

        if pos == 0:
            title = "Ensemble"
            name_suffix = f"{ensemble.name}_ensemble"
        else:
            title = f"Model {pos}"
            name_suffix = f"{ensemble.name}_model_{pos}"

        if show_prediction:
            fig, ax = plt.subplots(figsize=(15, 10))

            ax.imshow(class_pred.numpy(), cmap=ensemble.Dataset_class.CMAP_GT, vmin=0, vmax=20)
            
            if show_title:
                ax.set_title(title, fontsize=16)
            
            save_path = SAVE_FULL_AREA_TEST_PATH.format(name_suffix)

            ax.axis('off')
            fig.tight_layout()

            if save:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"Saved to {save_path}")
                plt.close(fig)
            else:
                plt.show()

        if show_entropy:
            fig, ax = plt.subplots(figsize=(15, 10))

            # Nicer cmap but not used before: "Spectral"
            im = ax.imshow(entropy.numpy(), cmap='RdBu_r', vmin=0, vmax=max_entropy)
            
            if show_title:
                ax.set_title(f"{title} — Entropy", fontsize=16)
            
            # no colorbar
            # fig.colorbar(im, ax=ax, label='Entropy', fraction=0.03, pad=0.02)
            save_path = SAVE_FULL_AREA_TEST_PATH.format(f"{name_suffix}_entropy")

            ax.axis('off')
            fig.tight_layout()

            if save:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"Saved to {save_path}")
                plt.close(fig)
            else:
                plt.show()


def visualize_ensemble_modulations(ensemble, save=False, save_dir='plots/', num_plots=None,
                                    force_cells_quadratic=True, model_indices=None, show_title=True):
    """ Visualize all modulations in an ensemble in a grid.

    Creates a grid showing each model's modulation matrix with:
    - Every subplot the same box size (width sized to the widest K across all
      models, height to C)
    - Shared Y-axes (channel labels)
    - Single colorbar for all

    force_cells_quadratic: if True (default), each modulation is drawn at a
    true 1:1 scale so every cell is exactly square - narrower-K models then
    render smaller within their (still same-sized) box, with padding rather
    than filling it. If False, narrower-K models are instead stretched to
    fill the full box width, so all subplots look the same width, at the
    cost of cells no longer being perfectly square.

    model_indices: explicit list of model indices (into
    ensemble.channel_modulations) to show, eg. [0, 3, 6] - shown in the given
    order. Takes precedence over num_plots (which only takes the first N).
    Subplot titles still show each model's original ("Model {i+1}") number
    and custom_model_labels.txt entry, not its position in this subset.

    show_title: if False, hides the whole-grid suptitle (ensemble name +
    model count) and reclaims most of the space reserved for it. Per-subplot
    titles (model number, K, custom label) are unaffected.
    """
    import matplotlib.gridspec as gridspec

    if model_indices is not None:
        selected_indices = list(model_indices)
    else:
        selected_indices = list(range(len(ensemble.channel_modulations)))
        if num_plots is not None:
            selected_indices = selected_indices[:num_plots]

    channel_modulations = [ensemble.channel_modulations[i] for i in selected_indices]

    # Custom per-model labels (eg. which modalities feed that model), one per
    # line, in the same order as ensemble.channel_modulations - same file/
    # convention as visualize_test_ensembles()
    custom_labels = None
    custom_labels_path = os.path.join(ensemble.save_dir_path, ENSEMBLE_VIZ_CUSTOM_MODEL_LABELS)
    if os.path.exists(custom_labels_path):
        with open(custom_labels_path) as f:
            custom_labels = [line.strip() for line in f if line.strip()]

    n_models = len(channel_modulations)
    MAX_COLS = 3
    ncols = min(MAX_COLS, n_models)
    nrows = int(np.ceil(n_models / ncols))

    # Scale all text with grid size: a saved PNG's absolute font size stays
    # constant regardless of subplot count, but the whole figure keeps
    # growing with more subplots, so at a typical (downscaled) viewing size
    # the same absolute text renders smaller relative to the bigger image -
    # scale fonts up to compensate, based on the grid's larger dimension
    font_scale = max(1.0, max(ncols, nrows) ** 0.5)

    # Get all K values and C
    K_values = [mod.shape[1] for mod in channel_modulations]
    C = channel_modulations[0].shape[0]

    # Inches per data unit (channel / dimension) - when force_cells_quadratic
    # is True, identical for both axes so every cell is exactly square
    cell_size = 0.12

    # Every subplot gets the same box size - width sized to the widest K
    # across ALL models (not just per column), height to C, shared by every
    # model - so all subplots are identical in width and height. Depending on
    # force_cells_quadratic, narrower-K models either render smaller within
    # their box (padding) or get stretched to fill it - see imshow() below.
    max_K = max(K_values)
    col_widths_in = [max_K * cell_size] * ncols
    row_height_in = C * cell_size

    def text_width_in(text, fontsize, weight='normal'):
        """ Precise rendered width (inches) of `text` at `fontsize` (points),
        via TextPath - font-metric based, no renderer/draw needed. """
        return TextPath((0, 0), text, size=fontsize, prop=FontProperties(weight=weight)).get_extents().width / 72.0

    def text_height_in(text, fontsize, weight='normal'):
        return TextPath((0, 0), text, size=fontsize, prop=FontProperties(weight=weight)).get_extents().height / 72.0

    # Left margin has to fit two things stacked left-to-right: the rotated
    # "Channels (C)" axis label, then the widest y-tick channel label (eg.
    # "LIDAR DSM (1ch)") - both grow with font_scale, so a flat per-scale
    # inch guess either wastes space or (as font_scale grows) undersizes and
    # the two start overlapping. Measure them precisely instead.
    y_tick_fontsize = 9 * font_scale
    axis_label_fontsize = 11 * font_scale

    longest_y_label = max(
        (f"{name} ({ensemble.Dataset_class.MODALITIES_CHANNEL_COUNTS[key]}ch)"
         for name, key in zip(ensemble.Dataset_class.MODALITIES_NAMES, ensemble.Dataset_class.MODALITIES)),
        key=len,
    )
    y_tick_label_width_in = text_width_in(longest_y_label, y_tick_fontsize, weight='bold')
    # Rotated 90°, so its rendered horizontal footprint is its unrotated height
    axis_label_width_in = text_height_in('Channels (C)', axis_label_fontsize, weight='bold')

    pad_in = 0.12
    left_margin = pad_in + axis_label_width_in + pad_in + y_tick_label_width_in + pad_in

    # Other margins (inches) reserved around the grid for the suptitle/
    # colorbar, and gaps between subplots - scaled by font_scale too, since
    # bigger text (tick labels, per-subplot titles) needs proportionally
    # more room or it starts overlapping
    right_margin = 1.1 * font_scale
    top_margin = (1.3 if show_title else 0.4) * font_scale
    bottom_margin = 0.9 * font_scale
    hspace_in, wspace_in = 0.5 * font_scale, 0.35 * font_scale

    grid_width_in = sum(col_widths_in) + wspace_in * (ncols - 1)
    grid_height_in = row_height_in * nrows + hspace_in * (nrows - 1)

    figwidth = left_margin + grid_width_in + right_margin
    figheight = top_margin + grid_height_in + bottom_margin

    # Create figure with GridSpec, sized in inches so the grid area exactly
    # matches col_widths_in/row_height_in
    fig = plt.figure(figsize=(figwidth, figheight))
    gs = gridspec.GridSpec(
        nrows, ncols, figure=fig,
        width_ratios=col_widths_in, height_ratios=[1] * nrows,
        left=left_margin / figwidth, right=1 - right_margin / figwidth,
        top=1 - top_margin / figheight, bottom=bottom_margin / figheight,
        hspace=hspace_in / row_height_in,
        wspace=wspace_in / (grid_width_in / ncols),
    )

    if show_title:
        fig.suptitle(f"{ensemble.name}\nChannel Modulations ({n_models} models)",
                     fontsize=14 * font_scale, fontweight='bold', y=1 - 0.4 * top_margin / figheight)

    axes = [[None] * ncols for _ in range(nrows)]
    images = []

    # Create all subplots. pos: position in the displayed grid (always
    # contiguous 0..n_models-1). orig_idx: the model's actual index in
    # ensemble.channel_modulations (may skip around when model_indices is
    # given) - used for the title/custom label, not grid placement.
    for pos, (orig_idx, modulation) in enumerate(zip(selected_indices, channel_modulations)):
        row = pos // ncols
        col = pos % ncols
        ax = fig.add_subplot(gs[row, col])
        axes[row][col] = ax

        modulation_np = modulation.cpu().numpy() if hasattr(modulation, 'cpu') else modulation
        C_curr, K_curr = modulation_np.shape

        # Display modulation matrix - 'equal' keeps cells square (may leave
        # padding for narrower-K models), 'auto' stretches to fill the box
        aspect = 'equal' if force_cells_quadratic else 'auto'
        im = ax.imshow(modulation_np, cmap='RdBu_r', aspect=aspect, vmin=-1, vmax=1)
        images.append(im)

        # Remove borders/spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

        # Add modality separators
        cumsum = 0
        for modality_name, modality_key in zip(ensemble.Dataset_class.MODALITIES_NAMES,
                                               ensemble.Dataset_class.MODALITIES):
            count = ensemble.Dataset_class.MODALITIES_CHANNEL_COUNTS[modality_key]
            cumsum += count
            if cumsum < C_curr:
                ax.axhline(y=cumsum - 0.5, color='white', linewidth=1.5)

        title = f'Model {orig_idx + 1} (K={K_curr})'
        if custom_labels is not None and orig_idx < len(custom_labels):
            title += f': {custom_labels[orig_idx]}'
        ax.set_title(title, fontsize=10 * font_scale, fontweight='bold', pad=12)

        # Set X-ticks with smart spacing to avoid overlap
        if K_curr <= 10:
            xtick_spacing = 1
        elif K_curr <= 20:
            xtick_spacing = 2
        elif K_curr <= 50:
            xtick_spacing = 5
        else:
            xtick_spacing = 10

        xticks = list(range(0, K_curr - xtick_spacing, xtick_spacing))
        if K_curr - 1 not in xticks and K_curr - 1 - (xticks[-1] if xticks else 0) > 3:
            xticks.append(K_curr - 1)

        # Filter ticks: don't show if less than 3 away from last shown tick
        # But always show the last tick if different from the last filtered one
        filtered_xticks = []
        for tick in xticks:
            if not filtered_xticks or tick - filtered_xticks[-1] > 3:
                filtered_xticks.append(tick)

        # Ensure last tick is always shown if different
        if K_curr - 1 not in filtered_xticks:
            filtered_xticks.append(K_curr - 1)

        ax.set_xticks(filtered_xticks)
        ax.set_xticklabels(filtered_xticks, fontsize=7 * font_scale)

    # Add Y-axis labels only to leftmost column
    for row in range(nrows):
        for col in range(ncols):
            ax = axes[row][col]
            if ax is not None:
                if col == 0:
                    # Leftmost: show labels
                    y_ticks = []
                    y_labels = []
                    cumsum = 0

                    for modality_name, modality_key in zip(ensemble.Dataset_class.MODALITIES_NAMES,
                                                           ensemble.Dataset_class.MODALITIES):
                        count = ensemble.Dataset_class.MODALITIES_CHANNEL_COUNTS[modality_key]
                        start = cumsum
                        end = cumsum + count
                        mid = (start + end) / 2
                        y_ticks.append(mid)
                        y_labels.append(f"{modality_name} ({count}ch)")
                        cumsum = end

                    ax.set_yticks(y_ticks)
                    ax.set_yticklabels(y_labels, fontsize=9 * font_scale, fontweight='bold')
                else:
                    # Non-leftmost: hide y-ticks (they share the left axis)
                    ax.set_yticks([])
                    ax.set_yticklabels([])

    # Add axis labels - positioned within the reserved bottom/left margins
    fig.text(0.5, 0.3 * bottom_margin / figheight, 'Modulation Dimensions (K)',
              ha='center', fontsize=11 * font_scale, fontweight='bold')
    fig.text((pad_in + axis_label_width_in / 2) / figwidth, 0.5, 'Channels (C)',
              ha='center', va='center', rotation='vertical', fontsize=11 * font_scale, fontweight='bold')

    # Single colorbar for all subplots - positioned within the reserved right margin
    cbar_ax = fig.add_axes([
        1 - right_margin / figwidth + 0.35 / figwidth, bottom_margin / figheight,
        0.25 / figwidth, 1 - (top_margin + bottom_margin) / figheight,
    ])
    cbar = fig.colorbar(images[0], cax=cbar_ax)
    cbar.set_label('Modulation Value', fontsize=11 * font_scale)
    cbar.ax.tick_params(labelsize=9 * font_scale)

    # Hide empty subplots
    for idx in range(n_models, nrows * ncols):
        row = idx // ncols
        col = idx % ncols
        ax = axes[row][col]
        if ax is not None:
            ax.set_visible(False)

    if save:
        save_path = SAVE_MODULATIONS_VIEW_PATH.format(ensemble.name)
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved ensemble modulations to {save_path}")
        plt.close(fig)
    else:
        plt.show()


def plot_class_counts_and_weights(Dataset_class, save=False):
    """
    Bar chart of per-class pixel counts and loss weights (skip class 0 = Unclassified).
    Counts and weights differ by orders of magnitude, so weights get their own
    twin y-axis - same convention as the LR/DICE twin-axes in visualize_train_ensemble.
    """
    class_names = Dataset_class.CLASS_NAMES[1:]
    counts      = Dataset_class.CLASS_COUNTS[1:].cpu().numpy()
    weights     = Dataset_class.CLASS_WEIGHTS[1:].cpu().numpy()
    n_classes   = len(class_names)

    x     = np.arange(n_classes)
    width = 0.38

    fig, ax1 = plt.subplots(figsize=(max(8, n_classes * 0.6), 5))

    bars_counts = ax1.bar(x - width / 2, counts, width, color="tab:blue", label="Pixel Count")
    ax1.set_ylabel("Pixel Count (log)", color="tab:blue")
    ax1.set_yscale("log")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    bars_weights = ax2.bar(x + width / 2, weights, width, color="tab:red", label="Loss Weight")
    ax2.set_ylabel("Loss Weight", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    ax1.set_xticks(x)
    ax1.set_xticklabels(class_names, rotation=45, ha="right")
    ax1.set_title(f"{Dataset_class.NAME_SHORT} — Class Counts & Weights")
    ax1.grid(axis="y", alpha=0.3)

    fig.legend(
        [bars_counts, bars_weights], ["Pixel Count (log)", "Loss Weight"],
        loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02),
    )

    fig.tight_layout()

    if save:
        save_name = get_unique_save_name(PLOTS_SAVE_DIR, f"class_counts_weights_{Dataset_class.NAME_SHORT}", extension=".png")
        save_path = f"{PLOTS_SAVE_DIR}{save_name}"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved to {save_path}")
    else:
        plt.show()


def _modulation_opts_key(opts):
    """ Hashable, display-friendly key for a modulation_opts dict: callables
    (eg. previous_modulation) become their __name__, same convention
    Ensemble.get_ensemble_name() already uses for naming on disk. """
    return tuple(sorted((k, v.__name__ if callable(v) else v) for k, v in opts.items()))


def _lighten(rgb, amount=0.6):
    """ Blend an RGB(A) color toward white. amount=0 -> unchanged, 1 -> white.
    Used so a group's model-level bars read as a tint of its ensemble-level
    color, rather than needing transparency alone to stay distinguishable. """
    r, g, b = rgb[:3]
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)


def _mean_std_var(values):
    """ Sample mean/std/var (ddof=1) - falls back to ddof=0 for a single
    value, where sample variance is undefined (would be NaN) rather than a
    meaningful "no spread observed yet". """
    values = np.asarray(values)
    ddof = 1 if len(values) > 1 else 0
    return values.mean(), values.std(ddof=ddof), values.var(ddof=ddof)


DEFAULT_N_MODELS = 7

# How to shorten a previous_modulation function name in a "prev: ..." title
# annotation - truncated to whichever end (front or back) is the distinctive
# part, falling back to a generic tail-truncation for anything unlisted
_PREV_MOD_ABBREV = {
    "powerset_of_all_modalities": "powerset of all modalities",
    "random_classes_simple_LDA": "...simple_LDA",
    "random_channel_drop": "...channel_drop",
    "random_drop_percentage_of_modality": "...drop_percentage",
    "random_linear_combination": "...linear_combination",
    "identity": "identity",
}


def _abbrev_prev_modulation(name):
    return _PREV_MOD_ABBREV.get(name, f"...{name[-12:]}")


def _title_math_annotation(func_name, opts, n_models, dataset_class):
    """ Short, math-formatted ($...$, rendered via matplotlib mathtext so it
    visually stands out from the plain-text title) per-group title
    annotation for visualize_ensemble_variance(), replacing the old
    footnote/legend disambiguation. Only random_subsets_PCA,
    random_channel_drop and random_classes_simple_LDA get parameter-specific
    annotations (their most relevant/varied opts); every function gets an
    N_E annotation whenever its model count isn't DEFAULT_N_MODELS. """
    math_parts = []  # combined into one $...$ block
    text_parts = []  # plain text, appended after, outside the math block

    if func_name == "random_subsets_PCA":
        prev_mod_name = opts.get("previous_modulation")

        K = opts.get("components_count", opts.get("max_components_count"))

        if prev_mod_name == "powerset_of_all_modalities":
            if K is not None:
                math_parts.append(f"max K = {K}")
            else:
                # No explicit cap - K = C by default, not a fixed number worth
                # stating, so just note the chaining instead
                text_parts.append(f"prev: {_abbrev_prev_modulation(prev_mod_name)}")
        else:
            if K is not None:
                math_parts.append(f"K = {K}")
            if prev_mod_name is not None:
                text_parts.append(f"prev: {_abbrev_prev_modulation(prev_mod_name)}")

        p = opts.get("samples_percentage")
        if p is not None and p != 0.75:
            math_parts.append(f"p = {p}")

    elif func_name == "random_channel_drop":
        p = opts.get("drop_prob")
        if p is not None and p != 0.3:
            math_parts.append(f"p = {p}")

    elif func_name == "random_classes_simple_LDA":
        K = opts.get("K")
        if K is not None:
            math_parts.append(f"K = {K}")

    if n_models != DEFAULT_N_MODELS:
        math_parts.append(f"N_E = {n_models}")

    annotation = f"${', '.join(math_parts)}$" if math_parts else ""
    if text_parts:
        annotation += (", " if annotation else "") + ", ".join(text_parts)
    return annotation


def visualize_ensemble_variance(ensembles, save=False, show_title=True):
    """
    One subplot per (channel_modulation_func, modulation_opts, n_models)
    group - auto-detected, not caller-provided - each a histogram of that
    group's ensemble-level Overall Accuracy (OA) plus the pooled OA of every
    individual (tested) model across all that group's ensembles (a lighter
    tint of the same color). All subplots share the same OA axis range for
    direct comparison. Subplot titles are the bare function name plus a
    short math-formatted annotation of whichever opts (and model count, if
    non-default) distinguish that group - see _title_math_annotation().
    """

    MAX_COLS = 2

    if not ensembles:
        print("No ensembles to visualize.")
        return
    if not isinstance(ensembles, list):
        ensembles = [ensembles]

    valid = []
    for e in ensembles:
        if getattr(e, "test_stats", None) is None:
            print(f"WARNING: '{e.name}' skipped — missing test stats")
            continue
        valid.append(e)
    if not valid:
        print("No valid ensembles.")
        return

    groups = {}  # (func_name, opts_key, n_models) -> list[Ensemble], insertion-ordered
    for e in valid:
        n_models = getattr(e, "n_models", len(e.models))
        key = (e.channel_modulation_func.__name__, _modulation_opts_key(getattr(e, "modulation_opts", {})), n_models)
        groups.setdefault(key, []).append(e)

    # Per group: ensemble-level OAs, and pooled OA of every individual (tested)
    # model across all of that group's ensembles
    group_entries = []  # [(func_name, opts_key, n_models, dataset_class, ensemble_oas, model_oas), ...]
    for (func_name, opts_key, n_models), group in groups.items():
        ensemble_oas = [e.test_stats["overall_accuracy"] * 100 for e in group]
        model_oas = [m.test_stats["overall_accuracy"] * 100
                     for e in group for m in e.models if getattr(m, "test_stats", None) is not None]
        group_entries.append((func_name, opts_key, n_models, group[0].Dataset_class, ensemble_oas, model_oas))

    # Shared bins across the full OA range (every group's ensembles + models), with padding.
    # 1%-wide bins with whole-number edges so the default tick locator (which
    # picks "nice" round numbers) lands on bin edges instead of mid-bar.
    all_oas = [oa for *_, ens_oas, mod_oas in group_entries for oa in ens_oas + mod_oas]
    oa_min, oa_max = min(all_oas), max(all_oas)
    pad = max(2.0, (oa_max - oa_min) * 0.15)
    bin_edges = np.arange(np.floor(oa_min - pad), np.ceil(oa_max + pad) + 1, 1.0)

    n_groups = len(group_entries)
    ncols = min(MAX_COLS, n_groups)
    nrows = int(np.ceil(n_groups / ncols))
    tab10 = plt.colormaps["tab10"].colors  # 10 discrete, maximally-distinct colors (not continuous-sampled)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3 * nrows), squeeze=False, sharex=True, sharey=True)

    for i, (func_name, opts_key, n_models, dataset_class, ensemble_oas, model_oas) in enumerate(group_entries):
        ax = axes[i // ncols][i % ncols]
        color = tab10[i % 10]
        model_color = _lighten(color, 0.6)

        # Models drawn first (usually far more of them, so taller bars) so the
        # ensemble-level bars stay visible drawn on top, not hidden behind them.
        # Stats go on a 2nd legend line per entry (not a wider one) so it stays
        # clear which numbers are model- vs ensemble-level without widening it.
        if model_oas:
            mod_mean, mod_std, mod_var = _mean_std_var(model_oas)
            ax.hist(model_oas, bins=bin_edges, color=model_color, alpha=0.9,
                    label=f"Models (n={len(model_oas)})  $\\mathbf{{\\mu}}$={mod_mean:.2f}%\n"
                          f"  $\\mathbf{{\\sigma}}$={mod_std:.2f}  $\\mathbf{{Var}}$={mod_var:.2f}")
            # Own series' color (not grey) so it's obvious which mean belongs
            # to which - a black outline (path_effects) keeps it visible even
            # sitting on top of a same-colored bar; dotted (vs ensemble's
            # dashed below) adds a second cue for when the two means overlap
            ax.axvline(mod_mean, color=model_color, linestyle="--", linewidth=2,
                       path_effects=[mpatheffects.withStroke(linewidth=3, foreground="white")])

        ens_mean, ens_std, ens_var = _mean_std_var(ensemble_oas)
        ax.hist(ensemble_oas, bins=bin_edges, color=color, alpha=0.9,
                label=f"Ensembles (n={len(ensemble_oas)})  $\\mathbf{{\\mu}}$={ens_mean:.2f}%\n"
                      f"  $\\mathbf{{\\sigma}}$={ens_std:.2f}  $\\mathbf{{Var}}$={ens_var:.2f}")
        ax.axvline(ens_mean, color=color, linestyle="--", linewidth=2,
                   path_effects=[mpatheffects.withStroke(linewidth=3, foreground="white")])

        display_title = func_name.replace("_", " ")
        display_title = display_title[:1].upper() + display_title[1:]
        annotation = _title_math_annotation(func_name, dict(opts_key), n_models, dataset_class)
        if annotation:
            display_title += f" | {annotation}"
        ax.set_title(display_title, fontsize=10)
        ax.set_xlim(bin_edges[0], bin_edges[-1])
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        # sharex/sharey=True hide tick labels on non-edge subplots by default;
        # show them everywhere since rows/columns can be far apart in a
        # multi-row grid
        ax.tick_params(axis="x", labelbottom=True)
        ax.tick_params(axis="y", labelleft=True)

        if i % ncols == 0:
            ax.set_ylabel("Count")
        if i // ncols == nrows - 1:
            ax.set_xlabel("Overall Accuracy (%)")

    # Hide unused grid cells (n_groups doesn't evenly fill nrows*ncols)
    for j in range(n_groups, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    if show_title:
        fig.suptitle("Distribution of Overall Accuracy across ensemble runs & their models",
                    fontsize=13, fontweight="bold")

    fig.tight_layout(rect=[0, 0.02, 1, 0.95], h_pad=1.5, w_pad=2)

    if save:
        save_name = get_unique_save_name(PLOTS_SAVE_DIR, "ensemble_variance", extension=".png")
        save_path = f"{PLOTS_SAVE_DIR}{save_name}"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved to {save_path}")
    else:
        plt.show()
