import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ── Config ────────────────────────────────────────────────────────────────────

CSV_PATH   = "fitness_tracker_cleaned.csv"
OUT_DIR    = "static/images/"
DPI        = 180

PALETTE    = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
BG         = "#F8F9FA"
GRID_COL   = "#DEE2E6"
TITLE_FS   = 15
LABEL_FS   = 12
TICK_FS    = 10
ANNOT_FS   = 9

# ── Load & Prepare Data ───────────────────────────────────────────────────────

df = pd.read_csv(CSV_PATH)

exp_map = {1.0: "Beginner", 2.0: "Intermediate", 3.0: "Advanced"}
df["experience_label"] = df["experience_level"].map(exp_map)

EXP_ORDER = ["Beginner", "Intermediate", "Advanced"]
WT_ORDER  = ["Strength", "Cardio", "Yoga", "Hiit"]

print(f"  Dataset loaded: {len(df):,} rows\n")

# ── Shared axis styler ────────────────────────────────────────────────────────

def style_ax(ax, title, xlabel="", ylabel=""):
    ax.set_facecolor(BG)
    ax.set_title(title, fontsize=TITLE_FS, fontweight="bold",
                 color="#343A40", pad=14)
    ax.set_xlabel(xlabel, fontsize=LABEL_FS, color="#495057", labelpad=8)
    ax.set_ylabel(ylabel, fontsize=LABEL_FS, color="#495057", labelpad=8)
    ax.tick_params(labelsize=TICK_FS, colors="#495057")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID_COL)
    ax.yaxis.grid(True, color=GRID_COL, linewidth=0.9, linestyle="--")
    ax.set_axisbelow(True)

def save(fig, filename, label):
    fig.savefig(OUT_DIR + filename, dpi=DPI, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"✓ Saved [{label}] → {OUT_DIR + filename}")

# ─────────────────────────────────────────────────────────────────────────────
# CHART 1 — BAR GRAPH
# Workout Type Count by Gender  (RQ9)
# ─────────────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 6), facecolor=BG)

wt_gender = pd.crosstab(df["workout_type"], df["gender"]).reindex(WT_ORDER)
x     = np.arange(len(WT_ORDER))
width = 0.36

bars_f = ax.bar(x - width / 2, wt_gender["Female"], width,
                label="Female", color=PALETTE[0], edgecolor="white", linewidth=0.8)
bars_m = ax.bar(x + width / 2, wt_gender["Male"],   width,
                label="Male",   color=PALETTE[1], edgecolor="white", linewidth=0.8)

for bar in list(bars_f) + list(bars_m):
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, h + 2,
            str(int(h)), ha="center", va="bottom",
            fontsize=ANNOT_FS, color="#343A40", fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(WT_ORDER, fontsize=TICK_FS + 1)
ax.legend(fontsize=11, framealpha=0.6, loc="upper right")

# Chi-square annotation
ct = pd.crosstab(df["workout_type"], df["gender"])
chi2, p_chi, dof, _ = stats.chi2_contingency(ct)
ax.text(0.01, 0.97,
        f"χ² = {chi2:.2f}   p = {p_chi:.4f}\n({'Significant' if p_chi < 0.05 else 'Not significant'} gender difference)",
        transform=ax.transAxes, fontsize=ANNOT_FS, va="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.75))

style_ax(ax,
         "Bar Graph: Workout Type Count by Gender  (RQ9)",
         "Workout Type", "Number of Users")

fig.tight_layout()
save(fig, "chart_1_bar_workout_by_gender.png", "Bar Graph")

# ─────────────────────────────────────────────────────────────────────────────
# CHART 2 — LINE GRAPH
# Mean BMI & Fat% Trend across Experience Levels  (RQ8)
# ─────────────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 6), facecolor=BG)
ax_r = ax.twinx()

bmi_means = [df[df["experience_label"] == lvl]["bmi"].mean()            for lvl in EXP_ORDER]
fat_means = [df[df["experience_label"] == lvl]["fat_percentage"].mean() for lvl in EXP_ORDER]
bmi_sems  = [df[df["experience_label"] == lvl]["bmi"].sem()             for lvl in EXP_ORDER]
fat_sems  = [df[df["experience_label"] == lvl]["fat_percentage"].sem()  for lvl in EXP_ORDER]

# BMI line (left axis)
ax.plot(EXP_ORDER, bmi_means, marker="o", linewidth=2.5,
        color=PALETTE[2], label="Mean BMI", markersize=10, zorder=3)
ax.fill_between(EXP_ORDER,
                [m - e for m, e in zip(bmi_means, bmi_sems)],
                [m + e for m, e in zip(bmi_means, bmi_sems)],
                alpha=0.15, color=PALETTE[2])

# Fat% line (right axis)
ax_r.plot(EXP_ORDER, fat_means, marker="s", linewidth=2.5,
          color=PALETTE[3], label="Mean Fat %", markersize=10,
          linestyle="--", zorder=3)
ax_r.fill_between(EXP_ORDER,
                  [m - e for m, e in zip(fat_means, fat_sems)],
                  [m + e for m, e in zip(fat_means, fat_sems)],
                  alpha=0.15, color=PALETTE[3])

# Data labels
for i, (bm, fm) in enumerate(zip(bmi_means, fat_means)):
    ax.annotate(f"{bm:.2f}",   (EXP_ORDER[i], bm),
                textcoords="offset points", xytext=(0, 12),
                ha="center", fontsize=ANNOT_FS + 1, color=PALETTE[2], fontweight="bold")
    ax_r.annotate(f"{fm:.2f}", (EXP_ORDER[i], fm),
                  textcoords="offset points", xytext=(0, -18),
                  ha="center", fontsize=ANNOT_FS + 1, color=PALETTE[3], fontweight="bold")

# Style
ax.set_facecolor(BG)
ax.set_title("Line Graph: BMI & Fat % Trend by Experience Level  (RQ8)",
             fontsize=TITLE_FS, fontweight="bold", color="#343A40", pad=14)
ax.set_xlabel("Experience Level", fontsize=LABEL_FS, color="#495057", labelpad=8)
ax.set_ylabel("Mean BMI",   fontsize=LABEL_FS, color=PALETTE[2], labelpad=8)
ax_r.set_ylabel("Mean Fat %", fontsize=LABEL_FS, color=PALETTE[3], labelpad=8)
ax.tick_params(labelsize=TICK_FS, colors="#495057")
ax_r.tick_params(labelsize=TICK_FS, colors="#495057")
ax.spines[["top", "right"]].set_visible(False)
ax_r.spines[["top", "left"]].set_visible(False)
ax.spines[["left", "bottom"]].set_color(GRID_COL)
ax.yaxis.grid(True, color=GRID_COL, linewidth=0.9, linestyle="--")
ax.set_axisbelow(True)

lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax_r.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2,
          fontsize=11, framealpha=0.6, loc="upper right")

# Kruskal-Wallis — run independently for BMI and Fat%
groups_bmi = [df[df["experience_label"] == lvl]["bmi"].dropna() for lvl in EXP_ORDER]
h_bmi, p_bmi = stats.kruskal(*groups_bmi)

groups_fat_kw = [df[df["experience_label"] == lvl]["fat_percentage"].dropna() for lvl in EXP_ORDER]
h_fat_kw, p_fat_kw = stats.kruskal(*groups_fat_kw)

ax.text(0.01, 0.03,
        f"Kruskal-Wallis — BMI : H={h_bmi:.2f}, p={p_bmi:.4f}\n"
        f"Kruskal-Wallis — Fat%: H={h_fat_kw:.2f}, p={p_fat_kw:.4f}",
        transform=ax.transAxes, fontsize=ANNOT_FS, va="bottom",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.75))

fig.tight_layout()
save(fig, "chart_2_line_bmi_fat_by_experience.png", "Line Graph")

# ─────────────────────────────────────────────────────────────────────────────
# CHART 3 — PIE CHART
# Overall Workout Type Distribution  (RQ9)
# ─────────────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 8), facecolor=BG)
ax.set_facecolor(BG)

wt_counts = df["workout_type"].value_counts().reindex(WT_ORDER)
explode   = [0.05] * len(WT_ORDER)

wedges, texts, autotexts = ax.pie(
    wt_counts,
    labels=WT_ORDER,
    autopct="%1.1f%%",
    startangle=140,
    explode=explode,
    colors=PALETTE,
    wedgeprops=dict(edgecolor="white", linewidth=1.8),
    pctdistance=0.78,
    labeldistance=1.13,
    textprops={"fontsize": TICK_FS + 1},
)

for at in autotexts:
    at.set_fontsize(ANNOT_FS + 1)
    at.set_color("white")
    at.set_fontweight("bold")

ax.set_title("Pie Chart: Overall Workout Type Distribution  (RQ9)",
             fontsize=TITLE_FS, fontweight="bold", color="#343A40", pad=18)

count_labels = [f"{wt}  (n={wt_counts[wt]:,})" for wt in WT_ORDER]
legend_patches = [mpatches.Patch(color=PALETTE[i], label=count_labels[i])
                  for i in range(len(WT_ORDER))]
ax.legend(handles=legend_patches, loc="lower center",
          bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize=10, framealpha=0.6)

fig.tight_layout()
save(fig, "chart_3_pie_workout_distribution.png", "Pie Chart")

# ─────────────────────────────────────────────────────────────────────────────
# CHART 4 — SCATTER PLOT
# Resting BPM vs Calories Burned, coloured by Experience Level  (RQ2)
# ─────────────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 6), facecolor=BG)

# Rasterise everything below zorder=1 (the scatter dots) so the file stays
# lightweight as the dataset grows, while keeping vector text/lines crisp.
ax.set_rasterization_zorder(1)

exp_colours = {
    "Beginner":     PALETTE[0],
    "Intermediate": PALETTE[1],
    "Advanced":     PALETTE[2],
}

for lvl in EXP_ORDER:
    grp = df[df["experience_label"] == lvl]
    ax.scatter(grp["resting_bpm"], grp["calories_burned"],
               alpha=0.40, s=22, color=exp_colours[lvl],
               label=lvl, rasterized=True, edgecolors="none")

# Regression line
mask   = df[["resting_bpm", "calories_burned"]].notna().all(axis=1)
x_data = df.loc[mask, "resting_bpm"].values
y_data = df.loc[mask, "calories_burned"].values
slope, intercept, r_val, p_val, _ = stats.linregress(x_data, y_data)
x_line = np.linspace(x_data.min(), x_data.max(), 300)
ax.plot(x_line, slope * x_line + intercept,
        color="#212529", linewidth=2.0, linestyle="-",
        label=f"Regression line  (r = {r_val:+.3f})", zorder=5)

# r & p annotation
ax.text(0.97, 0.05,
        f"r  = {r_val:+.4f}\np  = {p_val:.4f}\nn  = {mask.sum():,}",
        transform=ax.transAxes, fontsize=ANNOT_FS + 1,
        ha="right", va="bottom", color="#343A40",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.80))

ax.legend(fontsize=10, framealpha=0.6, loc="upper left", markerscale=1.6)
style_ax(ax,
         "Scatter Plot: Resting BPM vs Calories Burned  (RQ2)",
         "Resting BPM", "Calories Burned")

fig.tight_layout()
save(fig, "chart_4_scatter_restingbpm_calories.png", "Scatter Plot")

# ─────────────────────────────────────────────────────────────────────────────
# CHART 5 — SUMMARY TABLE
# Key calculated values across all Research Questions
# ─────────────────────────────────────────────────────────────────────────────

# ── Pre-compute all values needed for the table ───────────────────────────────

# RQ1
freq_by_exp = df.groupby("experience_label")["workout_frequency_days_per_week"].mean()
top_type_by_exp = df.groupby("experience_label")["workout_type"].agg(
    lambda x: x.value_counts().idxmax()
)

# RQ2
mask_rq2   = df[["resting_bpm", "calories_burned"]].notna().all(axis=1)
r_cal, p_cal = stats.pearsonr(
    df.loc[mask_rq2, "resting_bpm"], df.loc[mask_rq2, "calories_burned"]
)
mask_dur   = df[["resting_bpm", "session_duration_hours"]].notna().all(axis=1)
r_dur, p_dur = stats.pearsonr(
    df.loc[mask_dur, "resting_bpm"], df.loc[mask_dur, "session_duration_hours"]
)

# RQ3
r_bmi_fat, p_bmi_fat = stats.pearsonr(df["bmi"], df["fat_percentage"])
df["high_risk"] = (df["bmi"] > 25) & (df["fat_percentage"] > 25)
hr_freq  = df[df["high_risk"]]["workout_frequency_days_per_week"].mean()
nlr_freq = df[~df["high_risk"]]["workout_frequency_days_per_week"].mean()

# RQ5 — Cardiovascular baseline
cv_stats = {
    col: {"mean": df[col].mean(), "median": df[col].median(), "mode": df[col].mode()[0]}
    for col in ["resting_bpm", "avg_bpm", "max_bpm"]
}

# RQ6 — Workout variation
act_stats = {
    col: {"std": df[col].std(), "min": df[col].min(), "max": df[col].max()}
    for col in ["session_duration_hours", "calories_burned", "workout_frequency_days_per_week"]
}

# RQ7 — Correlations body comp vs activity
r_bmi_freq, _  = stats.pearsonr(df["bmi"], df["workout_frequency_days_per_week"])
r_fat_dur,  _  = stats.pearsonr(df["fat_percentage"], df["session_duration_hours"])
r_rest_max, _  = stats.pearsonr(df["resting_bpm"], df["max_bpm"])
r_rest_cal, _  = stats.pearsonr(df["resting_bpm"], df["calories_burned"])

# RQ8 — Experience trends
bmi_means_exp = {lvl: df[df["experience_label"] == lvl]["bmi"].mean() for lvl in EXP_ORDER}
fat_means_exp = {lvl: df[df["experience_label"] == lvl]["fat_percentage"].mean() for lvl in EXP_ORDER}

# RQ9 — Workout type distribution
wt_pct = (df["workout_type"].value_counts(normalize=True) * 100).reindex(WT_ORDER)

# ── Build table rows ──────────────────────────────────────────────────────────

rows = [
    # (RQ, Metric / Description, Value)
    # ── RQ1
    ("RQ1", "Beginner   — top workout type",              top_type_by_exp.get("Beginner", "N/A")),
    ("RQ1", "Intermediate — top workout type",            top_type_by_exp.get("Intermediate", "N/A")),
    ("RQ1", "Advanced   — top workout type",              top_type_by_exp.get("Advanced", "N/A")),
    ("RQ1", "Beginner   — mean freq (days/wk)",           f"{freq_by_exp.get('Beginner', 0):.2f}"),
    ("RQ1", "Intermediate — mean freq (days/wk)",         f"{freq_by_exp.get('Intermediate', 0):.2f}"),
    ("RQ1", "Advanced   — mean freq (days/wk)",           f"{freq_by_exp.get('Advanced', 0):.2f}"),
    # ── RQ2
    ("RQ2", "resting_bpm vs calories_burned  (r)",        f"{r_cal:+.4f}  (p={p_cal:.4f})"),
    ("RQ2", "resting_bpm vs session_duration (r)",        f"{r_dur:+.4f}  (p={p_dur:.4f})"),
    # ── RQ3
    ("RQ3", "BMI vs Fat%  overall  (r)",                  f"{r_bmi_fat:+.4f}  (p={p_bmi_fat:.4f})"),
    ("RQ3", "High-risk mean workout freq (days/wk)",      f"{hr_freq:.2f}"),
    ("RQ3", "Normal-risk mean workout freq (days/wk)",    f"{nlr_freq:.2f}"),
    # ── RQ5
    ("RQ5", "resting_bpm — Mean / Median / Mode",
     f"{cv_stats['resting_bpm']['mean']:.1f} / {cv_stats['resting_bpm']['median']:.1f} / {cv_stats['resting_bpm']['mode']:.0f}"),
    ("RQ5", "avg_bpm     — Mean / Median / Mode",
     f"{cv_stats['avg_bpm']['mean']:.1f} / {cv_stats['avg_bpm']['median']:.1f} / {cv_stats['avg_bpm']['mode']:.0f}"),
    ("RQ5", "max_bpm     — Mean / Median / Mode",
     f"{cv_stats['max_bpm']['mean']:.1f} / {cv_stats['max_bpm']['median']:.1f} / {cv_stats['max_bpm']['mode']:.0f}"),
    # ── RQ6
    ("RQ6", "session_duration_hours   — Std / Min / Max",
     f"{act_stats['session_duration_hours']['std']:.3f} / {act_stats['session_duration_hours']['min']:.2f} / {act_stats['session_duration_hours']['max']:.2f}"),
    ("RQ6", "calories_burned          — Std / Min / Max",
     f"{act_stats['calories_burned']['std']:.1f} / {act_stats['calories_burned']['min']:.0f} / {act_stats['calories_burned']['max']:.0f}"),
    ("RQ6", "workout_freq (days/wk)   — Std / Min / Max",
     f"{act_stats['workout_frequency_days_per_week']['std']:.3f} / {act_stats['workout_frequency_days_per_week']['min']:.0f} / {act_stats['workout_frequency_days_per_week']['max']:.0f}"),
    # ── RQ7
    ("RQ7", "BMI vs workout_frequency          (r)",      f"{r_bmi_freq:+.4f}"),
    ("RQ7", "Fat% vs session_duration          (r)",      f"{r_fat_dur:+.4f}"),
    ("RQ7", "resting_bpm vs max_bpm            (r)",      f"{r_rest_max:+.4f}"),
    ("RQ7", "resting_bpm vs calories_burned    (r)",      f"{r_rest_cal:+.4f}"),
    # ── RQ8
    ("RQ8", "BMI   Beginner / Intermediate / Advanced",
     f"{bmi_means_exp['Beginner']:.2f} / {bmi_means_exp['Intermediate']:.2f} / {bmi_means_exp['Advanced']:.2f}"),
    ("RQ8", "Fat%  Beginner / Intermediate / Advanced",
     f"{fat_means_exp['Beginner']:.2f} / {fat_means_exp['Intermediate']:.2f} / {fat_means_exp['Advanced']:.2f}"),
    ("RQ8", "Kruskal-Wallis BMI  (H / p)",                f"{h_bmi:.2f} / {p_bmi:.4f}"),
    ("RQ8", "Kruskal-Wallis Fat% (H / p)",                f"{h_fat_kw:.2f} / {p_fat_kw:.4f}"),
    # ── RQ9
    ("RQ9", "Strength  — overall share",                  f"{wt_pct.get('Strength', 0):.1f}%"),
    ("RQ9", "Cardio    — overall share",                  f"{wt_pct.get('Cardio', 0):.1f}%"),
    ("RQ9", "Yoga      — overall share",                  f"{wt_pct.get('Yoga', 0):.1f}%"),
    ("RQ9", "HIIT      — overall share",                  f"{wt_pct.get('Hiit', 0):.1f}%"),
    ("RQ9", "Chi-square  (χ² / p)",                       f"{chi2:.2f} / {p_chi:.4f}"),
]

col_labels = ["RQ", "Metric / Description", "Calculated Value"]
cell_data  = [[r[0], r[1], r[2]] for r in rows]

# ── Colour-band rows by RQ ────────────────────────────────────────────────────
RQ_COLOURS = {
    "RQ1": "#EAF0FB", "RQ2": "#FEF3E8", "RQ3": "#EBF6EE",
    "RQ5": "#F3EBF9", "RQ6": "#E8F7FB",
    "RQ7": "#FDF8E1", "RQ8": "#FDE8F0", "RQ9": "#EDF7EE",
}
row_colours = [[RQ_COLOURS.get(r[0], "#FFFFFF")] * 3 for r in rows]

# ── Draw table ────────────────────────────────────────────────────────────────
n_rows  = len(rows)
fig_h   = max(10, n_rows * 0.38 + 1.8)
fig, ax = plt.subplots(figsize=(16, fig_h), facecolor=BG)
ax.axis("off")

ax.set_title(
    "Summary Table: Key Calculated Values Across All Research Questions",
    fontsize=15, fontweight="bold", color="#212529",
    pad=16, loc="center"
)

tbl = ax.table(
    cellText=cell_data,
    colLabels=col_labels,
    cellLoc="left",
    loc="center",
    cellColours=row_colours,
)

tbl.auto_set_font_size(False)
tbl.set_fontsize(8.5)
tbl.auto_set_column_width([0, 1, 2])

# Style header row
for col_idx in range(3):
    cell = tbl[0, col_idx]
    cell.set_facecolor("#343A40")
    cell.set_text_props(color="white", fontweight="bold", fontsize=9.5)
    cell.set_height(0.045)

# Style data rows — tighten height and left-pad text
col_widths = [0.06, 0.44, 0.30]
for row_idx in range(1, n_rows + 1):
    for col_idx in range(3):
        cell = tbl[row_idx, col_idx]
        cell.set_height(0.032)
        cell.set_width(col_widths[col_idx])
        cell.PAD = 0.012
        # Bold the RQ label column
        if col_idx == 0:
            cell.set_text_props(fontweight="bold", color="#343A40")
        # Right-align value column
        if col_idx == 2:
            cell._loc = "right"
            cell.set_text_props(color="#1A1A2E", fontfamily="monospace")

# Outer border
for (r, c), cell in tbl.get_celld().items():
    cell.set_linewidth(0.4)
    cell.set_edgecolor(GRID_COL)

fig.tight_layout(pad=1.2)
save(fig, "chart_5_summary_table.png", "Summary Table")

# ── Done ──────────────────────────────────────────────────────────────────────
print(f"\n  All 5 outputs saved to {OUT_DIR}\n")