# ============================================================
# STEP 7: VISUALIZATION & DASHBOARD
# Viral Music Trend Prediction
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import os
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE    = "data/youtube_music_timeseries.csv"
FALLBACK_FILE = "data/youtube_music_stats.csv"
OUTPUT_DIR    = "outputs/visualizations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# CUSTOM COLOR PALETTE
# ============================================================

COLORS = {
    "primary"    : "#e74c3c",
    "secondary"  : "#3498db",
    "success"    : "#2ecc71",
    "warning"    : "#f39c12",
    "purple"     : "#9b59b6",
    "dark"       : "#2c3e50",
    "light"      : "#ecf0f1",
    "pink"       : "#fd79a8",
    "teal"       : "#00b894",
    "orange"     : "#e17055",
}

VIRAL_COLORS = {
    "LEGENDARY"  : "#8e44ad",
    "MEGA VIRAL" : "#e74c3c",
    "VIRAL"      : "#e67e22",
    "TRENDING"   : "#f1c40f",
    "RISING"     : "#2ecc71",
    "NORMAL"     : "#95a5a6",
}

# Global plot style
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"]   = "#fafafa"
plt.rcParams["axes.grid"]        = True
plt.rcParams["grid.alpha"]       = 0.3
plt.rcParams["grid.linestyle"]   = "--"
plt.rcParams["font.family"]      = "DejaVu Sans"
plt.rcParams["font.size"]        = 10
sns.set_theme(style="whitegrid", palette="husl")

# ============================================================
# LOAD DATA
# ============================================================

def load_data(filepath):
    print("\n📂 Loading dataset...")
    df = pd.read_csv(filepath)
    df["published_at"] = pd.to_datetime(
        df["published_at"], errors="coerce"
    )
    print(f"   Shape   : {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"   Columns : {list(df.columns[:8])}...")
    return df

# ============================================================
# STEP 7A: TREND GRAPHS
# ============================================================

def plot_trend_graphs(df):
    print("\n" + "=" * 60)
    print("STEP 7A: TREND GRAPHS")
    print("=" * 60)

    # Prepare time series
    df["month"] = df["published_at"].dt.to_period("M")\
        .dt.start_time
    df["week"]  = df["published_at"].dt.to_period("W")\
        .dt.start_time

    monthly = df.groupby("month").agg(
        avg_views     = ("views",          "mean"),
    ).reset_index().sort_values("month")

    weekly = df.groupby("week").agg(
        avg_views   = ("views",       "mean"),
        viral_count = ("is_viral",    "sum"),
    ).reset_index().sort_values("week")

    # ---- FIGURE 1: Monthly Views Trend ----
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    fig1.patch.set_facecolor("white")
    ax1.fill_between(monthly["month"],
                     monthly["avg_views"],
                     alpha=0.15, color=COLORS["secondary"])
    ax1.plot(monthly["month"], monthly["avg_views"],
             color=COLORS["secondary"], linewidth=2.5,
             marker="o", markersize=5, zorder=5)

    # Add trend line
    x_num = np.arange(len(monthly))
    if len(x_num) > 1:
        z = np.polyfit(x_num, monthly["avg_views"], 1)
        p = np.poly1d(z)
        ax1.plot(monthly["month"], p(x_num),
                 color=COLORS["primary"],
                 linestyle="--", linewidth=2,
                 label="Trend Line", alpha=0.8)

    ax1.set_title("Monthly Average Views Trend",
                  fontsize=13, fontweight="bold",
                  color=COLORS["dark"])
    ax1.set_xlabel("Month", fontsize=10)
    ax1.set_ylabel("Average Views", fontsize=10)
    ax1.tick_params(axis="x", rotation=30)
    ax1.legend(fontsize=9)
    ax1.yaxis.set_major_formatter(
        mticker.FuncFormatter(
            lambda x, p: f"{x/1e6:.1f}M" if x >= 1e6
            else f"{x/1e3:.0f}K"
        )
    )
    plt.tight_layout()
    path1 = f"{OUTPUT_DIR}/7a_monthly_views.png"
    plt.savefig(path1, dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print(f"   ✅ Saved: {path1}")

    # --- Chart 2: Viral Count Trend ---
    fig2, ax3 = plt.subplots(figsize=(10, 6))
    ax3.bar(weekly["week"], weekly["viral_count"],
            color=COLORS["primary"], alpha=0.7,
            width=5, label="Viral Songs/Week")
    ax3_twin = ax3.twinx()
    ax3_twin.plot(weekly["week"], weekly["avg_views"],
                  color=COLORS["secondary"],
                  linewidth=2, label="Avg Views",
                  alpha=0.8)
    ax3.set_title("Weekly Viral Count & Average Views",
                  fontsize=13, fontweight="bold",
                  color=COLORS["dark"])
    ax3.set_xlabel("Week", fontsize=10)
    ax3.set_ylabel("Viral Songs Count",
                   color=COLORS["primary"], fontsize=10)
    ax3_twin.set_ylabel("Average Views",
                        color=COLORS["secondary"],
                        fontsize=10)
    ax3.tick_params(axis="x", rotation=30)

    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2,
               fontsize=9, loc="upper left")

    plt.tight_layout()
    path2 = f"{OUTPUT_DIR}/7b_weekly_viral.png"
    plt.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"   ✅ Saved: {path2}")

# ============================================================
# STEP 7G: FINAL VISUAL SUMMARY
# ============================================================

def print_visual_summary(df):
    print("\n" + "=" * 60)
    print("STEP 7G: FINAL VISUAL SUMMARY")
    print("=" * 60)

    print("\n📊 DATASET SUMMARY FOR VISUALIZATION:")
    print(f"   Total Songs       : {len(df):,}")
    print(f"   Viral Songs       : {df['is_viral'].sum():,}")
    print(f"   Languages         : {df['language_name'].nunique()}")
    print(f"   Regions           : {df['region_fetched'].nunique()}")
    print(f"   Avg Views         : {df['views'].mean():>15,.0f}")
    print(f"   Avg Likes         : {df['likes'].mean():>15,.0f}")
    print(f"   Avg Viral Score   : {df['viral_score'].mean():.6f}")
    print(f"   Avg Virality Index: {df['virality_index'].mean():.6f}")

    print("\n🔥 TOP 10 VIRAL SONGS RIGHT NOW:")
    top10 = df.nlargest(10, "virality_index")[
        ["title", "channel_name", "language_name",
         "views", "viral_score",
         "virality_index", "virality_prediction"]
    ]
    print(top10.to_string())

    print("\n🌍 LANGUAGE DOMINATION:")
    lang_dom = df.groupby("language_name").agg(
        songs       = ("video_id",      "count"),
        avg_viral   = ("viral_score",   "mean"),
        viral_songs = ("is_viral",      "sum")
    ).sort_values("avg_viral", ascending=False).head(10)
    print(lang_dom.round(4).to_string())

    print("\n📁 ALL VISUALIZATION FILES GENERATED:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        size = os.path.getsize(f"{OUTPUT_DIR}/{f}")
        print(f"   📊 {f:<45} ({size/1024:.1f} KB)")

# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    print("=" * 60)
    print("   🎵 VIRAL MUSIC TREND PREDICTION")
    print("   STEP 7: VISUALIZATION & DASHBOARD")
    print("=" * 60)

    if not os.path.exists(INPUT_FILE):
        if os.path.exists(FALLBACK_FILE):
            print(f"\n⚠️ WARNING: '{INPUT_FILE}' not found.")
            print(f"   Falling back to '{FALLBACK_FILE}'.")
            print("   If you want the dedicated time-series file, run step6_time_series.py first.")
            input_path = FALLBACK_FILE
        else:
            print(f"\n❌ ERROR: '{INPUT_FILE}' not found!")
            print("   Please run step6_time_series.py first!")
            return
    else:
        input_path = INPUT_FILE

    # Load data
    df = load_data(input_path)

    # Run all visualizations
    print("\n🎨 Generating all visualizations...")

    plot_trend_graphs(df)
    print_visual_summary(df)

    print("\n" + "=" * 60)
    print("✅ STEP 7 COMPLETE!")
    print(f"   All graphs saved in : {OUTPUT_DIR}/")
    print("\n   Files Generated:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        print(f"   📊 {f}")
    print("\n   ➡️  Next: python step8_results.py")
    print("=" * 60)


if __name__ == "__main__":
    main()