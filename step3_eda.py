# ============================================================
# STEP 3: EXPLORATORY DATA ANALYSIS (EDA)
# Viral Music Trend Prediction
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE  = "data/youtube_music_cleaned.csv"
OUTPUT_DIR  = "outputs/eda"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set global plot style
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"]   = "white"
plt.rcParams["axes.grid"]        = True
plt.rcParams["grid.alpha"]       = 0.3
plt.rcParams["font.size"]        = 10
sns.set_theme(style="whitegrid", palette="husl")

# ============================================================
# LOAD DATA
# ============================================================

def load_data(filepath):
    print("\n📂 Loading cleaned dataset...")
    df = pd.read_csv(filepath)
    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")
    print(f"   Shape   : {df.shape[0]} rows x {df.shape[1]} columns")
    return df

# ============================================================
# STEP 3A: UNIVARIATE ANALYSIS - HISTOGRAMS
# ============================================================

def plot_histograms(df):
    print("\n" + "=" * 60)
    print("STEP 3A: UNIVARIATE ANALYSIS - HISTOGRAMS")
    print("=" * 60)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Univariate Analysis - Distribution of Key Metrics",
                 fontsize=16, fontweight="bold", y=1.01)

    cols   = ["views", "likes", "comments", "views_per_day"]
    colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12"]
    titles = ["Views Distribution", "Likes Distribution",
              "Comments Distribution", "Views Per Day Distribution"]

    for idx, (col, color, title) in enumerate(zip(cols, colors, titles)):
        ax  = axes[idx // 2][idx % 2]
        data = df[col].dropna()

        # Histogram with KDE
        ax.hist(data, bins=40, color=color, alpha=0.7,
                edgecolor="white", linewidth=0.5)

        # Add mean and median lines
        mean_val   = data.mean()
        median_val = data.median()

        ax.axvline(mean_val,   color="red",    linestyle="--",
                   linewidth=2, label=f"Mean: {mean_val:,.0f}")
        ax.axvline(median_val, color="black",  linestyle="-.",
                   linewidth=2, label=f"Median: {median_val:,.0f}")

        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel(col.replace("_", " ").title(), fontsize=11)
        ax.set_ylabel("Frequency", fontsize=11)
        ax.legend(fontsize=9)

        # Stats box
        stats_text = (f"Mean:   {mean_val:>12,.0f}\n"
                      f"Median: {median_val:>12,.0f}\n"
                      f"Std:    {data.std():>12,.0f}\n"
                      f"Skew:   {data.skew():>12.2f}")
        ax.text(0.97, 0.97, stats_text,
                transform=ax.transAxes,
                verticalalignment="top",
                horizontalalignment="right",
                fontsize=8,
                bbox=dict(boxstyle="round", facecolor="lightyellow",
                          alpha=0.8))

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/3a_histograms.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    
    print(f"   ✅ Saved: {path}")
    plt.close()

def plot_heatmap(df):
    print("\n" + "=" * 60)
    print("STEP 3B: CORRELATION HEATMAP")
    print("=" * 60)

    # Select specific important columns for a cleaner heatmap
    important_cols = ["views", "likes", "comments", "views_per_day", 
                      "engagement_rate", "like_view_ratio", "is_viral"]
    
    # Filter to only columns that actually exist in df
    plot_cols = [c for c in important_cols if c in df.columns]
    
    # Calculate correlation matrix
    corr = df[plot_cols].corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.suptitle("Correlation Heatmap of Key Variables", fontsize=16, fontweight="bold")
    
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r", center=0, 
                vmin=-1, vmax=1, square=True, linewidths=.5, ax=ax,
                cbar_kws={"shrink": .8})
    
    # Rotate x labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/3e_heatmap_full.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"   ✅ Saved: {path}")

# ============================================================
# STEP 3C: KEY OBSERVATIONS SUMMARY
# ============================================================

def print_key_observations(df):
    print("\n" + "=" * 60)
    print("STEP 3C: KEY OBSERVATIONS & INSIGHTS")
    print("=" * 60)

    print("\n📌 OBSERVATION 1: Dataset Overview")
    print(f"   Total Songs Analyzed : {len(df)}")
    print(f"   Viral Songs          : {df['is_viral'].sum()} ({df['is_viral'].mean()*100:.1f}%)")
    print(f"   Languages Covered    : {df['language_name'].nunique()}")
    print(f"   Date Range           : {df['published_at'].min().date()} to {df['published_at'].max().date()}")

    print("\n📌 OBSERVATION 2: Most Dominant Language")
    top_lang = df["language_name"].value_counts()
    print(f"   #1 Language : {top_lang.index[0]} ({top_lang.iloc[0]} songs)")
    print(f"   #2 Language : {top_lang.index[1]} ({top_lang.iloc[1]} songs)")
    print(f"   #3 Language : {top_lang.index[2]} ({top_lang.iloc[2]} songs)")

    print("\n📌 OBSERVATION 3: Language with Most Viral Songs")
    viral_lang = df[df["is_viral"] == 1]["language_name"].value_counts()
    if len(viral_lang) > 0:
        print(f"   Most Viral Language : {viral_lang.index[0]} ({viral_lang.iloc[0]} viral songs)")

    print("\n📌 OBSERVATION 4: Top 5 Most Viral Songs Right Now")
    top5 = df.nlargest(5, "views_per_day")[
        ["title", "channel_name", "language_name",
         "views", "views_per_day", "viral_category"]
    ]
    print(top5.to_string())

    print("\n📌 OBSERVATION 5: Engagement Analysis")
    print(f"   Avg Engagement Rate  : {df['engagement_rate'].mean():.2f}%")
    print(f"   Max Engagement Rate  : {df['engagement_rate'].max():.2f}%")
    print(f"   Avg Like/View Ratio  : {df['like_view_ratio'].mean():.2f}%")

    print("\n📌 OBSERVATION 6: Views Statistics")
    print(f"   Mean Views           : {df['views'].mean():>15,.0f}")
    print(f"   Median Views         : {df['views'].median():>15,.0f}")
    print(f"   Max Views            : {df['views'].max():>15,.0f}")
    print(f"   Min Views            : {df['views'].min():>15,.0f}")

    print("\n📌 OBSERVATION 7: Viral Category Distribution")
    print(df["viral_category"].value_counts().to_string())

    print("\n📌 OBSERVATION 8: Best Day to Publish")
    day_views = df.groupby("publish_day_name")["views"]\
        .mean().sort_values(ascending=False)
    print(f"   Best Day  : {day_views.index[0]} (Avg: {day_views.iloc[0]:,.0f} views)")
    print(f"   Worst Day : {day_views.index[-1]} (Avg: {day_views.iloc[-1]:,.0f} views)")

    print("\n📌 OBSERVATION 9: Correlation Insights")
    corr = df[["views", "likes", "comments",
               "views_per_day", "is_viral"]].corr()
    print("   Correlation with is_viral:")
    viral_corr = corr["is_viral"].drop("is_viral")\
        .sort_values(ascending=False)
    for col, val in viral_corr.items():
        print(f"   {col:25s} : {val:.4f}")

# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    print("=" * 60)
    print("   🎵 VIRAL MUSIC TREND PREDICTION")
    print("   STEP 3: EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 60)

    if not os.path.exists(INPUT_FILE):
        print(f"\n❌ ERROR: '{INPUT_FILE}' not found!")
        print("   Please run step2_data_cleaning.py first!")
        return

    # Load data
    df = load_data(INPUT_FILE)

    # Run all EDA steps
    print("\n🔄 Running all EDA steps...")

    plot_histograms(df)
    plot_heatmap(df)
    print_key_observations(df)

    print("\n" + "=" * 60)
    print("✅ STEP 3 COMPLETE!")
    print(f"   All graphs saved in : {OUTPUT_DIR}/")
    print("   Files generated:")
    for f in os.listdir(OUTPUT_DIR):
        print(f"   📊 {f}")
    print("   ➡️  Next: python step4_feature_engineering.py")
    print("=" * 60)


if __name__ == "__main__":
    main()