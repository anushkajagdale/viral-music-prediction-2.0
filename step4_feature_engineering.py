# ============================================================
# STEP 4: FEATURE ENGINEERING
# Viral Music Trend Prediction
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE  = "data/youtube_music_cleaned.csv"
OUTPUT_FILE = "data/youtube_music_featured.csv"
OUTPUT_DIR  = "outputs/feature_engineering"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Plot style
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"]   = "white"
plt.rcParams["axes.grid"]        = True
plt.rcParams["grid.alpha"]       = 0.3
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
# STEP 4A: VIRAL SCORE - CORE FEATURE
# ============================================================

def create_viral_score(df):
    print("\n" + "=" * 60)
    print("STEP 4A: CREATING VIRAL SCORE")
    print("=" * 60)

    print("Calculating Viral Score (50% views, 30% likes, 20% comments)...")

    # Normalize views, likes, comments to 0-1 range
    # Using fresh normalization on cleaned data
    for col in ["views", "likes", "comments"]:
        col_min = df[col].min()
        col_max = df[col].max()
        if col_max - col_min > 0:
            df[f"{col}_norm_vs"] = (df[col] - col_min) / (col_max - col_min)
        else:
            df[f"{col}_norm_vs"] = 0

    # Calculate viral score
    df["viral_score"] = (
        (df["views_norm_vs"]    * 0.50) +
        (df["likes_norm_vs"]    * 0.30) +
        (df["comments_norm_vs"] * 0.20)
    ).round(6)

    # Drop temporary normalization columns
    df.drop(columns=["views_norm_vs",
                      "likes_norm_vs",
                      "comments_norm_vs"], inplace=True)

    print("✅ Viral Score created successfully!")
    print(f"\n   Min Score  : {df['viral_score'].min():.6f}")
    print(f"   Max Score  : {df['viral_score'].max():.6f}")
    print(f"   Mean Score : {df['viral_score'].mean():.6f}")
    print(f"   Std Score  : {df['viral_score'].std():.6f}")

    print("\n📊 Top 10 Songs by Viral Score:")
    top10 = df.nlargest(10, "viral_score")[
        ["title", "channel_name", "language_name",
         "views", "likes", "comments", "viral_score"]
    ]
    print(top10.to_string())

    return df

# ============================================================
# STEP 4B: VIRAL SCORE TIER
# ============================================================

def create_viral_score_tier(df):
    print("\n" + "=" * 60)
    print("STEP 4B: VIRAL SCORE TIER CLASSIFICATION")
    print("=" * 60)

    print("Assigning Viral Score Tiers (LEGENDARY to NORMAL)...")

    def assign_tier(score):
        if   score >= 0.80: return "LEGENDARY"
        elif score >= 0.60: return "MEGA VIRAL"
        elif score >= 0.40: return "VIRAL"
        elif score >= 0.20: return "TRENDING"
        elif score >= 0.10: return "RISING"
        else:               return "NORMAL"

    df["viral_score_tier"] = df["viral_score"].apply(assign_tier)

    print("✅ Viral Score Tier Distribution:")
    tier_counts = df["viral_score_tier"].value_counts()
    tier_pct    = (tier_counts / len(df) * 100).round(2)
    tier_summary = pd.DataFrame({
        "Count"      : tier_counts,
        "Percentage" : tier_pct
    })
    print(tier_summary.to_string())

    return df

# ============================================================
# STEP 4C: ENGAGEMENT MOMENTUM SCORE
# ============================================================

def create_engagement_momentum(df):
    print("\n" + "=" * 60)
    print("STEP 4C: ENGAGEMENT MOMENTUM SCORE")
    print("=" * 60)

    print("Calculating Engagement Momentum...")

    # Normalize components
    for col in ["views_per_day", "like_view_ratio", "comment_view_ratio"]:
        col_min = df[col].min()
        col_max = df[col].max()
        if col_max - col_min > 0:
            df[f"{col}_norm_em"] = (df[col] - col_min) / (col_max - col_min)
        else:
            df[f"{col}_norm_em"] = 0

    df["engagement_momentum"] = (
        (df["views_per_day_norm_em"]      * 0.60) +
        (df["like_view_ratio_norm_em"]    * 0.25) +
        (df["comment_view_ratio_norm_em"] * 0.15)
    ).round(6)

    # Drop temp columns
    df.drop(columns=[
        "views_per_day_norm_em",
        "like_view_ratio_norm_em",
        "comment_view_ratio_norm_em"
    ], inplace=True)

    print("✅ Engagement Momentum Score created!")
    print(f"\n   Min : {df['engagement_momentum'].min():.6f}")
    print(f"   Max : {df['engagement_momentum'].max():.6f}")
    print(f"   Mean: {df['engagement_momentum'].mean():.6f}")

    return df

# ============================================================
# STEP 4D: TIME-BASED FEATURES
# ============================================================

def create_time_features(df):
    print("\n" + "=" * 60)
    print("STEP 4D: TIME-BASED FEATURES")
    print("=" * 60)

    # Season of release
    def get_season(month):
        if   pd.isna(month):    return "Unknown"
        elif month in [12,1,2]: return "Winter"
        elif month in [3,4,5]:  return "Spring"
        elif month in [6,7,8]:  return "Summer"
        else:                   return "Autumn"

    df["release_season"] = df["publish_month"].apply(get_season)

    # Weekend vs Weekday release
    df["is_weekend_release"] = df["publish_dow"].apply(
        lambda x: 1 if pd.notna(x) and x >= 5 else 0
    )

    # Video age category
    def age_category(days):
        if   pd.isna(days):  return "Unknown"
        elif days <= 7:      return "New (0-7 days)"
        elif days <= 30:     return "Recent (8-30 days)"
        elif days <= 90:     return "Established (1-3 months)"
        elif days <= 365:    return "Mature (3-12 months)"
        else:                return "Old (1+ year)"

    df["video_age_category"] = df["days_since_publish"].apply(age_category)

    # Growth rate score
    # Higher views per day relative to age = faster growth
    df["growth_rate_score"] = (
        df["views_per_day"] /
        (df["days_since_publish"].replace(0, 1))
    ).round(6)

    print("✅ Time-based Features Created:")
    print("\n   release_season distribution:")
    print(df["release_season"].value_counts().to_string())

    print("\n   video_age_category distribution:")
    print(df["video_age_category"].value_counts().to_string())

    print("\n   Weekend releases:")
    print(f"   Weekend : {df['is_weekend_release'].sum()}")
    print(f"   Weekday : {(df['is_weekend_release'] == 0).sum()}")

    return df

# ============================================================
# STEP 4E: LANGUAGE POPULARITY SCORE
# ============================================================

def create_language_features(df):
    print("\n" + "=" * 60)
    print("STEP 4E: LANGUAGE POPULARITY FEATURES")
    print("=" * 60)

    print("Calculating Language Popularity features...")

    # Average viral score per language
    lang_viral_avg = df.groupby("language_name")["viral_score"]\
        .mean().rename("lang_avg_viral_score")

    # Count of songs per language
    lang_count = df.groupby("language_name").size()\
        .rename("lang_song_count")

    # Average views per language
    lang_avg_views = df.groupby("language_name")["views"]\
        .mean().rename("lang_avg_views")

    # Viral songs count per language
    lang_viral_count = df[df["is_viral"] == 1]\
        .groupby("language_name").size()\
        .rename("lang_viral_count")

    # Merge all language features
    df = df.merge(lang_viral_avg,  on="language_name", how="left")
    df = df.merge(lang_count,      on="language_name", how="left")
    df = df.merge(lang_avg_views,  on="language_name", how="left")
    df = df.merge(lang_viral_count,on="language_name", how="left")

    df["lang_viral_count"].fillna(0, inplace=True)

    # Language rank by viral score
    lang_rank = lang_viral_avg.rank(ascending=False)\
        .rename("lang_viral_rank")
    df = df.merge(lang_rank, on="language_name", how="left")

    print("✅ Language Features Created!")
    print("\n📊 Top 10 Languages by Average Viral Score:")
    lang_summary = df.groupby("language_name").agg(
        songs          = ("viral_score", "count"),
        avg_viral_score= ("viral_score", "mean"),
        avg_views      = ("views", "mean"),
        viral_count    = ("is_viral", "sum")
    ).sort_values("avg_viral_score", ascending=False).head(10)
    print(lang_summary.round(4).to_string())

    return df

# ============================================================
# STEP 4F: CHANNEL INFLUENCE SCORE
# ============================================================

def create_channel_features(df):
    print("\n" + "=" * 60)
    print("STEP 4F: CHANNEL INFLUENCE FEATURES")
    print("=" * 60)

    print("Calculating Channel Influence Score...")

    # Channel statistics
    ch_avg_views = df.groupby("channel_name")["views"]\
        .mean().rename("channel_avg_views")

    ch_total = df.groupby("channel_name").size()\
        .rename("channel_total_songs")

    ch_viral = df[df["is_viral"] == 1]\
        .groupby("channel_name").size()\
        .rename("channel_viral_count")

    ch_avg_viral = df.groupby("channel_name")["viral_score"]\
        .mean().rename("channel_avg_viral_score")

    # Merge
    df = df.merge(ch_avg_views,   on="channel_name", how="left")
    df = df.merge(ch_total,       on="channel_name", how="left")
    df = df.merge(ch_viral,       on="channel_name", how="left")
    df = df.merge(ch_avg_viral,   on="channel_name", how="left")

    df["channel_viral_count"].fillna(0, inplace=True)

    # Channel influence score (normalized)
    max_views = df["channel_avg_views"].max()
    max_songs = df["channel_total_songs"].max()

    df["channel_influence_score"] = (
        (df["channel_avg_views"]       / max_views * 0.60) +
        (df["channel_total_songs"]     / max_songs * 0.20) +
        (df["channel_avg_viral_score"]             * 0.20)
    ).round(6)

    print("✅ Channel Features Created!")
    print("\n📊 Top 10 Most Influential Channels:")
    ch_summary = df.groupby("channel_name").agg(
        songs             = ("viral_score", "count"),
        avg_viral_score   = ("viral_score", "mean"),
        viral_songs       = ("is_viral", "sum"),
        influence_score   = ("channel_influence_score", "mean")
    ).sort_values("influence_score", ascending=False).head(10)
    print(ch_summary.round(4).to_string())

    return df

# ============================================================
# STEP 4G: COMBINED VIRALITY INDEX
# ============================================================

def create_virality_index(df):
    print("\n" + "=" * 60)
    print("STEP 4G: COMBINED VIRALITY INDEX")
    print("=" * 60)

    print("Calculating Combined Virality Index...")

    # Normalize channel_influence_score to 0-1
    ci_min = df["channel_influence_score"].min()
    ci_max = df["channel_influence_score"].max()
    if ci_max - ci_min > 0:
        df["channel_influence_norm"] = (
            df["channel_influence_score"] - ci_min
        ) / (ci_max - ci_min)
    else:
        df["channel_influence_norm"] = 0

    # Normalize lang_avg_viral_score
    lv_min = df["lang_avg_viral_score"].min()
    lv_max = df["lang_avg_viral_score"].max()
    if lv_max - lv_min > 0:
        df["lang_viral_norm"] = (
            df["lang_avg_viral_score"] - lv_min
        ) / (lv_max - lv_min)
    else:
        df["lang_viral_norm"] = 0

    # Calculate virality index with realistic noise (variance)
    np.random.seed(42)
    noise = np.random.normal(0, 0.08, len(df))

    df["virality_index"] = (
        (df["viral_score"]           * 0.40) +
        (df["engagement_momentum"]   * 0.30) +
        (df["channel_influence_norm"]* 0.20) +
        (df["lang_viral_norm"]       * 0.10) +
        noise
    ).round(6)

    # Cap virality index between 0 and 1
    df["virality_index"] = df["virality_index"].clip(0, 1)

    # Drop temp columns
    df.drop(columns=["channel_influence_norm",
                      "lang_viral_norm"], inplace=True)

    # Virality prediction label
    def predict_virality(score):
        if   score >= 0.70: return "Will Go MEGA VIRAL"
        elif score >= 0.50: return "Will Go VIRAL"
        elif score >= 0.30: return "Likely Trending"
        elif score >= 0.15: return "Possibly Rising"
        else:               return "Unlikely to Viral"

    df["virality_prediction"] = df["virality_index"]\
        .apply(predict_virality)

    print("✅ Virality Index Created!")
    print(f"\n   Min : {df['virality_index'].min():.6f}")
    print(f"   Max : {df['virality_index'].max():.6f}")
    print(f"   Mean: {df['virality_index'].mean():.6f}")

    print("\n📊 Virality Prediction Distribution:")
    print(df["virality_prediction"].value_counts().to_string())

    print("\n🔥 TOP 15 SONGS PREDICTED TO GO VIRAL:")
    top15 = df.nlargest(15, "virality_index")[
        ["title", "channel_name", "language_name",
         "viral_score", "engagement_momentum",
         "virality_index", "virality_prediction"]
    ]
    print(top15.to_string())

    return df

# ============================================================
# STEP 4H: VISUALIZATIONS
# ============================================================

def plot_feature_analysis(df):
    print("\n" + "=" * 60)
    print("STEP 4H: FEATURE ENGINEERING VISUALIZATIONS")
    print("=" * 60)

    # ---- Plot 1: Viral Score Distribution ----
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    ax1.hist(df["viral_score"], bins=50,
             color="#e74c3c", alpha=0.7, edgecolor="white")
    ax1.axvline(df["viral_score"].mean(), color="black",
                linestyle="--", linewidth=2,
                label=f"Mean: {df['viral_score'].mean():.4f}")
    ax1.set_title("Viral Score Distribution",
                  fontsize=12, fontweight="bold")
    ax1.set_xlabel("Viral Score", fontsize=10)
    ax1.set_ylabel("Count", fontsize=10)
    ax1.legend()
    
    plt.tight_layout()
    path1 = f"{OUTPUT_DIR}/4a_viral_score_dist.png"
    plt.savefig(path1, dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print(f"   ✅ Saved: {path1}")

    # ---- Plot 2: Virality Prediction Distribution ----
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    pred_counts = df["virality_prediction"].value_counts()
    pred_colors = ["#8e44ad", "#e74c3c", "#f39c12",
                   "#3498db", "#95a5a6"]
    pred_colors = pred_colors[:len(pred_counts)]
    wedges, texts, autotexts = ax2.pie(
        pred_counts.values,
        labels=pred_counts.index,
        autopct="%1.1f%%",
        colors=pred_colors,
        startangle=140,
        pctdistance=0.8
    )
    for t in autotexts:
        t.set_fontsize(9)
    ax2.set_title("Virality Prediction Distribution",
                   fontsize=13, fontweight="bold")

    plt.tight_layout()
    path2 = f"{OUTPUT_DIR}/4b_virality_prediction.png"
    plt.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"   ✅ Saved: {path2}")

# ============================================================
# STEP 4I: FINAL FEATURE SUMMARY
# ============================================================

def final_feature_summary(df):
    print("\n" + "=" * 60)
    print("STEP 4I: FINAL FEATURE SUMMARY")
    print("=" * 60)

    print("\n📋 ALL ENGINEERED FEATURES:")
    new_features = [
        ("viral_score",              "Core virality score (views*0.5 + likes*0.3 + comments*0.2)"),
        ("viral_score_tier",         "LEGENDARY/MEGA VIRAL/VIRAL/TRENDING/RISING/NORMAL"),
        ("engagement_momentum",      "Speed of engagement growth"),
        ("release_season",           "Winter/Spring/Summer/Autumn"),
        ("is_weekend_release",       "1=Weekend release, 0=Weekday"),
        ("video_age_category",       "New/Recent/Established/Mature/Old"),
        ("growth_rate_score",        "Views growth relative to age"),
        ("lang_avg_viral_score",     "Avg viral score for this language"),
        ("lang_song_count",          "Total songs from this language"),
        ("lang_avg_views",           "Avg views for this language"),
        ("lang_viral_count",         "Viral song count for language"),
        ("lang_viral_rank",          "Language rank by virality"),
        ("channel_avg_views",        "Channel average views"),
        ("channel_total_songs",      "Total songs by channel"),
        ("channel_viral_count",      "Viral songs by channel"),
        ("channel_avg_viral_score",  "Channel avg viral score"),
        ("channel_influence_score",  "Overall channel influence"),
        ("virality_index",           "MASTER score - predicts future virality"),
        ("virality_prediction",      "Will Go MEGA VIRAL / VIRAL / Trending etc"),
    ]

    for feat, desc in new_features:
        if feat in df.columns:
            print(f"   ✅ {feat:35s} : {desc}")

    print(f"\n   Total columns in dataset : {len(df.columns)}")
    print(f"   Total records            : {len(df)}")

    print("\n🔥 FINAL TOP 20 SONGS BY VIRALITY INDEX:")
    top20 = df.nlargest(20, "virality_index")[
        ["title", "channel_name", "language_name",
         "views", "viral_score", "virality_index",
         "virality_prediction"]
    ]
    print(top20.to_string())

    print("\n🌍 LANGUAGE VIRALITY RANKING:")
    lang_rank = df.groupby("language_name").agg(
        total_songs   = ("viral_score", "count"),
        avg_viral     = ("viral_score", "mean"),
        viral_songs   = ("is_viral", "sum"),
        avg_virality  = ("virality_index", "mean")
    ).sort_values("avg_virality", ascending=False).head(15)
    print(lang_rank.round(4).to_string())

# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    print("=" * 60)
    print("   🎵 VIRAL MUSIC TREND PREDICTION")
    print("   STEP 4: FEATURE ENGINEERING")
    print("=" * 60)

    if not os.path.exists(INPUT_FILE):
        print(f"\n❌ ERROR: '{INPUT_FILE}' not found!")
        print("   Please run step2_data_cleaning.py first!")
        return

    # Load data
    df = load_data(INPUT_FILE)

    # Run all feature engineering steps
    df = create_viral_score(df)
    df = create_viral_score_tier(df)
    df = create_engagement_momentum(df)
    df = create_time_features(df)
    df = create_language_features(df)
    df = create_channel_features(df)
    df = create_virality_index(df)

    # Visualizations
    plot_feature_analysis(df)

    # Final summary
    final_feature_summary(df)

    # Save featured dataset
    df.reset_index(drop=True, inplace=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print("\n" + "=" * 60)
    print("✅ STEP 4 COMPLETE!")
    print(f"   Input File  : {INPUT_FILE}")
    print(f"   Output File : {OUTPUT_FILE}")
    print(f"   Records     : {len(df)}")
    print(f"   Columns     : {len(df.columns)}")
    print("   ➡️  Next: python step5_statistical_analysis.py")
    print("=" * 60)


if __name__ == "__main__":
    main()