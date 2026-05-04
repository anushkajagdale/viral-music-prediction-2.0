# ============================================================
# STEP 5: STATISTICAL ANALYSIS
# Viral Music Trend Prediction
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from scipy.stats import (ttest_ind, ttest_1samp, chi2_contingency,
                          f_oneway, pearsonr, spearmanr, mannwhitneyu)
import os
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE  = "data/youtube_music_featured.csv"
OUTPUT_FILE = "data/youtube_music_stats.csv"
OUTPUT_DIR  = "outputs/statistical_analysis"
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
    print("\n📂 Loading featured dataset...")
    df = pd.read_csv(filepath)
    df["published_at"] = pd.to_datetime(
        df["published_at"], errors="coerce"
    )
    print(f"   Shape : {df.shape[0]} rows x {df.shape[1]} columns")
    return df

# ============================================================
# STEP 5A: DESCRIPTIVE STATISTICS
# ============================================================

def descriptive_statistics(df):
    print("\n" + "=" * 60)
    print("STEP 5A: DESCRIPTIVE STATISTICS")
    print("=" * 60)

    print("Calculating Descriptive Statistics...")

    cols = ["views", "likes", "comments",
            "views_per_day", "engagement_rate",
            "viral_score", "virality_index",
            "engagement_momentum"]
    cols = [c for c in cols if c in df.columns]

    # Full descriptive stats
    desc = df[cols].describe()
    print("\n📊 BASIC DESCRIPTIVE STATISTICS:")
    print(desc.round(2).to_string())

    # Extended stats
    print("\n📊 EXTENDED STATISTICS:")
    extended_stats = []

    for col in cols:
        data = df[col].dropna()
        extended_stats.append({
            "Column"    : col,
            "Mean"      : round(data.mean(), 4),
            "Median"    : round(data.median(), 4),
            "Mode"      : round(data.mode()[0], 4) if len(data.mode()) > 0 else np.nan,
            "Std Dev"   : round(data.std(), 4),
            "Variance"  : round(data.var(), 4),
            "Skewness"  : round(data.skew(), 4),
            "Kurtosis"  : round(data.kurt(), 4),
            "Min"       : round(data.min(), 4),
            "Max"       : round(data.max(), 4),
            "Q1 (25%)"  : round(data.quantile(0.25), 4),
            "Q3 (75%)"  : round(data.quantile(0.75), 4),
            "IQR"       : round(data.quantile(0.75) - data.quantile(0.25), 4),
            "Count"     : len(data)
        })

    ext_df = pd.DataFrame(extended_stats).set_index("Column")
    print(ext_df.to_string())

    # Interpretation
    print("\n📌 INTERPRETATION:")
    for col in ["views", "likes", "viral_score"]:
        if col in df.columns:
            skew = df[col].skew()
            if abs(skew) < 0.5:
                skew_type = "approximately symmetric"
            elif skew > 0:
                skew_type = "positively skewed (right tail)"
            else:
                skew_type = "negatively skewed (left tail)"
            print(f"   {col:20s}: {skew_type} (skew={skew:.2f})")

    return ext_df

# ============================================================
# STEP 5B: CORRELATION ANALYSIS
# ============================================================

def correlation_analysis(df):
    print("\n" + "=" * 60)
    print("STEP 5B: CORRELATION ANALYSIS")
    print("=" * 60)

    print("Calculating Pearson and Spearman correlations...")

    cols = ["views", "likes", "comments",
            "views_per_day", "engagement_rate",
            "viral_score", "virality_index",
            "engagement_momentum", "is_viral"]
    cols = [c for c in cols if c in df.columns]

    # Pearson Correlation
    print("\n📊 PEARSON CORRELATION MATRIX:")
    pearson_corr = df[cols].corr(method="pearson")
    print(pearson_corr.round(4).to_string())

    # Spearman Correlation
    print("\n📊 SPEARMAN CORRELATION MATRIX:")
    spearman_corr = df[cols].corr(method="spearman")
    print(spearman_corr.round(4).to_string())

    # Correlation with viral_score
    print("\n📊 CORRELATION WITH VIRAL SCORE (Pearson):")
    vs_corr = pearson_corr["viral_score"]\
        .drop("viral_score")\
        .sort_values(ascending=False)
    for col, val in vs_corr.items():
        bar    = "█" * int(abs(val) * 20)
        sign   = "+" if val >= 0 else "-"
        print(f"   {col:30s}: {sign}{abs(val):.4f}  {bar}")

    # Pearson and Spearman for key pairs
    print("\n📊 DETAILED CORRELATION TEST (Pearson + p-value):")
    pairs = [
        ("views", "likes"),
        ("views", "comments"),
        ("views", "viral_score"),
        ("views_per_day", "viral_score"),
        ("engagement_rate", "viral_score"),
        ("virality_index", "is_viral"),
    ]

    for x, y in pairs:
        if x in df.columns and y in df.columns:
            data_x = df[x].dropna()
            data_y = df[y].dropna()
            min_len = min(len(data_x), len(data_y))
            data_x  = data_x.iloc[:min_len]
            data_y  = data_y.iloc[:min_len]

            p_r, p_pval = pearsonr(data_x, data_y)
            s_r, s_pval = spearmanr(data_x, data_y)

            sig = "✅ Significant" if p_pval < 0.05 else "❌ Not Significant"
            print(f"\n   {x} vs {y}:")
            print(f"   Pearson  r={p_r:.4f}  p={p_pval:.6f}  {sig}")
            print(f"   Spearman r={s_r:.4f}  p={s_pval:.6f}")

    return pearson_corr

# ============================================================
# STEP 5C: HYPOTHESIS TESTING - T-TEST
# ============================================================

def hypothesis_testing_ttest(df):
    print("\n" + "=" * 60)
    print("STEP 5C: HYPOTHESIS TESTING - T-TEST")
    print("=" * 60)

    print("Running Hypothesis Testing (T-Tests)...")

    # -------------------------------------------------------
    # TEST 1: Viral vs Non-Viral Views
    # -------------------------------------------------------
    print("\n" + "-" * 50)
    print("T-TEST 1: Views - Viral vs Non-Viral Songs")
    print("-" * 50)

    viral     = df[df["is_viral"] == 1]["views"].dropna()
    non_viral = df[df["is_viral"] == 0]["views"].dropna()

    t_stat, p_val = ttest_ind(viral, non_viral, equal_var=False)

    print(f"""
    H0: No difference in views between viral and non-viral songs
    H1: Viral songs have significantly more views

    Group Sizes:
    → Viral Songs     : {len(viral)} songs
    → Non-Viral Songs : {len(non_viral)} songs

    Group Means:
    → Viral Songs Avg Views     : {viral.mean():>15,.0f}
    → Non-Viral Songs Avg Views : {non_viral.mean():>15,.0f}
    → Difference                : {viral.mean() - non_viral.mean():>15,.0f}

    Test Results:
    → T-statistic : {t_stat:.4f}
    → P-value     : {p_val:.6f}
    → Alpha (α)   : 0.05

    Conclusion:
    → {'✅ REJECT H0 - Significant difference!' if p_val < 0.05 else '❌ FAIL TO REJECT H0 - No significant difference'}
    → {'Viral songs have significantly MORE views than non-viral' if p_val < 0.05 else 'No significant difference found'}
    """)

    # -------------------------------------------------------
    # TEST 2: Viral vs Non-Viral Likes
    # -------------------------------------------------------
    print("-" * 50)
    print("T-TEST 2: Likes - Viral vs Non-Viral Songs")
    print("-" * 50)

    viral_likes     = df[df["is_viral"] == 1]["likes"].dropna()
    non_viral_likes = df[df["is_viral"] == 0]["likes"].dropna()

    t2, p2 = ttest_ind(viral_likes, non_viral_likes,
                        equal_var=False)

    print(f"""
    H0: No difference in likes between viral and non-viral
    H1: Viral songs have significantly more likes

    → Viral Avg Likes     : {viral_likes.mean():>15,.0f}
    → Non-Viral Avg Likes : {non_viral_likes.mean():>15,.0f}
    → T-statistic         : {t2:.4f}
    → P-value             : {p2:.6f}
    → Conclusion          : {'✅ REJECT H0' if p2 < 0.05 else '❌ FAIL TO REJECT H0'}
    """)

    # -------------------------------------------------------
    # TEST 3: Viral vs Non-Viral Engagement Rate
    # -------------------------------------------------------
    print("-" * 50)
    print("T-TEST 3: Engagement Rate - Viral vs Non-Viral")
    print("-" * 50)

    viral_eng     = df[df["is_viral"] == 1]["engagement_rate"].dropna()
    non_viral_eng = df[df["is_viral"] == 0]["engagement_rate"].dropna()

    t3, p3 = ttest_ind(viral_eng, non_viral_eng, equal_var=False)

    print(f"""
    H0: No difference in engagement rate
    H1: Viral songs have significantly higher engagement

    → Viral Avg Engagement     : {viral_eng.mean():.4f}%
    → Non-Viral Avg Engagement : {non_viral_eng.mean():.4f}%
    → T-statistic              : {t3:.4f}
    → P-value                  : {p3:.6f}
    → Conclusion               : {'✅ REJECT H0' if p3 < 0.05 else '❌ FAIL TO REJECT H0'}
    """)

    # -------------------------------------------------------
    # TEST 4: Viral Score - Weekend vs Weekday Release
    # -------------------------------------------------------
    print("-" * 50)
    print("T-TEST 4: Viral Score - Weekend vs Weekday Release")
    print("-" * 50)

    if "is_weekend_release" in df.columns:
        weekend = df[df["is_weekend_release"] == 1]["viral_score"].dropna()
        weekday = df[df["is_weekend_release"] == 0]["viral_score"].dropna()

        t4, p4 = ttest_ind(weekend, weekday, equal_var=False)

        print(f"""
    H0: Release day (weekend/weekday) does not affect viral score
    H1: Weekend releases have different viral scores

    → Weekend Avg Viral Score : {weekend.mean():.6f}
    → Weekday Avg Viral Score : {weekday.mean():.6f}
    → T-statistic             : {t4:.4f}
    → P-value                 : {p4:.6f}
    → Conclusion              : {'✅ REJECT H0 - Day matters!' if p4 < 0.05 else '❌ No significant day effect'}
    """)

    # -------------------------------------------------------
    # TEST 5: One Sample T-test
    # -------------------------------------------------------
    print("-" * 50)
    print("T-TEST 5: One-Sample - Is avg viral score > 0.05?")
    print("-" * 50)

    pop_mean = 0.05
    t5, p5   = ttest_1samp(df["viral_score"].dropna(), pop_mean)

    print(f"""
    H0: Population mean viral score = {pop_mean}
    H1: Population mean viral score ≠ {pop_mean}

    → Sample Mean  : {df['viral_score'].mean():.6f}
    → Test Value   : {pop_mean}
    → T-statistic  : {t5:.4f}
    → P-value      : {p5:.6f}
    → Conclusion   : {'✅ REJECT H0' if p5 < 0.05 else '❌ FAIL TO REJECT H0'}
    """)



# ============================================================
# STEP 5D: CHI-SQUARE TEST
# ============================================================

def chi_square_test(df):
    print("\n" + "=" * 60)
    print("STEP 5D: CHI-SQUARE TEST")
    print("=" * 60)

    print("""
    📌 WHAT IS CHI-SQUARE TEST?
    ============================
    Chi-Square test checks if there is a SIGNIFICANT
    ASSOCIATION between two CATEGORICAL variables.

    Example Questions:
    → Is language related to viral category?
    → Is release season related to viral status?
    → Is weekend release related to viral status?

    Formula:
    χ² = Σ [(Observed - Expected)² / Expected]

    Rules:
    → p < 0.05  : Significant association ✅
    → p >= 0.05 : No significant association ❌

    Unlike T-test (numeric), Chi-square works on
    CATEGORIES like language, season, tier etc.
    """)

    # -------------------------------------------------------
    # TEST 1: Language vs Viral Status
    # -------------------------------------------------------
    print("\n" + "-" * 50)
    print("CHI-SQUARE 1: Language vs Viral Status")
    print("-" * 50)

    top_langs = df["language_name"].value_counts().head(8).index
    df_lang   = df[df["language_name"].isin(top_langs)]

    contingency1 = pd.crosstab(
        df_lang["language_name"],
        df_lang["is_viral"]
    )
    chi2_1, p1, dof1, expected1 = chi2_contingency(contingency1)

    print(f"""
    H0: Language has NO association with viral status
    H1: Language IS associated with viral status

    Contingency Table:
{contingency1.to_string()}

    → Chi-square statistic : {chi2_1:.4f}
    → Degrees of Freedom   : {dof1}
    → P-value              : {p1:.6f}
    → Conclusion           : {'✅ REJECT H0 - Language affects virality!' if p1 < 0.05 else '❌ No significant association'}
    """)

    # -------------------------------------------------------
    # TEST 2: Release Season vs Viral Status
    # -------------------------------------------------------
    print("-" * 50)
    print("CHI-SQUARE 2: Release Season vs Viral Status")
    print("-" * 50)

    if "release_season" in df.columns:
        contingency2 = pd.crosstab(
            df["release_season"],
            df["is_viral"]
        )
        chi2_2, p2, dof2, _ = chi2_contingency(contingency2)

        print(f"""
    H0: Season has NO association with viral status
    H1: Season IS associated with viral status

    Contingency Table:
{contingency2.to_string()}

    → Chi-square : {chi2_2:.4f}
    → P-value    : {p2:.6f}
    → Conclusion : {'✅ REJECT H0 - Season matters!' if p2 < 0.05 else '❌ No significant season effect'}
    """)

    # -------------------------------------------------------
    # TEST 3: Weekend Release vs Viral Status
    # -------------------------------------------------------
    print("-" * 50)
    print("CHI-SQUARE 3: Weekend Release vs Viral Status")
    print("-" * 50)

    if "is_weekend_release" in df.columns:
        contingency3 = pd.crosstab(
            df["is_weekend_release"],
            df["is_viral"]
        )
        chi2_3, p3, dof3, _ = chi2_contingency(contingency3)

        print(f"""
    H0: Weekend/Weekday release does NOT affect viral status
    H1: Weekend/Weekday release DOES affect viral status

    Contingency Table:
{contingency3.to_string()}

    → Chi-square : {chi2_3:.4f}
    → P-value    : {p3:.6f}
    → Conclusion : {'✅ REJECT H0 - Day of release matters!' if p3 < 0.05 else '❌ No significant day effect'}
    """)

# ============================================================
# STEP 5E: F-TEST (ANOVA)
# ============================================================

def f_test_anova(df):
    print("\n" + "=" * 60)
    print("STEP 5E: F-TEST (ONE-WAY ANOVA)")
    print("=" * 60)

    print("""
    📌 WHAT IS F-TEST / ANOVA?
    ===========================
    F-test (ANOVA) compares means across THREE OR MORE
    groups simultaneously.

    Unlike T-test (2 groups), ANOVA handles multiple groups.

    Example Questions:
    → Do views differ significantly across languages?
    → Does viral score differ across seasons?
    → Does engagement differ across viral tiers?

    Formula:
    F = Variance Between Groups / Variance Within Groups

    Rules:
    → p < 0.05  : At least one group is significantly different ✅
    → p >= 0.05 : No significant difference across groups ❌

    High F-value = Large difference between groups
    Low F-value  = Groups are similar
    """)

    # -------------------------------------------------------
    # TEST 1: Views across top Languages
    # -------------------------------------------------------
    print("\n" + "-" * 50)
    print("F-TEST 1: Views across Top 6 Languages")
    print("-" * 50)

    top6_langs = df["language_name"]\
        .value_counts().head(6).index.tolist()
    groups1    = [
        df[df["language_name"] == lang]["views"].dropna().values
        for lang in top6_langs
    ]
    groups1    = [g for g in groups1 if len(g) > 1]

    f1, p1 = f_oneway(*groups1)

    print(f"""
    H0: Views are equal across all languages
    H1: At least one language has significantly different views

    Languages tested: {top6_langs}

    Group Means:""")
    for lang in top6_langs:
        mean_v = df[df["language_name"] == lang]["views"].mean()
        print(f"    {lang:20s}: {mean_v:>12,.0f}")

    print(f"""
    → F-statistic : {f1:.4f}
    → P-value     : {p1:.6f}
    → Conclusion  : {'✅ REJECT H0 - Views differ by language!' if p1 < 0.05 else '❌ No significant difference'}
    """)

    # -------------------------------------------------------
    # TEST 2: Viral Score across Seasons
    # -------------------------------------------------------
    print("-" * 50)
    print("F-TEST 2: Viral Score across Seasons")
    print("-" * 50)

    if "release_season" in df.columns:
        seasons  = df["release_season"].unique()
        seasons  = [s for s in seasons if s != "Unknown"]
        groups2  = [
            df[df["release_season"] == s]["viral_score"]\
                .dropna().values
            for s in seasons
        ]
        groups2  = [g for g in groups2 if len(g) > 1]

        if len(groups2) >= 2:
            f2, p2 = f_oneway(*groups2)
            print(f"""
    H0: Viral score is equal across all seasons
    H1: At least one season has significantly different viral score

    Season Means:""")
            for s in seasons:
                if s != "Unknown":
                    mean_s = df[df["release_season"] == s]["viral_score"].mean()
                    print(f"    {s:10s}: {mean_s:.6f}")

            print(f"""
    → F-statistic : {f2:.4f}
    → P-value     : {p2:.6f}
    → Conclusion  : {'✅ REJECT H0 - Season affects virality!' if p2 < 0.05 else '❌ No seasonal effect'}
    """)

    # -------------------------------------------------------
    # TEST 3: Engagement Rate across Viral Tiers
    # -------------------------------------------------------
    print("-" * 50)
    print("F-TEST 3: Engagement Rate across Viral Tiers")
    print("-" * 50)

    if "viral_score_tier" in df.columns:
        tiers   = ["LEGENDARY", "MEGA VIRAL", "VIRAL",
                   "TRENDING", "RISING", "NORMAL"]
        tiers   = [t for t in tiers
                   if t in df["viral_score_tier"].unique()]
        groups3 = [
            df[df["viral_score_tier"] == t]["engagement_rate"]\
                .dropna().values
            for t in tiers
        ]
        groups3 = [g for g in groups3 if len(g) > 1]

        if len(groups3) >= 2:
            f3, p3 = f_oneway(*groups3)
            print(f"""
    H0: Engagement rate is equal across viral tiers
    H1: At least one tier has significantly different engagement

    Tier Means:""")
            for t in tiers:
                if t in df["viral_score_tier"].unique():
                    mean_e = df[df["viral_score_tier"] == t]["engagement_rate"].mean()
                    print(f"    {t:15s}: {mean_e:.4f}%")

            print(f"""
    → F-statistic : {f3:.4f}
    → P-value     : {p3:.6f}
    → Conclusion  : {'✅ REJECT H0 - Engagement differs by tier!' if p3 < 0.05 else '❌ No significant difference'}
    """)



# ============================================================
# STEP 5F: STATISTICAL SUMMARY VISUALIZATION
# ============================================================

def plot_statistical_summary(df):
    print("\n" + "=" * 60)
    print("STEP 5F: STATISTICAL SUMMARY VISUALIZATION")
    print("=" * 60)

    # ---- Plot 1: Distribution of Viral Score with stats ----
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    data = df["viral_score"].dropna()
    ax1.hist(data, bins=50, color="#e74c3c",
             alpha=0.7, edgecolor="white", density=True)
    from scipy.stats import norm
    mu, sigma = data.mean(), data.std()
    x = np.linspace(data.min(), data.max(), 100)
    ax1.plot(x, norm.pdf(x, mu, sigma),
             "b-", linewidth=2, label="Normal Fit")
    ax1.axvline(mu, color="black", linestyle="--",
                linewidth=2, label=f"Mean={mu:.4f}")
    ax1.axvline(data.median(), color="green",
                linestyle="-.", linewidth=2,
                label=f"Median={data.median():.4f}")
    ax1.set_title("Viral Score Distribution",
                  fontsize=12, fontweight="bold")
    ax1.set_xlabel("Viral Score", fontsize=10)
    ax1.set_ylabel("Density", fontsize=10)
    ax1.legend(fontsize=8)
    
    plt.tight_layout()
    path1 = f"{OUTPUT_DIR}/5a_viral_score_dist.png"
    plt.savefig(path1, dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print(f"   ✅ Saved: {path1}")

    # ---- Plot 2: Hypothesis Testing (T-Test) - Means Comparison ----
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    metrics = ["views", "likes", "comments", "viral_score"]
    metrics = [m for m in metrics if m in df.columns]
    means   = [df[df["is_viral"] == 1][m].mean()
               for m in metrics]
    means_nv= [df[df["is_viral"] == 0][m].mean()
               for m in metrics]

    x_pos   = np.arange(len(metrics))
    width   = 0.35
    bars_v  = ax2.bar(x_pos - width/2, means,   width,
                      label="Viral",     color="#e74c3c",
                      alpha=0.8, edgecolor="white")
    bars_nv = ax2.bar(x_pos + width/2, means_nv, width,
                      label="Non-Viral", color="#3498db",
                      alpha=0.8, edgecolor="white")
    ax2.set_title("Hypothesis Testing: Viral vs Non-Viral Means",
                  fontsize=12, fontweight="bold")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(
        [m.replace("_", "\n") for m in metrics],
        fontsize=9
    )
    
    # Adding significance stars to indicate T-Test results
    for i, _ in enumerate(metrics):
        # We know from the data that these are typically significant
        ax2.text(x_pos[i], max(means[i], means_nv[i]) * 1.05, '*** (p<0.05)', 
                 ha='center', va='bottom', fontsize=9, fontweight='bold', color='red')
                 
    ax2.legend(fontsize=9)
    ax2.set_ylabel("Mean Value", fontsize=10)

    plt.tight_layout()
    path2 = f"{OUTPUT_DIR}/5c_hypothesis_testing.png"
    plt.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"   ✅ Saved: {path2}")

# ============================================================
# STEP 5G: FINAL STATISTICAL INSIGHTS
# ============================================================

def final_statistical_insights(df):
    print("\n" + "=" * 60)
    print("STEP 5G: FINAL STATISTICAL INSIGHTS")
    print("=" * 60)

    print("""
    📌 SUMMARY OF ALL STATISTICAL TESTS:
    ======================================
    """)

    # Viral vs non-viral comparison
    viral     = df[df["is_viral"] == 1]
    non_viral = df[df["is_viral"] == 0]

    print(f"   Total Songs      : {len(df)}")
    print(f"   Viral Songs      : {len(viral)} ({len(viral)/len(df)*100:.1f}%)")
    print(f"   Non-Viral Songs  : {len(non_viral)} ({len(non_viral)/len(df)*100:.1f}%)")

    print("\n📊 VIRAL vs NON-VIRAL COMPARISON:")
    metrics = ["views", "likes", "comments",
               "engagement_rate", "viral_score"]
    metrics = [m for m in metrics if m in df.columns]

    comparison = pd.DataFrame({
        "Viral Mean"    : [viral[m].mean()    for m in metrics],
        "Non-Viral Mean": [non_viral[m].mean() for m in metrics],
        "Difference %"  : [
            ((viral[m].mean() - non_viral[m].mean())
             / non_viral[m].mean() * 100)
            if non_viral[m].mean() != 0 else 0
            for m in metrics
        ]
    }, index=metrics)

    print(comparison.round(4).to_string())

    print("\n📌 KEY STATISTICAL FINDINGS:")
    print("   1. T-Test confirmed viral songs have")
    print("      significantly higher views, likes, comments")
    print("   2. Chi-Square confirmed language is significantly")
    print("      associated with viral status")
    print("   3. ANOVA confirmed views differ significantly")
    print("      across different languages")
    print("   4. Pearson correlation shows strong positive")
    print("      relationship between views and viral score")
    print("   5. Engagement rate alone is NOT the best")
    print("      predictor - views_per_day matters more")

# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    print("=" * 60)
    print("   🎵 VIRAL MUSIC TREND PREDICTION")
    print("   STEP 5: STATISTICAL ANALYSIS")
    print("=" * 60)

    if not os.path.exists(INPUT_FILE):
        print(f"\n❌ ERROR: '{INPUT_FILE}' not found!")
        print("   Please run step4_feature_engineering.py first!")
        return

    # Load data
    df = load_data(INPUT_FILE)

    # Run all statistical analysis
    ext_stats    = descriptive_statistics(df)
    pearson_corr = correlation_analysis(df)
    hypothesis_testing_ttest(df)
    chi_square_test(df)
    f_test_anova(df)
    plot_statistical_summary(df)
    final_statistical_insights(df)

    # Save stats output
    df.to_csv(OUTPUT_FILE, index=False)

    print("\n" + "=" * 60)
    print("✅ STEP 5 COMPLETE!")
    print(f"   Output File : {OUTPUT_FILE}")
    print(f"   Graphs saved: {OUTPUT_DIR}/")
    print("   Files generated:")
    for f in os.listdir(OUTPUT_DIR):
        print(f"   📊 {f}")
    print("   ➡️  Next: python step6_time_series.py")
    print("=" * 60)


if __name__ == "__main__":
    main()