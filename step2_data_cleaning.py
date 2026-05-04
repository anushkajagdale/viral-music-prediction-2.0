# ============================================================
# STEP 2: DATA CLEANING & PREPROCESSING
# Viral Music Trend Prediction
# ============================================================

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import os

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE  = "data/youtube_music_data.csv"
OUTPUT_FILE = "data/youtube_music_cleaned.csv"

# ============================================================
# LOAD DATA
# ============================================================

def load_data(filepath):
    print("\n📂 Loading dataset...")
    df = pd.read_csv(filepath)
    print(f"   Shape   : {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"   Columns : {list(df.columns)}")
    return df

# ============================================================
# STEP 2A: INITIAL DATA INSPECTION
# ============================================================

def initial_inspection(df):
    print("\n" + "=" * 60)
    print("STEP 2A: INITIAL DATA INSPECTION")
    print("=" * 60)

    print("\n📋 First 5 Rows:")
    print(df.head().to_string())

    print("\n📊 Data Types:")
    print(df.dtypes.to_string())

    print("\n📈 Basic Statistics:")
    print(df[["views", "likes", "comments", "views_per_day"]].describe().to_string())

    print("\n🔍 Missing Values (Before Cleaning):")
    missing = df.isnull().sum()
    missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
    missing_df = pd.DataFrame({
        "Missing Count"  : missing,
        "Missing Percent": missing_pct
    })
    print(missing_df[missing_df["Missing Count"] > 0].to_string())

    print("\n🔢 Duplicate Rows:", df.duplicated().sum())

    return df

# ============================================================
# STEP 2B: HANDLE MISSING VALUES
# ============================================================

def handle_missing_values(df):
    print("\n" + "=" * 60)
    print("STEP 2B: HANDLING MISSING VALUES")
    print("=" * 60)

    print("\n📊 BEFORE - Missing Values:")
    print(df.isnull().sum().to_string())

    # --- views: drop rows where views = 0 or missing ---
    before = len(df)
    df = df[df["views"] > 0]
    df = df.dropna(subset=["views"])
    print(f"\n✅ Removed {before - len(df)} rows with 0 or missing views")
    print(f"   Justification: A video with 0 views has no analytical value")

    # --- likes: fill missing with MEDIAN ---
    # Median is better than mean because likes can be skewed
    likes_median = df["likes"].median()
    missing_likes = df["likes"].isnull().sum()
    df["likes"].fillna(likes_median, inplace=True)
    print(f"\n✅ Filled {missing_likes} missing 'likes' with MEDIAN: {likes_median:,.0f}")
    print(f"   Justification: Median is robust to outliers in skewed data")

    # --- comments: fill missing with MEAN ---
    comments_mean = df["comments"].mean()
    missing_comments = df["comments"].isnull().sum()
    df["comments"].fillna(comments_mean, inplace=True)
    print(f"\n✅ Filled {missing_comments} missing 'comments' with MEAN: {comments_mean:,.0f}")
    print(f"   Justification: Comments follow near-normal distribution")

    # --- views_per_day: fill missing with 0 ---
    df["views_per_day"].fillna(0, inplace=True)
    print(f"\n✅ Filled missing 'views_per_day' with 0")

    # --- language_name: fill missing with Unknown ---
    df["language_name"].fillna("Unknown", inplace=True)
    df["language_code"].fillna("unknown", inplace=True)
    print(f"\n✅ Filled missing language fields with 'Unknown'")

    # --- published_at: drop rows with missing date ---
    before = len(df)
    df = df.dropna(subset=["published_at"])
    print(f"\n✅ Removed {before - len(df)} rows with missing published_at")

    print("\n📊 AFTER - Missing Values:")
    print(df.isnull().sum().to_string())

    print(f"\n📦 Shape after missing value treatment: {df.shape}")

    return df

# ============================================================
# STEP 2C: CONVERT & FIX DATA TYPES
# ============================================================

def fix_data_types(df):
    print("\n" + "=" * 60)
    print("STEP 2C: DATA TYPE CONVERSION")
    print("=" * 60)

    print("\n📊 BEFORE - Data Types:")
    print(df.dtypes.to_string())

    # Convert published_at to datetime
    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")

    # Extract date features
    df["publish_year"]    = df["published_at"].dt.year.astype("Int64")
    df["publish_month"]   = df["published_at"].dt.month.astype("Int64")
    df["publish_day"]     = df["published_at"].dt.day.astype("Int64")
    df["publish_dow"]     = df["published_at"].dt.dayofweek.astype("Int64")
    df["publish_quarter"] = df["published_at"].dt.quarter.astype("Int64")
    df["publish_week"]    = df["published_at"].dt.isocalendar().week.astype("Int64")

    # Day of week name
    days = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    df["publish_day_name"] = df["publish_dow"].apply(
        lambda x: days[x] if pd.notna(x) and 0 <= x <= 6 else "Unknown"
    )

    # Month name
    months = ["Jan","Feb","Mar","Apr","May","Jun",
              "Jul","Aug","Sep","Oct","Nov","Dec"]
    df["publish_month_name"] = df["publish_month"].apply(
        lambda x: months[x-1] if pd.notna(x) and 1 <= x <= 12 else "Unknown"
    )

    # Convert numeric columns
    df["views"]         = pd.to_numeric(df["views"],         errors="coerce").fillna(0).astype(int)
    df["likes"]         = pd.to_numeric(df["likes"],         errors="coerce").fillna(0).astype(int)
    df["comments"]      = pd.to_numeric(df["comments"],      errors="coerce").fillna(0).astype(int)
    df["views_per_day"] = pd.to_numeric(df["views_per_day"], errors="coerce").fillna(0)
    df["is_viral"]      = pd.to_numeric(df["is_viral"],      errors="coerce").fillna(0).astype(int)

    # Recalculate days_since_publish
    df["days_since_publish"] = (pd.Timestamp.now() - df["published_at"]).dt.days

    print("\n📊 AFTER - Data Types:")
    print(df.dtypes.to_string())

    print("\n✅ New Date Columns Created:")
    print(df[["published_at", "publish_year", "publish_month_name",
              "publish_day_name", "publish_quarter"]].head(5).to_string())

    return df

# ============================================================
# STEP 2D: REMOVE DUPLICATES
# ============================================================

def remove_duplicates(df):
    print("\n" + "=" * 60)
    print("STEP 2D: REMOVING DUPLICATES")
    print("=" * 60)

    before = len(df)
    df.drop_duplicates(subset=["video_id"], keep="first", inplace=True)
    df.drop_duplicates(subset=["title", "channel_name"], keep="first", inplace=True)
    after = len(df)

    print(f"\n   Rows BEFORE : {before}")
    print(f"   Rows AFTER  : {after}")
    print(f"   Removed     : {before - after} duplicate rows")
    print(f"   Justification: Same video appearing from multiple regions removed")

    return df

# ============================================================
# STEP 2E: OUTLIER DETECTION - IQR METHOD
# ============================================================

def remove_outliers_iqr(df):
    print("\n" + "=" * 60)
    print("STEP 2E: OUTLIER REMOVAL - IQR METHOD")
    print("=" * 60)

    cols   = ["views", "likes", "comments", "views_per_day"]
    before = len(df)

    print(f"\n   Columns Checked : {cols}")
    print(f"   Formula         : Lower = Q1 - 1.5*IQR | Upper = Q3 + 1.5*IQR")

    for col in cols:
        Q1  = df[col].quantile(0.25)
        Q3  = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower  = Q1 - 1.5 * IQR
        upper  = Q3 + 1.5 * IQR

        outliers_count = df[(df[col] < lower) | (df[col] > upper)].shape[0]
        df = df[(df[col] >= lower) & (df[col] <= upper)]

        print(f"\n   Column  : {col}")
        print(f"   Q1      : {Q1:>12,.2f}")
        print(f"   Q3      : {Q3:>12,.2f}")
        print(f"   IQR     : {IQR:>12,.2f}")
        print(f"   Lower   : {lower:>12,.2f}")
        print(f"   Upper   : {upper:>12,.2f}")
        print(f"   Removed : {outliers_count} outliers")

    after = len(df)
    print(f"\n   Rows BEFORE : {before}")
    print(f"   Rows AFTER  : {after}")
    print(f"   Justification: IQR method removes extreme viral/non-viral")
    print(f"   outliers that would skew our prediction model")

    return df

# ============================================================
# STEP 2F: FEATURE SCALING
# ============================================================

def feature_scaling(df):
    print("\n" + "=" * 60)
    print("STEP 2F: FEATURE SCALING")
    print("=" * 60)

    cols = ["views", "likes", "comments", "views_per_day"]

    # --- Normalization: MinMax (0 to 1) ---
    print("\n📊 NORMALIZATION (MinMax Scaler: 0 to 1)")
    print("   Used for: Neural networks, KNN, visualization")
    scaler_mm  = MinMaxScaler()
    normalized = scaler_mm.fit_transform(df[cols])

    df["views_normalized"]         = normalized[:, 0]
    df["likes_normalized"]         = normalized[:, 1]
    df["comments_normalized"]      = normalized[:, 2]
    df["views_per_day_normalized"] = normalized[:, 3]

    print("\n   BEFORE vs AFTER Normalization:")
    comparison = pd.DataFrame({
        "views_original"   : df["views"].head(5).values,
        "views_normalized" : df["views_normalized"].head(5).round(4).values
    })
    print(comparison.to_string())

    # --- Standardization: Z-score (mean=0, std=1) ---
    print("\n📊 STANDARDIZATION (Standard Scaler: mean=0, std=1)")
    print("   Used for: Linear regression, SVM, ARIMA")
    scaler_std    = StandardScaler()
    standardized  = scaler_std.fit_transform(df[cols])

    df["views_standardized"]         = standardized[:, 0]
    df["likes_standardized"]         = standardized[:, 1]
    df["comments_standardized"]      = standardized[:, 2]
    df["views_per_day_standardized"] = standardized[:, 3]

    print("\n   BEFORE vs AFTER Standardization:")
    comparison2 = pd.DataFrame({
        "views_original"      : df["views"].head(5).values,
        "views_standardized"  : df["views_standardized"].head(5).round(4).values
    })
    print(comparison2.to_string())

    print("\n✅ Scaling Summary:")
    print(f"   Normalization Range  : {df['views_normalized'].min():.2f} to {df['views_normalized'].max():.2f}")
    print(f"   Standardization Mean : {df['views_standardized'].mean():.4f}")
    print(f"   Standardization Std  : {df['views_standardized'].std():.4f}")

    return df

# ============================================================
# STEP 2G: ADD ENGAGEMENT METRICS
# ============================================================

def add_engagement_metrics(df):
    print("\n" + "=" * 60)
    print("STEP 2G: ADDING ENGAGEMENT METRICS")
    print("=" * 60)

    # Engagement Rate = (likes + comments) / views * 100
    df["engagement_rate"] = (
        (df["likes"] + df["comments"]) / df["views"] * 100
    ).round(4)

    # Like to View Ratio
    df["like_view_ratio"] = (
        df["likes"] / df["views"] * 100
    ).round(4)

    # Comment to View Ratio
    df["comment_view_ratio"] = (
        df["comments"] / df["views"] * 100
    ).round(4)

    # Viral Category based on views_per_day
    def categorize_viral(vpd):
        if vpd >= 1_000_000:  return "Mega Viral"
        elif vpd >= 500_000:  return "Super Viral"
        elif vpd >= 100_000:  return "Viral"
        elif vpd >= 50_000:   return "Trending"
        elif vpd >= 10_000:   return "Growing"
        else:                 return "Normal"

    df["viral_category"] = df["views_per_day"].apply(categorize_viral)

    print("\n✅ New Columns Added:")
    print("   engagement_rate    = (likes + comments) / views * 100")
    print("   like_view_ratio    = likes / views * 100")
    print("   comment_view_ratio = comments / views * 100")
    print("   viral_category     = Mega Viral / Super Viral / Viral / Trending / Growing / Normal")

    print("\n📊 Viral Category Distribution:")
    print(df["viral_category"].value_counts().to_string())

    print("\n📊 Sample Engagement Metrics:")
    print(df[["title", "views", "likes", "engagement_rate",
              "viral_category"]].head(10).to_string())

    return df

# ============================================================
# STEP 2H: FINAL CLEAN DATASET SUMMARY
# ============================================================

def final_summary(df):
    print("\n" + "=" * 60)
    print("STEP 2H: FINAL CLEANED DATASET SUMMARY")
    print("=" * 60)

    print(f"\n   Total Records      : {len(df)}")
    print(f"   Total Columns      : {len(df.columns)}")
    print(f"   Missing Values     : {df.isnull().sum().sum()}")
    print(f"   Duplicate Rows     : {df.duplicated().sum()}")
    print(f"   Languages Found    : {df['language_name'].nunique()}")
    print(f"   Viral Songs        : {df['is_viral'].sum()}")
    print(f"   Date Range         : {df['published_at'].min()} to {df['published_at'].max()}")

    print("\n📊 Viral Category Breakdown:")
    print(df["viral_category"].value_counts().to_string())

    print("\n🌍 Top 10 Languages in Dataset:")
    print(df["language_name"].value_counts().head(10).to_string())

    print("\n🔥 Top 10 Most Viral Songs Right Now:")
    top_viral = df[["title", "channel_name", "language_name",
                    "views", "views_per_day", "viral_category"]]\
        .sort_values("views_per_day", ascending=False).head(10)
    print(top_viral.to_string())

    print("\n📋 All Columns in Cleaned Dataset:")
    for i, col in enumerate(df.columns, 1):
        print(f"   {i:02d}. {col}")

# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    print("=" * 60)
    print("   🎵 VIRAL MUSIC TREND PREDICTION")
    print("   STEP 2: DATA CLEANING & PREPROCESSING")
    print("=" * 60)

    # Check input file exists
    if not os.path.exists(INPUT_FILE):
        print(f"\n❌ ERROR: '{INPUT_FILE}' not found!")
        print("   Please run step1_data_collection.py first!")
        return

    # Load data
    df = load_data(INPUT_FILE)

    # Run all cleaning steps
    df = initial_inspection(df)
    df = handle_missing_values(df)
    df = fix_data_types(df)
    df = remove_duplicates(df)
    df = remove_outliers_iqr(df)
    df = feature_scaling(df)
    df = add_engagement_metrics(df)

    # Final summary
    final_summary(df)

    # Save cleaned dataset
    df.reset_index(drop=True, inplace=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print("\n" + "=" * 60)
    print("✅ STEP 2 COMPLETE!")
    print(f"   Input File  : {INPUT_FILE}")
    print(f"   Output File : {OUTPUT_FILE}")
    print(f"   Records     : {len(df)}")
    print(f"   Columns     : {len(df.columns)}")
    print("   ➡️  Next: python step3_eda.py")
    print("=" * 60)


if __name__ == "__main__":
    main()