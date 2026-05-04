# =========================
# 🔥 IMPORTANT: SET ENV BEFORE IMPORTS
# =========================
import os
os.environ["DASK_DISTRIBUTED__SCHEDULER__PORT"] = "0"


# =========================
# IMPORTS
# =========================
import pandas as pd
import numpy as np
from tpot import TPOTRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import time


def main():

    print("🚀 Starting TPOT AutoML for Viral Music Prediction...")

    # =========================
    # 1. LOAD DATA
    # =========================
    data_path = 'data/youtube_music_stats.csv'

    if not os.path.exists(data_path):
        print(f"❌ Error: Could not find {data_path}")
        return

    print("📂 Loading dataset...")
    df = pd.read_csv(data_path)

    # =========================
    # 2. FEATURES & TARGET
    # =========================
    target_col = 'virality_index'

    features = [
        'views',
        'likes',
        'comments',
        'views_per_day',
        'engagement_rate',
        'like_view_ratio',
        'comment_view_ratio'
    ]

    df_clean = df.dropna(subset=features + [target_col])

    X = df_clean[features]
    y = df_clean[target_col]

    print(f"✅ Dataset Shape: {X.shape}")
    print(f"📊 Features Used: {', '.join(features)}")

    # =========================
    # 3. TRAIN-TEST SPLIT
    # =========================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("🔀 Data split complete (80% train / 20% test)")

    # =========================
    # 4. TPOT AUTOML
    # =========================
    print("\n🤖 Initializing TPOT AutoML...")

    start_time = time.time()

    tpot = TPOTRegressor(
        generations=2,
        population_size=10,
        verbose=2,
        random_state=42,
        n_jobs=1,
        max_time_mins=None
    )

    tpot.fit(X_train, y_train)

    end_time = time.time()

    print(f"\n⏱️ TPOT Optimization Completed in {end_time - start_time:.2f} seconds")

    # =========================
    # 5. MODEL EVALUATION
    # =========================
    print("\n📈 Evaluating model...")

    y_pred = tpot.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"✅ R2 Score: {r2:.4f}")
    print(f"✅ RMSE: {rmse:.4f}")

    # =========================
    # 6. SAVE MODEL
    # =========================
    print("\n💾 Saving model...")

    os.makedirs('outputs', exist_ok=True)

    model_path = 'outputs/tpot_viral_model.pkl'
    joblib.dump(tpot.fitted_pipeline_, model_path)

    print(f"✅ Model saved at: {model_path}")

    # =========================
    # 7. SHOW BEST MODEL
    # =========================
    print("\n🏆 BEST MODEL FOUND:")
    print(tpot.fitted_pipeline_)

    print("\n🎉 ALL DONE! AutoML model is ready for prediction.")


# =========================
# RUN
# =========================
if __name__ == "__main__":
    main()