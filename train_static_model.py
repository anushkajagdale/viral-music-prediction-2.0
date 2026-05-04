import os
import pandas as pd
import numpy as np
import joblib
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def train_and_evaluate():
    data_path = 'data/youtube_music_stats.csv'
    tpot_model_path = 'outputs/tpot_viral_model.pkl'
    static_model_path = 'outputs/static_model.pkl'
    metrics_path = 'outputs/model_metrics.json'

    if not os.path.exists(data_path):
        print(f"Data not found at {data_path}. Cannot train model.")
        return

    print("Loading data...")
    df = pd.read_csv(data_path)
    target_col = 'virality_index'
    features = [
        'views', 'likes', 'comments', 'views_per_day', 'engagement_rate',
        'like_view_ratio', 'comment_view_ratio'
    ]
    df_clean = df.dropna(subset=features + [target_col])
    X = df_clean[features]
    y = df_clean[target_col]

    print("Training static model (Random Forest)...")
    static_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    static_model.fit(X, y)

    # Save static model
    joblib.dump(static_model, static_model_path)
    print(f"Static model saved to {static_model_path}")

    # Evaluate Static Model
    print("Evaluating Static Model...")
    y_pred_static = static_model.predict(X)
    y_pred_static_capped = np.clip(y_pred_static, 0, 1)
    static_rmse = np.sqrt(mean_squared_error(y, y_pred_static_capped))
    static_r2 = r2_score(y, y_pred_static_capped)

    # Evaluate TPOT Model (if it exists)
    tpot_rmse = 0.0
    tpot_r2 = 0.0
    if os.path.exists(tpot_model_path):
        print("Evaluating TPOT Model...")
        tpot_model = joblib.load(tpot_model_path)
        y_pred_tpot = tpot_model.predict(X)
        y_pred_tpot_capped = np.clip(y_pred_tpot, 0, 1)
        tpot_rmse = np.sqrt(mean_squared_error(y, y_pred_tpot_capped))
        tpot_r2 = r2_score(y, y_pred_tpot_capped)
    else:
        print(f"Warning: TPOT model not found at {tpot_model_path}. TPOT metrics will be 0.")

    # Save metrics for both
    metrics = {
        "tpot": {
            "r2_score": float(tpot_r2),
            "rmse": float(tpot_rmse)
        },
        "static": {
            "name": "Random Forest",
            "r2_score": float(static_r2),
            "rmse": float(static_rmse)
        },
        "total_songs": len(df_clean)
    }

    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"Metrics saved to {metrics_path}")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    train_and_evaluate()
