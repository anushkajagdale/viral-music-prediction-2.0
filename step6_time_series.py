# ============================================================
# STEP 6: TIME SERIES FORECASTING
# Viral Music Trend Prediction
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings("ignore")
import os

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE  = "data/youtube_music_stats.csv"
OUTPUT_FILE = "data/youtube_music_timeseries.csv"
OUTPUT_DIR  = "outputs/time_series"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Plot style
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"]   = "white"
plt.rcParams["axes.grid"]        = True
plt.rcParams["grid.alpha"]       = 0.3
plt.rcParams["font.size"]        = 10
sns.set_theme(style="whitegrid")

# ============================================================
# LOAD DATA
# ============================================================

def load_data(filepath):
    print("\n📂 Loading dataset...")
    df = pd.read_csv(filepath)
    df["published_at"] = pd.to_datetime(
        df["published_at"], errors="coerce"
    )
    df = df.dropna(subset=["published_at"])
    print(f"   Shape : {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"   Date Range: {df['published_at'].min().date()} "
          f"to {df['published_at'].max().date()}")
    return df

# ============================================================
# STEP 6A: PREPARE TIME SERIES DATA
# ============================================================

def prepare_time_series(df):
    print("\n" + "=" * 60)
    print("STEP 6A: PREPARING TIME SERIES DATA")
    print("=" * 60)

    print("""
    📌 WHAT IS TIME SERIES?
    ========================
    Time series is data collected at regular time intervals.

    In our project:
    → X-axis = Date (published_at)
    → Y-axis = Views / Viral Score / Song Count

    We will analyze:
    → How many songs were published each week?
    → How have average views changed over time?
    → What is the viral trend over time?
    → Can we forecast future viral trends?

    Time Series Components:
    → Trend     : Overall direction (up/down)
    → Seasonality: Regular repeating patterns
    → Noise     : Random fluctuations
    """)

    # -------------------------------------------------------
    # Create Daily time series
    # -------------------------------------------------------
    df["date"] = df["published_at"].dt.date
    df["date"] = pd.to_datetime(df["date"])

    # Daily aggregation
    daily_ts = df.groupby("date").agg(
        total_songs    = ("video_id",     "count"),
        avg_views      = ("views",        "mean"),
        total_views    = ("views",        "sum"),
        avg_likes      = ("likes",        "mean"),
        avg_comments   = ("comments",     "mean"),
        viral_count    = ("is_viral",     "sum"),
        avg_viral_score= ("viral_score",  "mean"),
        avg_engagement = ("engagement_rate","mean"),
        avg_virality   = ("virality_index","mean")
    ).reset_index()

    # Sort by date
    daily_ts = daily_ts.sort_values("date").reset_index(drop=True)

    # -------------------------------------------------------
    # Create Weekly time series (smoother)
    # -------------------------------------------------------
    df["week"] = df["published_at"].dt.to_period("W")\
        .dt.start_time

    weekly_ts = df.groupby("week").agg(
        total_songs    = ("video_id",     "count"),
        avg_views      = ("views",        "mean"),
        total_views    = ("views",        "sum"),
        avg_likes      = ("likes",        "mean"),
        viral_count    = ("is_viral",     "sum"),
        avg_viral_score= ("viral_score",  "mean"),
        avg_engagement = ("engagement_rate","mean"),
        avg_virality   = ("virality_index","mean")
    ).reset_index()

    weekly_ts = weekly_ts.sort_values("week")\
        .reset_index(drop=True)

    # -------------------------------------------------------
    # Create Monthly time series
    # -------------------------------------------------------
    df["month"] = df["published_at"].dt.to_period("M")\
        .dt.start_time

    monthly_ts = df.groupby("month").agg(
        total_songs    = ("video_id",     "count"),
        avg_views      = ("views",        "mean"),
        total_views    = ("views",        "sum"),
        avg_likes      = ("likes",        "mean"),
        viral_count    = ("is_viral",     "sum"),
        avg_viral_score= ("viral_score",  "mean"),
        avg_engagement = ("engagement_rate","mean"),
        avg_virality   = ("virality_index","mean")
    ).reset_index()

    monthly_ts = monthly_ts.sort_values("month")\
        .reset_index(drop=True)

    print(f"\n✅ Time Series Data Prepared:")
    print(f"   Daily  records  : {len(daily_ts)}")
    print(f"   Weekly records  : {len(weekly_ts)}")
    print(f"   Monthly records : {len(monthly_ts)}")

    print("\n📊 Weekly Time Series Sample:")
    print(weekly_ts[["week", "total_songs", "avg_views",
                      "viral_count",
                      "avg_viral_score"]].head(10).to_string())

    return daily_ts, weekly_ts, monthly_ts

# ============================================================
# STEP 6B: PLOT TIME SERIES
# ============================================================

def plot_time_series(daily_ts, weekly_ts, monthly_ts):
    print("\n" + "=" * 60)
    print("STEP 6B: TIME SERIES VISUALIZATION")
    print("=" * 60)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    fig.suptitle("Time Series Analysis - Viral Music Trends",
                 fontsize=16, fontweight="bold")

    ax1.plot(weekly_ts["week"], weekly_ts["avg_views"],
             color="#e74c3c", linewidth=2,
             marker="s", markersize=3)
    ax1.fill_between(weekly_ts["week"],
                     weekly_ts["avg_views"],
                     alpha=0.2, color="#e74c3c")
    ax1.set_title("Average Views Per Week",
                  fontsize=12, fontweight="bold")
    ax1.set_xlabel("Week", fontsize=10)
    ax1.set_ylabel("Avg Views", fontsize=10)
    ax1.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/6b_time_series.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    
    print(f"   ✅ Saved: {path}")

# ============================================================
# STEP 6C: MOVING AVERAGE
# ============================================================

def moving_average_analysis(weekly_ts):
    print("\n" + "=" * 60)
    print("STEP 6C: MOVING AVERAGE ANALYSIS")
    print("=" * 60)

    print("""
    📌 WHAT IS MOVING AVERAGE?
    ===========================
    Moving Average smooths out short-term fluctuations
    to reveal long-term trends.

    Types:
    → Simple Moving Average (SMA):
      Average of last N data points
      SMA(t) = (X(t) + X(t-1) + ... + X(t-N+1)) / N

    → Exponential Moving Average (EMA):
      Gives more weight to recent data points
      EMA(t) = α * X(t) + (1-α) * EMA(t-1)
      where α = 2/(N+1) is the smoothing factor

    Window Sizes we use:
    → 4-week  MA : Short-term trend
    → 8-week  MA : Medium-term trend
    → 12-week MA : Long-term trend

    Why Moving Average?
    → Removes noise from data
    → Shows clear trend direction
    → Baseline for comparison with ARIMA
    → Simple and interpretable
    """)

    ts = weekly_ts.copy()
    ts = ts.set_index("week")

    # Calculate Moving Averages for avg_views
    ts["MA_4"]  = ts["avg_views"].rolling(window=4,  min_periods=1).mean()
    ts["MA_8"]  = ts["avg_views"].rolling(window=8,  min_periods=1).mean()
    ts["MA_12"] = ts["avg_views"].rolling(window=12, min_periods=1).mean()

    # EMA
    ts["EMA_4"]  = ts["avg_views"].ewm(span=4,  adjust=False).mean()
    ts["EMA_8"]  = ts["avg_views"].ewm(span=8,  adjust=False).mean()
    ts["EMA_12"] = ts["avg_views"].ewm(span=12, adjust=False).mean()

    # Moving Average for viral score
    ts["vs_MA_4"]  = ts["avg_viral_score"].rolling(window=4,  min_periods=1).mean()
    ts["vs_MA_8"]  = ts["avg_viral_score"].rolling(window=8,  min_periods=1).mean()
    ts["vs_MA_12"] = ts["avg_viral_score"].rolling(window=12, min_periods=1).mean()

    # Calculate MA errors
    ts["MA_4_error"]  = np.abs(ts["avg_views"] - ts["MA_4"])
    ts["MA_8_error"]  = np.abs(ts["avg_views"] - ts["MA_8"])
    ts["MA_12_error"] = np.abs(ts["avg_views"] - ts["MA_12"])

    mae_4  = ts["MA_4_error"].mean()
    mae_8  = ts["MA_8_error"].mean()
    mae_12 = ts["MA_12_error"].mean()

    print(f"\n📊 Moving Average Performance (MAE):")
    print(f"   MA-4  (4-week)  MAE : {mae_4:>15,.2f}")
    print(f"   MA-8  (8-week)  MAE : {mae_8:>15,.2f}")
    print(f"   MA-12 (12-week) MAE : {mae_12:>15,.2f}")
    print(f"\n   Best Window: {'4-week' if mae_4 == min(mae_4, mae_8, mae_12) else '8-week' if mae_8 == min(mae_4, mae_8, mae_12) else '12-week'}")



    ts.reset_index(inplace=True)
    return ts

# ============================================================
# STEP 6D: STATIONARITY TEST (ADF TEST)
# ============================================================

def stationarity_test(weekly_ts):
    print("\n" + "=" * 60)
    print("STEP 6D: STATIONARITY TEST (ADF TEST)")
    print("=" * 60)

    print("""
    📌 WHAT IS STATIONARITY?
    =========================
    A time series is stationary when:
    → Mean is constant over time
    → Variance is constant over time
    → No trend or seasonality

    WHY IT MATTERS FOR ARIMA?
    → ARIMA requires stationary data
    → Non-stationary data must be differenced

    ADF TEST (Augmented Dickey-Fuller):
    → H0: Series is NON-stationary (has unit root)
    → H1: Series IS stationary

    If p < 0.05 → Series is STATIONARY ✅
    If p >= 0.05 → Series is NON-STATIONARY ❌
                   → Need to difference the data
    """)

    series = weekly_ts["avg_views"].dropna()

    def run_adf(data, label):
        result = adfuller(data.dropna())
        print(f"\n   ADF Test: {label}")
        print(f"   ADF Statistic : {result[0]:.4f}")
        print(f"   P-value       : {result[1]:.6f}")
        print(f"   Critical Values:")
        for key, val in result[4].items():
            print(f"      {key}: {val:.4f}")
        stationary = result[1] < 0.05
        print(f"   Result        : {'✅ STATIONARY' if stationary else '❌ NON-STATIONARY'}")
        return stationary

    # Test original series
    is_stationary = run_adf(series, "Original Series (avg_views)")

    # If not stationary, difference it
    series_diff = series.diff().dropna()
    run_adf(series_diff, "First Differenced Series")

    series_diff2 = series_diff.diff().dropna()
    run_adf(series_diff2, "Second Differenced Series")



    return is_stationary

# ============================================================
# STEP 6E: SEASONAL DECOMPOSITION
# ============================================================

def seasonal_decomposition(weekly_ts):
    print("\n" + "=" * 60)
    print("STEP 6E: SEASONAL DECOMPOSITION")
    print("=" * 60)

    print("""
    📌 WHAT IS SEASONAL DECOMPOSITION?
    =====================================
    Breaks time series into 3 components:

    → Trend      : Overall direction of data
    → Seasonality: Repeating patterns
    → Residual   : Random noise after removing trend & seasonality

    Model Types:
    → Additive    : Y = Trend + Seasonal + Residual
      (Use when seasonal variation is CONSTANT)
    → Multiplicative: Y = Trend × Seasonal × Residual
      (Use when seasonal variation GROWS with trend)

    We use Additive model for music views data.
    """)

    series = weekly_ts.set_index("week")["avg_views"].dropna()

    if len(series) >= 8:
        try:
            decomp = seasonal_decompose(
                series, model="additive", period=4
            )


        except Exception as e:
            print(f"   ⚠️  Decomposition warning: {e}")
    else:
        print("   ⚠️  Not enough data points for decomposition")

# ============================================================
# STEP 6F: ARIMA MODEL
# ============================================================

def arima_model(weekly_ts):
    print("\n" + "=" * 60)
    print("STEP 6F: ARIMA MODEL")
    print("=" * 60)

    print("""
    📌 WHAT IS ARIMA?
    ==================
    ARIMA = AutoRegressive Integrated Moving Average

    Parameters: ARIMA(p, d, q)
    → p (AR) : Number of lag observations (AutoRegressive)
               "How many past values to use?"
    → d (I)  : Degree of differencing (Integrated)
               "How many times to difference for stationarity?"
    → q (MA) : Size of moving average window
               "How many past errors to use?"

    How to choose p, d, q:
    → d: Number of times you need to difference
         for stationarity (from ADF test)
    → p: Look at PACF plot - where it cuts off
    → q: Look at ACF plot - where it cuts off

    We will use: ARIMA(1,1,1) as baseline
    And test:    ARIMA(2,1,2) as improved model

    We forecast:
    → Next 4 weeks of viral song trends
    → Next 4 weeks of average views

    Evaluation Metrics:
    → MAE  (Mean Absolute Error)
    → RMSE (Root Mean Square Error)
    → Lower = Better model
    """)

    series = weekly_ts.set_index("week")["avg_views"].dropna()

    if len(series) < 10:
        print("   ⚠️  Not enough data for ARIMA. Need 10+ weeks.")
        return None, None, None, None

    # Train/Test split (80/20)
    train_size = int(len(series) * 0.80)
    train      = series[:train_size]
    test       = series[train_size:]

    print(f"\n   Total weeks   : {len(series)}")
    print(f"   Training size : {len(train)} weeks (80%)")
    print(f"   Testing size  : {len(test)} weeks (20%)")

    if len(test) == 0:
        test = series[-3:]
        train= series[:-3]

    results_summary = []

    # -------------------------------------------------------
    # MODEL 1: ARIMA(1,1,1)
    # -------------------------------------------------------
    print("\n" + "-" * 50)
    print("ARIMA MODEL 1: ARIMA(1,1,1)")
    print("-" * 50)

    try:
        model1     = ARIMA(train, order=(1, 1, 1))
        fitted1    = model1.fit()
        forecast1  = fitted1.forecast(steps=len(test))
        forecast1  = pd.Series(forecast1.values,
                               index=test.index)

        mae1  = mean_absolute_error(test, forecast1)
        rmse1 = np.sqrt(mean_squared_error(test, forecast1))
        mape1 = np.mean(np.abs((test - forecast1) / test.replace(0, np.nan))) * 100

        results_summary.append({
            "Model": "ARIMA(1,1,1)",
            "MAE"  : mae1,
            "RMSE" : rmse1,
            "MAPE" : mape1
        })

        print(f"\n   ✅ ARIMA(1,1,1) fitted successfully")
        print(f"   MAE  : {mae1:>15,.2f}")
        print(f"   RMSE : {rmse1:>15,.2f}")
        print(f"   MAPE : {mape1:>15,.2f}%")
        print(f"\n   Model Summary:")
        print(f"   AIC  : {fitted1.aic:.2f}")
        print(f"   BIC  : {fitted1.bic:.2f}")

    except Exception as e:
        print(f"   ❌ ARIMA(1,1,1) failed: {e}")
        fitted1   = None
        forecast1 = None
        mae1      = rmse1 = mape1 = None

    # -------------------------------------------------------
    # MODEL 2: ARIMA(2,1,2)
    # -------------------------------------------------------
    print("\n" + "-" * 50)
    print("ARIMA MODEL 2: ARIMA(2,1,2)")
    print("-" * 50)

    try:
        model2    = ARIMA(train, order=(2, 1, 2))
        fitted2   = model2.fit()
        forecast2 = fitted2.forecast(steps=len(test))
        forecast2 = pd.Series(forecast2.values,
                               index=test.index)

        mae2  = mean_absolute_error(test, forecast2)
        rmse2 = np.sqrt(mean_squared_error(test, forecast2))
        mape2 = np.mean(np.abs((test - forecast2) / test.replace(0, np.nan))) * 100

        results_summary.append({
            "Model": "ARIMA(2,1,2)",
            "MAE"  : mae2,
            "RMSE" : rmse2,
            "MAPE" : mape2
        })

        print(f"\n   ✅ ARIMA(2,1,2) fitted successfully")
        print(f"   MAE  : {mae2:>15,.2f}")
        print(f"   RMSE : {rmse2:>15,.2f}")
        print(f"   MAPE : {mape2:>15,.2f}%")
        print(f"\n   Model Summary:")
        print(f"   AIC  : {fitted2.aic:.2f}")
        print(f"   BIC  : {fitted2.bic:.2f}")

    except Exception as e:
        print(f"   ❌ ARIMA(2,1,2) failed: {e}")
        fitted2   = None
        forecast2 = None
        mae2      = rmse2 = mape2 = None

    # -------------------------------------------------------
    # MODEL 3: ARIMA(1,1,2)
    # -------------------------------------------------------
    print("\n" + "-" * 50)
    print("ARIMA MODEL 3: ARIMA(1,1,2)")
    print("-" * 50)

    try:
        model3    = ARIMA(train, order=(1, 1, 2))
        fitted3   = model3.fit()
        forecast3 = fitted3.forecast(steps=len(test))
        forecast3 = pd.Series(forecast3.values,
                               index=test.index)

        mae3  = mean_absolute_error(test, forecast3)
        rmse3 = np.sqrt(mean_squared_error(test, forecast3))
        mape3 = np.mean(np.abs((test - forecast3) / test.replace(0, np.nan))) * 100

        results_summary.append({
            "Model": "ARIMA(1,1,2)",
            "MAE"  : mae3,
            "RMSE" : rmse3,
            "MAPE" : mape3
        })

        print(f"\n   ✅ ARIMA(1,1,2) fitted successfully")
        print(f"   MAE  : {mae3:>15,.2f}")
        print(f"   RMSE : {rmse3:>15,.2f}")
        print(f"   MAPE : {mape3:>15,.2f}%")
        print(f"\n   Model Summary:")
        print(f"   AIC  : {fitted3.aic:.2f}")
        print(f"   BIC  : {fitted3.bic:.2f}")

    except Exception as e:
        print(f"   ❌ ARIMA(1,1,2) failed: {e}")
        fitted3   = None
        forecast3 = None
        mae3      = rmse3 = mape3 = None

    # Model comparison
    print("\n" + "=" * 60)
    print("📊 MODEL COMPARISON SUMMARY:")
    print("=" * 60)
    results_df = pd.DataFrame(results_summary)
    if not results_df.empty:
        print(results_df.round(2).to_string(index=False))
        best_model = results_df.loc[results_df["RMSE"].idxmin(), "Model"]
        print(f"\n   🏆 BEST MODEL: {best_model}")
        print(f"   (Lowest RMSE = Most accurate predictions)")

    return fitted1, fitted2, fitted3, test, train, series

# ============================================================
# STEP 6G: FUTURE FORECAST
# ============================================================

def future_forecast(fitted_models, weekly_ts):
    print("\n" + "=" * 60)
    print("STEP 6G: FUTURE FORECAST (NEXT 8 WEEKS)")
    print("=" * 60)

    print("""
    📌 FORECASTING FUTURE VIRAL TRENDS:
    =====================================
    Using our best ARIMA model, we forecast:
    → Next 8 weeks of average views
    → Confidence interval (upper & lower bounds)
    → Trend direction (growing / declining)
    """)

    forecast_steps = 8

    fitted1, fitted2, fitted3, test, train, full_series = fitted_models

    # Use best available model
    best_fitted = fitted1 or fitted2 or fitted3
    if best_fitted is None:
        print("   ❌ No valid ARIMA model available")
        return

    try:
        # Refit on full series for best forecast
        model_full  = ARIMA(full_series, order=(1, 1, 1))
        fitted_full = model_full.fit()

        # Forecast
        forecast_result = fitted_full.get_forecast(
            steps=forecast_steps
        )
        forecast_mean   = forecast_result.predicted_mean
        conf_int        = forecast_result.conf_int()

        # Create future dates
        last_date    = full_series.index[-1]
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(weeks=1),
            periods=forecast_steps,
            freq="W"
        )
        forecast_mean.index = future_dates
        conf_int.index      = future_dates

        print("\n📊 FORECASTED VALUES (Next 8 Weeks):")
        print(f"\n   {'Week':<15} {'Forecast':>15} "
              f"{'Lower CI':>15} {'Upper CI':>15}")
        print("   " + "-" * 60)
        for date, val in forecast_mean.items():
            lo = conf_int.loc[date].iloc[0]
            hi = conf_int.loc[date].iloc[1]
            print(f"   {str(date.date()):<15} "
                  f"{val:>15,.0f} "
                  f"{lo:>15,.0f} "
                  f"{hi:>15,.0f}")

        trend = "📈 UPWARD" if forecast_mean.iloc[-1] > \
                forecast_mean.iloc[0] else "📉 DOWNWARD"
        change_pct = ((forecast_mean.iloc[-1] -
                       forecast_mean.iloc[0]) /
                      forecast_mean.iloc[0] * 100)
        print(f"\n   Trend Direction : {trend}")
        print(f"   Expected Change : {change_pct:+.1f}% over 8 weeks")

        # Plot forecast
        fig, ax1 = plt.subplots(figsize=(10, 6))
        fig.suptitle("ARIMA Forecast - Future Viral Music Trends",
                     fontsize=16, fontweight="bold")

        # Historical
        ax1.plot(full_series.index, full_series.values,
                 color="#3498db", linewidth=2,
                 label="Historical Data")

        # Forecast
        ax1.plot(forecast_mean.index, forecast_mean.values,
                 color="#e74c3c", linewidth=2.5,
                 linestyle="--", marker="o",
                 markersize=6, label="Forecast")

        # Confidence interval
        ax1.fill_between(
            conf_int.index,
            conf_int.iloc[:, 0],
            conf_int.iloc[:, 1],
            alpha=0.3, color="#e74c3c",
            label="95% Confidence Interval"
        )

        # Vertical line at forecast start
        ax1.axvline(x=full_series.index[-1],
                    color="gray", linestyle=":",
                    linewidth=2, label="Forecast Start")

        ax1.set_title(
            "Historical + 8-Week Forecast (Avg Views per Week)",
            fontsize=13, fontweight="bold"
        )
        ax1.set_xlabel("Week", fontsize=11)
        ax1.set_ylabel("Average Views", fontsize=11)
        ax1.legend(fontsize=10)
        ax1.tick_params(axis="x", rotation=30)

        plt.tight_layout()
        path = f"{OUTPUT_DIR}/6g_forecast.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        
        print(f"\n   ✅ Saved: {path}")

        return forecast_mean, conf_int

    except Exception as e:
        print(f"   ❌ Forecast error: {e}")
        return None, None

# ============================================================
# STEP 6H: MODEL EVALUATION COMPARISON
# ============================================================

def model_evaluation(weekly_ts):
    print("\n" + "=" * 60)
    print("STEP 6H: COMPLETE MODEL EVALUATION")
    print("=" * 60)

    print("""
    📌 EVALUATION METRICS EXPLAINED:
    ==================================
    MAE (Mean Absolute Error):
    → Average of absolute differences
    → MAE = mean(|actual - predicted|)
    → Easy to interpret, same unit as data
    → Less sensitive to large errors

    RMSE (Root Mean Square Error):
    → Square root of average squared differences
    → RMSE = sqrt(mean((actual - predicted)²))
    → Penalizes large errors more than MAE
    → Most commonly used metric

    MAPE (Mean Absolute Percentage Error):
    → Average percentage error
    → MAPE = mean(|actual - predicted| / actual) * 100
    → Scale-independent, easy to understand
    → 10% MAPE = predictions off by 10% on average

    LOWER VALUES = BETTER MODEL
    """)

    series = weekly_ts.set_index("week")["avg_views"].dropna()

    if len(series) < 10:
        print("   ⚠️  Insufficient data for evaluation")
        return

    train_size = int(len(series) * 0.80)
    train      = series[:train_size]
    test       = series[train_size:]

    if len(test) == 0:
        test  = series[-3:]
        train = series[:-3]

    all_results = []

    # Test multiple ARIMA configurations
    configs = [(1,1,1), (2,1,1), (1,1,2), (2,1,2), (0,1,1)]

    for p, d, q in configs:
        try:
            model      = ARIMA(train, order=(p, d, q))
            fitted     = model.fit()
            forecast   = fitted.forecast(steps=len(test))
            forecast_s = pd.Series(forecast.values,
                                   index=test.index)

            mae  = mean_absolute_error(test, forecast_s)
            rmse = np.sqrt(mean_squared_error(test, forecast_s))
            mape = np.mean(
                np.abs((test - forecast_s) /
                       test.replace(0, np.nan))
            ) * 100

            all_results.append({
                "Model"    : f"ARIMA({p},{d},{q})",
                "MAE"      : round(mae, 2),
                "RMSE"     : round(rmse, 2),
                "MAPE(%)"  : round(mape, 2),
                "AIC"      : round(fitted.aic, 2),
                "BIC"      : round(fitted.bic, 2)
            })
            print(f"   ✅ ARIMA({p},{d},{q}) - "
                  f"MAE={mae:,.0f}  "
                  f"RMSE={rmse:,.0f}  "
                  f"MAPE={mape:.1f}%")

        except Exception as e:
            print(f"   ⚠️  ARIMA({p},{d},{q}) failed: {e}")

    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df = results_df.sort_values("RMSE")

        print("\n📊 COMPLETE MODEL COMPARISON:")
        print(results_df.to_string(index=False))

        best = results_df.iloc[0]
        print(f"\n🏆 BEST MODEL: {best['Model']}")
        print(f"   MAE    : {best['MAE']:>15,.2f}")
        print(f"   RMSE   : {best['RMSE']:>15,.2f}")
        print(f"   MAPE   : {best['MAPE(%)']:>15,.2f}%")
        print(f"   AIC    : {best['AIC']:>15,.2f}")
        print(f"   BIC    : {best['BIC']:>15,.2f}")

        # Moving Average comparison
        ma_series = weekly_ts["avg_views"].dropna()
        ma_train  = ma_series.iloc[:train_size]
        ma_test   = ma_series.iloc[train_size:]

        if len(ma_test) > 0:
            ma4_pred = ma_train.rolling(window=4,
                                        min_periods=1)\
                .mean().iloc[-1]
            ma4_fore = pd.Series(
                [ma4_pred] * len(ma_test),
                index=ma_test.index
            )
            ma_mae  = mean_absolute_error(ma_test, ma4_fore)
            ma_rmse = np.sqrt(
                mean_squared_error(ma_test, ma4_fore)
            )

            print(f"\n📊 ARIMA vs MOVING AVERAGE:")
            print(f"   {'Model':<20} {'MAE':>15} {'RMSE':>15}")
            print("   " + "-" * 52)
            print(f"   {'Best ARIMA':<20} "
                  f"{best['MAE']:>15,.2f} "
                  f"{best['RMSE']:>15,.2f}")
            print(f"   {'Moving Average-4':<20} "
                  f"{ma_mae:>15,.2f} "
                  f"{ma_rmse:>15,.2f}")

            winner = "ARIMA" if best["RMSE"] < ma_rmse \
                     else "Moving Average"
            print(f"\n   🏆 WINNER: {winner}")
            print(f"   Reason: Lower RMSE = "
                  f"More accurate predictions")



        return results_df

# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    print("=" * 60)
    print("   🎵 VIRAL MUSIC TREND PREDICTION")
    print("   STEP 6: TIME SERIES FORECASTING")
    print("=" * 60)

    if not os.path.exists(INPUT_FILE):
        print(f"\n❌ ERROR: '{INPUT_FILE}' not found!")
        print("   Please run step5_statistical_analysis.py first!")
        return

    # Load data
    df = load_data(INPUT_FILE)

    # Prepare time series
    daily_ts, weekly_ts, monthly_ts = prepare_time_series(df)

    # Visualize time series
    plot_time_series(daily_ts, weekly_ts, monthly_ts)

    # Moving Average
    weekly_ts_ma = moving_average_analysis(weekly_ts)

    # Stationarity test
    stationarity_test(weekly_ts)

    # Seasonal decomposition
    seasonal_decomposition(weekly_ts)

    # ARIMA Models
    fitted_models = arima_model(weekly_ts)

    # Future Forecast
    if fitted_models[0] is not None or \
       fitted_models[1] is not None:
        future_forecast(fitted_models, weekly_ts)

    # Model Evaluation
    model_evaluation(weekly_ts)

    # Save output
    df.to_csv(OUTPUT_FILE, index=False)

    print("\n" + "=" * 60)
    print("✅ STEP 6 COMPLETE!")
    print(f"   Output File : {OUTPUT_FILE}")
    print(f"   Graphs saved: {OUTPUT_DIR}/")
    print("   Files generated:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        print(f"   📊 {f}")
    print("   ➡️  Next: python step7_visualization.py")
    print("=" * 60)


if __name__ == "__main__":
    main()