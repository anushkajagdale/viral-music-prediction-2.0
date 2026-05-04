# Viral Music Prediction Dashboard

A comprehensive dashboard for analyzing viral music trends on YouTube and predicting new song virality.

## Features

- **Overview**: Dataset summary and top viral songs
- **How Predictions Work**: Detailed explanation of the prediction algorithm
- **Predict New Song**: Interactive form to predict virality of new songs
- **Trend Analysis**: Monthly trends in views, viral count, and engagement
- **Viral Ranking**: Top songs by viral score, distribution by language, and tier breakdowns
- **Language Analysis**: Song count, viral rates, and performance metrics by language

## Prediction System

### What the Predictions Mean
- **"Will Go MEGA VIRAL"**: Top 5% - Massive global hits (100M+ views)
- **"Will Go VIRAL"**: Top 15% - Strong viral momentum (10M-50M views)
- **"Likely Trending"**: Top 35% - Growing popularity
- **"Possibly Rising"**: Top 60% - Early growth signs
- **"Unlikely to Viral"**: Bottom 40% - Average performance

### How It Works
The system uses a weighted formula combining:
- **Viral Score** (40%): Current views, likes, comments
- **Engagement Momentum** (30%): Growth speed and engagement quality
- **Channel Influence** (20%): Artist popularity and track record
- **Language Trend** (10%): Historical performance by language

### Input Requirements
To predict a new song's virality, you need:
- Song title and artist name
- Language
- Current views, likes, comments
- Days since release

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the backend analysis (optional, data is already processed):
   ```bash
   python step1_data_collection.py
   python step2_data_cleaning.py
   python step3_eda.py
   python step4_feature_engineering.py
   python step5_statistical_analysis.py
   python step6_time_series.py
   python step7_visualization.py
   ```

3. Launch the frontend:
   ```bash
   streamlit run frontend.py
   ```

4. Open your browser to `http://localhost:8501`

## Data

The dashboard uses processed data from `data/youtube_music_stats.csv` containing:
- 3,215 songs from various languages
- Viral score predictions
- Engagement metrics
- Language and regional analysis

## Technologies

- **Backend**: Python, pandas, scikit-learn, statsmodels
- **Frontend**: Streamlit
- **Visualization**: Matplotlib, Seaborn

## API Integration

The project integrates with YouTube Data API v3 for music video data collection.