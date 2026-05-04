import os
import joblib
import webbrowser
import threading
import time
from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__, static_folder='.', static_url_path='')

# Load trained models
TPOT_MODEL_PATH = "outputs/tpot_viral_model.pkl"
STATIC_MODEL_PATH = "outputs/static_model.pkl"

tpot_model = None
static_model = None

try:
    tpot_model = joblib.load(TPOT_MODEL_PATH)
    print("TPOT AutoML model loaded successfully.")
except Exception as e:
    print(f"Warning: Could not load TPOT model at {TPOT_MODEL_PATH}. Error: {e}")

try:
    static_model = joblib.load(STATIC_MODEL_PATH)
    print("Static ML model loaded successfully.")
except Exception as e:
    print(f"Warning: Could not load Static model at {STATIC_MODEL_PATH}. Error: {e}")

@app.route('/')
def index():
    """Serve the main HTML page."""
    return send_from_directory('.', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests using both AutoML and Static ML models."""
    if tpot_model is None and static_model is None:
        return jsonify({"success": False, "error": "No models loaded. Please train the models first."})

    try:
        data = request.json

        features = [
            data.get("views", 0),
            data.get("likes", 0),
            data.get("comments", 0),
            data.get("views_per_day", 0),
            data.get("engagement_rate", 0),
            data.get("like_view_ratio", 0),
            data.get("comment_view_ratio", 0)
        ]

        tpot_pred_val = 0.0
        tpot_model_name = "TPOT Pipeline"
        if tpot_model:
            tpot_pred = tpot_model.predict([features])[0]
            tpot_pred_val = float(tpot_pred)
            tpot_pred_val = max(0.0, min(1.0, tpot_pred_val))
            try:
                if hasattr(tpot_model, 'steps'):
                    tpot_model_name = tpot_model.steps[-1][1].__class__.__name__
            except:
                pass

        static_pred_val = 0.0
        static_model_name = "Random Forest"
        if static_model:
            # Assuming Random Forest takes a DataFrame or list of lists, but some require feature names if trained with them
            # We will pass the list just like TPOT does.
            # To avoid feature name warnings, we can pass it as a dataframe if needed, but list usually works, 
            # though sklearn might warn. We will ignore the warning.
            import pandas as pd
            feature_cols = ['views', 'likes', 'comments', 'views_per_day', 'engagement_rate',
                            'like_view_ratio', 'comment_view_ratio']
            feature_df = pd.DataFrame([features], columns=feature_cols)
            
            static_pred = static_model.predict(feature_df)[0]
            static_pred_val = float(static_pred)
            static_pred_val = max(0.0, min(1.0, static_pred_val))
        
        # Ensure engagement rate isn't logically impossible (> 1.0)
        safe_engagement_rate = min(1.0, data.get("engagement_rate", 0))

        return jsonify({
            "success": True,
            "tpot_prediction": tpot_pred_val,
            "tpot_model": tpot_model_name,
            "static_prediction": static_pred_val,
            "static_model": static_model_name,
            "stats": {
                "views": data.get("views", 0),
                "likes": data.get("likes", 0),
                "comments": data.get("comments", 0),
                "engagement_rate": safe_engagement_rate
            }
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/analyze_url', methods=['POST'])
def analyze_url():
    """Dynamically analyze a YouTube URL to extract or simulate metrics."""
    data = request.json
    url = data.get('url', '')
    if not url:
        return jsonify({"success": False, "error": "No URL provided."})
    
    try:
        import yt_dlp
        from datetime import datetime
        ydl_opts = {
            'quiet': True, 
            'extract_flat': False, 
            'skip_download': True,
            'noplaylist': True,
            'playlist_items': '1' # Only get the first item if it's a playlist
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            
            if 'entries' in info and info['entries']:
                info = info['entries'][0]
            
            views = info.get('view_count', 0)
            likes = info.get('like_count', 0)
            comments = info.get('comment_count', 0)
            upload_date_str = info.get('upload_date', None)
            
            days_old = 7
            if upload_date_str:
                try:
                    upload_date = datetime.strptime(upload_date_str, '%Y%m%d')
                    days_old = max(1, (datetime.now() - upload_date).days)
                except ValueError:
                    pass
                
            # Fallbacks if info is completely missing
            if not views:
                views = 500000
            if not likes:
                likes = int(views * 0.05)
            if not comments:
                comments = int(views * 0.005)
                
            return jsonify({
                "success": True,
                "views": views,
                "likes": likes,
                "comments": comments,
                "days_old": days_old,
                "title": info.get('title', 'Unknown Video')
            })
    except Exception as e:
        # Fallback to pseudo-random dynamic based on URL hash if yt-dlp fails
        import hashlib
        import random
        
        # Seed random with URL so it's consistent for the same URL
        hash_val = int(hashlib.md5(url.encode()).hexdigest(), 16)
        random.seed(hash_val)
        
        views = random.randint(50000, 5000000)
        likes = int(views * random.uniform(0.02, 0.12))
        comments = int(likes * random.uniform(0.05, 0.15))
        days_old = random.randint(10, 365)
        
        return jsonify({
            "success": True,
            "views": views,
            "likes": likes,
            "comments": comments,
            "days_old": days_old,
            "title": "Dynamic Simulation (Failed to fetch real data)",
            "warning": str(e)
        })

def open_browser():
    """Wait a brief moment for the server to start and then open the browser."""
    time.sleep(1.5)
    print("\nOpening your browser to view the app...")
    webbrowser.open('http://127.0.0.1:5000/')

if __name__ == "__main__":
    PORT = 5000
    print(f"Starting Unified Viral Music Application on port {PORT}...")
    
    # Open the browser automatically in a separate thread
    threading.Thread(target=open_browser, daemon=True).start()
    
    # Run the Flask app
    app.run(host='127.0.0.1', port=PORT, debug=False)
