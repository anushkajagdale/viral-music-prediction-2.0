# ============================================================
# STEP 1: DATA COLLECTION - Viral Music Trend Prediction
# MAXIMUM DATA - All Languages, All Regions, Viral Detection
# ============================================================

import requests
import pandas as pd
import numpy as np
import os
import time
from datetime import datetime, timedelta

# ============================================================
# CONFIGURATION
# ============================================================

API_KEY = "AIzaSyBxRSnbpg7Am5PsZCKpcfqNjIr7nJe1GzQ"   # Paste your real YouTube API key here

os.makedirs("data", exist_ok=True)
OUTPUT_FILE = "data/youtube_music_data.csv"

# ============================================================
# ALL REGIONS WITH LANGUAGE INFO
# ============================================================

REGIONS = {
    "IN" : "Hindi/Marathi/Tamil/Telugu/Punjabi/Bengali",
    "US" : "English",
    "GB" : "English-UK",
    "KR" : "Korean",
    "BR" : "Portuguese",
    "MX" : "Spanish-Mexico",
    "JP" : "Japanese",
    "FR" : "French",
    "DE" : "German",
    "TR" : "Turkish",
    "NG" : "Nigerian/Afrobeats",
    "PH" : "Filipino",
    "ID" : "Indonesian",
    "PK" : "Urdu",
    "EG" : "Arabic",
    "TH" : "Thai",
    "VN" : "Vietnamese",
    "IT" : "Italian",
    "ES" : "Spanish-Spain",
    "RU" : "Russian",
    "SA" : "Arabic-Saudi",
    "MA" : "Arabic-Moroccan",
    "GH" : "Ghanaian",
    "ZA" : "South African",
    "AR" : "Spanish-Argentina",
    "CO" : "Spanish-Colombia",
    "AU" : "English-Australia",
    "CA" : "English-Canada",
    "NP" : "Nepali",
    "BD" : "Bengali",
    "LK" : "Sinhala/Tamil",
    "MM" : "Burmese",
    "KH" : "Khmer",
    "MN" : "Mongolian",
    "UZ" : "Uzbek",
    "KZ" : "Kazakh",
}

# ============================================================
# SEARCH KEYWORDS - All Languages + Viral Focus
# ============================================================

SEARCH_QUERIES = [
    # Marathi
    "marathi song 2024", "marathi song 2025",
    "trending marathi song", "viral marathi song",
    "new marathi song 2025", "marathi remix 2024",

    # Hindi
    "hindi song 2024", "hindi song 2025",
    "trending hindi song", "viral hindi song",
    "bollywood 2024", "bollywood 2025",
    "new bollywood song 2025",

    # Tamil
    "tamil song 2024", "tamil song 2025",
    "trending tamil song", "viral tamil song",
    "kollywood 2024",

    # Telugu
    "telugu song 2024", "telugu song 2025",
    "trending telugu song", "tollywood 2024",

    # Punjabi
    "punjabi song 2024", "punjabi song 2025",
    "trending punjabi song", "viral punjabi song",

    # Bengali
    "bengali song 2024", "bangla song 2025",
    "trending bengali song",

    # Kannada
    "kannada song 2024", "kannada song 2025",
    "trending kannada song",

    # Malayalam
    "malayalam song 2024", "malayalam song 2025",
    "trending malayalam song",

    # Gujarati
    "gujarati song 2024", "viral gujarati song",

    # Bhojpuri
    "bhojpuri song 2024", "bhojpuri song 2025",

    # English
    "english song 2024", "english song 2025",
    "trending english song", "viral english song",
    "pop song 2024", "pop song 2025",
    "top hits 2024", "top hits 2025",

    # Korean
    "kpop 2024", "kpop 2025",
    "trending kpop song", "viral kpop",
    "blackpink 2024", "bts 2024",

    # Spanish
    "reggaeton 2024", "latin song 2024",
    "spanish song 2025", "viral spanish song",

    # Portuguese
    "musica brasileira 2024", "funk brasileiro 2024",
    "sertanejo 2024",

    # Arabic
    "arabic song 2024", "arabic song 2025",
    "trending arabic song",

    # Turkish
    "turk muzik 2024", "turkish song 2025",
    "trending turkish song",

    # Japanese
    "japanese song 2024", "j-pop 2025",
    "trending japanese song",

    # French
    "french song 2024", "musique francaise 2025",

    # Afrobeats
    "afrobeats 2024", "afrobeats 2025",
    "afropop 2024", "viral afrobeats",

    # Nepali / Urdu
    "nepali song 2024", "urdu song 2024",

    # Viral / Trending
    "viral song 2024", "viral song 2025",
    "trending song 2024", "trending song 2025",
    "most viral music 2024", "most viral music 2025",
    "viral music video 2025",
    "youtube trending music 2025",
    "top viral songs 2025",
]

# ============================================================
# LANGUAGE DETECTION FROM TITLE / CHANNEL
# ============================================================

def detect_language_from_title(title, channel, default_lang, region):
    """
    Smarter language detection using title characters,
    region, and YouTube's default_language field combined.
    """
    if default_lang and default_lang not in ["unknown", "", None]:
        return default_lang

    title_lower = (title + " " + channel).lower()

    # Detect script-based languages
    for char in title:
        code = ord(char)
        if 0x0900 <= code <= 0x097F:   return "hi"   # Devanagari = Hindi/Marathi
        if 0x0980 <= code <= 0x09FF:   return "bn"   # Bengali
        if 0x0A00 <= code <= 0x0A7F:   return "pa"   # Punjabi/Gurmukhi
        if 0x0A80 <= code <= 0x0AFF:   return "gu"   # Gujarati
        if 0x0B00 <= code <= 0x0B7F:   return "or"   # Odia
        if 0x0B80 <= code <= 0x0BFF:   return "ta"   # Tamil
        if 0x0C00 <= code <= 0x0C7F:   return "te"   # Telugu
        if 0x0C80 <= code <= 0x0CFF:   return "kn"   # Kannada
        if 0x0D00 <= code <= 0x0D7F:   return "ml"   # Malayalam
        if 0x0E00 <= code <= 0x0E7F:   return "th"   # Thai
        if 0x0E80 <= code <= 0x0EFF:   return "lo"   # Lao
        if 0x1000 <= code <= 0x109F:   return "my"   # Burmese
        if 0x1800 <= code <= 0x18AF:   return "mn"   # Mongolian
        if 0x3040 <= code <= 0x309F:   return "ja"   # Japanese Hiragana
        if 0x30A0 <= code <= 0x30FF:   return "ja"   # Japanese Katakana
        if 0x4E00 <= code <= 0x9FFF:   return "zh"   # Chinese
        if 0xAC00 <= code <= 0xD7AF:   return "ko"   # Korean
        if 0x0600 <= code <= 0x06FF:   return "ar"   # Arabic/Urdu
        if 0x0400 <= code <= 0x04FF:   return "ru"   # Cyrillic = Russian

    # Keyword-based detection
    if any(w in title_lower for w in ["marathi", "मराठी"]):           return "mr"
    if any(w in title_lower for w in ["hindi", "bollywood", "हिंदी"]): return "hi"
    if any(w in title_lower for w in ["punjabi", "ਪੰਜਾਬੀ"]):           return "pa"
    if any(w in title_lower for w in ["tamil", "kollywood"]):          return "ta"
    if any(w in title_lower for w in ["telugu", "tollywood"]):         return "te"
    if any(w in title_lower for w in ["kannada"]):                     return "kn"
    if any(w in title_lower for w in ["malayalam"]):                   return "ml"
    if any(w in title_lower for w in ["bengali", "bangla"]):           return "bn"
    if any(w in title_lower for w in ["gujarati"]):                    return "gu"
    if any(w in title_lower for w in ["bhojpuri"]):                    return "bho"
    if any(w in title_lower for w in ["nepali"]):                      return "ne"
    if any(w in title_lower for w in ["urdu"]):                        return "ur"
    if any(w in title_lower for w in ["kpop", "k-pop"]):               return "ko"
    if any(w in title_lower for w in ["afrobeat", "afropop"]):         return "yo"
    if any(w in title_lower for w in ["reggaeton", "latin"]):          return "es"

    # Region-based fallback
    region_lang_map = {
        "IN": "hi", "US": "en", "GB": "en", "KR": "ko",
        "BR": "pt", "MX": "es", "JP": "ja", "FR": "fr",
        "DE": "de", "TR": "tr", "NG": "yo", "PH": "tl",
        "ID": "id", "PK": "ur", "EG": "ar", "TH": "th",
        "VN": "vi", "IT": "it", "ES": "es", "RU": "ru",
        "SA": "ar", "MA": "ar", "AU": "en", "CA": "en",
        "NP": "ne", "BD": "bn", "LK": "ta",
    }

    return region_lang_map.get(region, "unknown")


# ============================================================
# LANGUAGE CODE TO FULL NAME MAPPING
# ============================================================

LANGUAGE_NAMES = {
    "hi"  : "Hindi",
    "mr"  : "Marathi",
    "ta"  : "Tamil",
    "te"  : "Telugu",
    "pa"  : "Punjabi",
    "bn"  : "Bengali",
    "kn"  : "Kannada",
    "ml"  : "Malayalam",
    "gu"  : "Gujarati",
    "bho" : "Bhojpuri",
    "ne"  : "Nepali",
    "ur"  : "Urdu",
    "en"  : "English",
    "ko"  : "Korean",
    "ja"  : "Japanese",
    "zh"  : "Chinese",
    "es"  : "Spanish",
    "pt"  : "Portuguese",
    "fr"  : "French",
    "de"  : "German",
    "ar"  : "Arabic",
    "tr"  : "Turkish",
    "ru"  : "Russian",
    "id"  : "Indonesian",
    "th"  : "Thai",
    "vi"  : "Vietnamese",
    "tl"  : "Filipino",
    "it"  : "Italian",
    "yo"  : "Yoruba/Afrobeats",
    "my"  : "Burmese",
    "mn"  : "Mongolian",
    "unknown": "Unknown",
}


# ============================================================
# STEP 1A: GET VIDEO IDs BY REGION
# ============================================================

def get_video_ids_by_region(api_key, region_code):
    all_ids    = []
    page_token = None
    page_count = 0

    while page_count < 5:
        url    = "https://www.googleapis.com/youtube/v3/videos"
        params = {
            "part"            : "id",
            "chart"           : "mostPopular",
            "videoCategoryId" : "10",
            "regionCode"      : region_code,
            "maxResults"      : 50,
            "key"             : api_key
        }
        if page_token:
            params["pageToken"] = page_token

        try:
            response = requests.get(url, params=params, timeout=10)

            if response.status_code == 403:
                return all_ids

            if response.status_code != 200:
                break

            data       = response.json()
            items      = data.get("items", [])
            page_token = data.get("nextPageToken", None)

            for item in items:
                all_ids.append(item["id"])

            page_count += 1
            time.sleep(0.2)

            if not page_token:
                break

        except Exception:
            break

    return all_ids


# ============================================================
# STEP 1B: SEARCH VIDEOS BY QUERY
# ============================================================

def search_videos_by_query(api_key, query):
    all_ids = []
    url     = "https://www.googleapis.com/youtube/v3/search"
    params  = {
        "part"            : "id",
        "q"               : query,
        "type"            : "video",
        "videoCategoryId" : "10",
        "order"           : "viewCount",
        "maxResults"      : 50,
        "key"             : api_key
    }

    try:
        response = requests.get(url, params=params, timeout=10)

        if response.status_code == 403:
            return None   # Quota exceeded

        if response.status_code != 200:
            return all_ids

        data  = response.json()
        items = data.get("items", [])

        for item in items:
            if item.get("id", {}).get("kind") == "youtube#video":
                all_ids.append(item["id"]["videoId"])

        time.sleep(0.3)

    except Exception:
        pass

    return all_ids


# ============================================================
# STEP 1C: GET FULL VIDEO DETAILS
# ============================================================

def get_video_details(api_key, video_ids, region_map):
    all_videos    = []
    batch_size    = 50
    total         = len(video_ids)
    total_batches = (total // batch_size) + 1

    print(f"\n   Processing {total} videos in {total_batches} batches...")

    for batch_num in range(total_batches):
        start = batch_num * batch_size
        end   = start + batch_size
        batch = video_ids[start:end]

        if not batch:
            break

        url    = "https://www.googleapis.com/youtube/v3/videos"
        params = {
            "part" : "snippet,statistics,contentDetails",
            "id"   : ",".join(batch),
            "key"  : api_key
        }

        try:
            response = requests.get(url, params=params, timeout=10)

            if response.status_code != 200:
                continue

            data  = response.json()
            items = data.get("items", [])

            for video in items:
                snippet = video.get("snippet", {})
                stats   = video.get("statistics", {})
                content = video.get("contentDetails", {})

                vid_id      = video.get("id", "")
                title       = snippet.get("title", "Unknown")
                channel     = snippet.get("channelTitle", "Unknown")
                region      = region_map.get(vid_id, "search")
                default_lang= snippet.get("defaultLanguage", "")
                pub_at      = snippet.get("publishedAt", "")

                # Smart language detection
                detected_lang = detect_language_from_title(
                    title, channel, default_lang, region
                )
                language_name = LANGUAGE_NAMES.get(detected_lang, detected_lang)

                # Parse published date
                try:
                    pub_dt    = datetime.strptime(pub_at, "%Y-%m-%dT%H:%M:%SZ")
                    pub_year  = pub_dt.year
                    pub_month = pub_dt.month
                    pub_date  = pub_dt.strftime("%Y-%m-%d")
                    days_old  = (datetime.now() - pub_dt).days
                except:
                    pub_year  = None
                    pub_month = None
                    pub_date  = None
                    days_old  = None

                views    = int(stats.get("viewCount", 0))
                likes    = int(stats.get("likeCount", 0))    if "likeCount"    in stats else np.nan
                comments = int(stats.get("commentCount", 0)) if "commentCount" in stats else np.nan

                # Calculate views per day (virality indicator)
                views_per_day = round(views / days_old, 2) if days_old and days_old > 0 else 0

                # Viral flag: views per day > 100,000
                is_viral = 1 if views_per_day > 100000 else 0

                video_info = {
                    "video_id"          : vid_id,
                    "title"             : title,
                    "channel_name"      : channel,
                    "published_at"      : pub_date,
                    "publish_year"      : pub_year,
                    "publish_month"     : pub_month,
                    "days_since_publish": days_old,
                    "duration"          : content.get("duration", ""),
                    "views"             : views,
                    "likes"             : likes,
                    "comments"          : comments,
                    "views_per_day"     : views_per_day,
                    "is_viral"          : is_viral,
                    "language_code"     : detected_lang,
                    "language_name"     : language_name,
                    "region_fetched"    : region,
                    "category_id"       : snippet.get("categoryId", ""),
                    "fetched_at"        : datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }

                all_videos.append(video_info)

            if (batch_num + 1) % 10 == 0:
                print(f"   ✅ Processed {min(end, total)}/{total} videos...")

            time.sleep(0.2)

        except Exception as e:
            continue

    return all_videos


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    print("=" * 65)
    print("   🎵 VIRAL MUSIC TREND PREDICTION")
    print("   STEP 1: MAXIMUM DATA COLLECTION")
    print("   All Languages | All Regions | Viral Detection")
    print("=" * 65)

    if API_KEY == "YOUR_API_KEY_HERE":
        print("\n❌ ERROR: Paste your YouTube API Key on line 16!")
        return

    all_ids    = []
    region_map = {}
    quota_done = False

    # ===========================================================
    # PART 1: MOST POPULAR CHART - ALL 36 REGIONS
    # ===========================================================
    print("\n" + "=" * 65)
    print("PART 1: MOST POPULAR MUSIC - 36 REGIONS")
    print("=" * 65)

    for i, (region, lang) in enumerate(REGIONS.items()):
        print(f"   [{i+1:02d}/{len(REGIONS)}] Region: {region} ({lang}) ...", end=" ")

        ids = get_video_ids_by_region(API_KEY, region)

        for vid_id in ids:
            if vid_id not in region_map:
                region_map[vid_id] = region

        all_ids.extend(ids)
        print(f"{len(ids)} videos | Total: {len(set(all_ids))}")
        time.sleep(0.3)

    print(f"\n✅ Part 1 Done | Unique IDs: {len(set(all_ids))}")

    # ===========================================================
    # PART 2: SEARCH BY KEYWORDS - ALL LANGUAGES
    # ===========================================================
    print("\n" + "=" * 65)
    print("PART 2: KEYWORD SEARCH - ALL LANGUAGES")
    print("=" * 65)

    for i, query in enumerate(SEARCH_QUERIES):
        print(f"   [{i+1:02d}/{len(SEARCH_QUERIES)}] '{query}' ...", end=" ")

        result = search_videos_by_query(API_KEY, query)

        if result is None:
            print("⚠️  Quota exceeded! Stopping search.")
            quota_done = True
            break

        for vid_id in result:
            if vid_id not in region_map:
                region_map[vid_id] = "search"

        all_ids.extend(result)
        print(f"{len(result)} videos | Total: {len(set(all_ids))}")
        time.sleep(0.3)

    # ===========================================================
    # DEDUPLICATE ALL IDs
    # ===========================================================
    unique_ids = list(set(all_ids))
    print(f"\n📦 Total Unique Video IDs: {len(unique_ids)}")

    # ===========================================================
    # PART 3: FETCH FULL DETAILS
    # ===========================================================
    print("\n" + "=" * 65)
    print("PART 3: FETCHING FULL DETAILS")
    print("=" * 65)

    all_videos = get_video_details(API_KEY, unique_ids, region_map)

    # ===========================================================
    # SAVE TO CSV
    # ===========================================================
    print(f"\n💾 Saving to CSV...")
    df = pd.DataFrame(all_videos)
    df.drop_duplicates(subset=["video_id"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.to_csv(OUTPUT_FILE, index=False)

    # ===========================================================
    # FINAL SUMMARY
    # ===========================================================
    print("\n" + "=" * 65)
    print("📋 SAMPLE DATA (First 5 Rows):")
    print("=" * 65)
    print(df[["title", "channel_name", "language_name", "views", "likes", "is_viral"]].head().to_string())

    print("\n" + "=" * 65)
    print("📈 BASIC STATISTICS:")
    print("=" * 65)
    print(df[["views", "likes", "comments", "views_per_day"]].describe().to_string())

    print("\n" + "=" * 65)
    print("🌍 REGION DISTRIBUTION:")
    print("=" * 65)
    print(df["region_fetched"].value_counts().to_string())

    print("\n" + "=" * 65)
    print("🗣️  LANGUAGE DISTRIBUTION (Top 20):")
    print("=" * 65)
    print(df["language_name"].value_counts().head(20).to_string())

    print("\n" + "=" * 65)
    print("🔥 VIRAL SONGS DETECTED:")
    print("=" * 65)
    viral_df = df[df["is_viral"] == 1][["title", "channel_name", "language_name", "views", "views_per_day"]]
    viral_df = viral_df.sort_values("views_per_day", ascending=False)
    print(viral_df.head(20).to_string())

    print("\n" + "=" * 65)
    print("🔍 MISSING VALUES:")
    print("=" * 65)
    print(df.isnull().sum().to_string())

    print("\n" + "=" * 65)
    print("✅ STEP 1 COMPLETE!")
    print(f"   Total Records    : {len(df)}")
    print(f"   Viral Songs      : {df['is_viral'].sum()}")
    print(f"   Languages Found  : {df['language_name'].nunique()}")
    print(f"   Regions Covered  : {df['region_fetched'].nunique()}")
    print(f"   File Saved At    : {OUTPUT_FILE}")
    print("   ➡️  Next: python step2_data_cleaning.py")
    print("=" * 65)


if __name__ == "__main__":
    main()