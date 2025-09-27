import pandas as pd
import numpy as np
from scipy.stats import poisson
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
import requests
import os
import time
from datetime import datetime, timedelta
from rapidfuzz import fuzz
import logging
import sys
import joblib
from tqdm import tqdm
from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash
import json
from paystackapi.paystack import Paystack
import threading
import bcrypt
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.events import EVENT_JOB_EXECUTED, EVENT_JOB_ERROR
import pytz
from urllib.parse import quote_plus
from sqlalchemy.exc import OperationalError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# === New: Define RUN_MODE ===
RUN_MODE = os.getenv('RUN_MODE', 'web')  # 'web' for Flask, 'worker' for scheduler

# === Configure Logging ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('prediction_log.txt', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger()

# === API Configuration ===
API_KEY = os.getenv('SPORTS_API_KEY', '4ff6b5869f85aac30e4d39711a7079d4fb95bece286672f340aac81cce20ef1a')
API_BASE_URL = "https://apiv2.allsportsapi.com/football"
HEADERS = {'Content-Type': 'application/json'}
SEASON_ID = "2024-2025"

# === League Exclusion Rules ===
EXCLUDED_KEYWORDS = [
    'cup', 'copa', 
    'conference league', 'trophy', 'supercup', 'super cup', 'women', 'ladies',
    'female', 'fa cup', 'league cup', 'playoff', 'play-off', 'knockout',
    'u21', 'u19', 'u18', 'u17', 'youth', 'reserve', 'esiliiga', 'ekstraliga women'
]

# === Load Models and Scalers ===
logger.info("Loading models...")
start_time = time.time()
try:
    scaler_base = joblib.load("favour_v6_base_scaler.pkl")
    logistic_model = joblib.load("favour_v6_logistic_model.pkl")
    gb_model = joblib.load("favour_v6_gb_model.pkl")
    rf_model = joblib.load("favour_v6_rf_model.pkl")
    nb_model = joblib.load("favour_v6_nb_model.pkl")
    et_model = joblib.load("favour_v6_et_model.pkl")
    scaler_meta = joblib.load("favour_v6_meta_scaler.pkl")
    meta_model = joblib.load("hybrid_meta_model.pkl")
    logger.info(f"✅ Models and scalers loaded in {time.time() - start_time:.2f} seconds")
except FileNotFoundError as e:
    logger.error(f"❌ Error: Model or scaler file not found: {e}")
    sys.exit(1)
except Exception as e:
    logger.error(f"❌ Error loading models: {e}")
    sys.exit(1)

# === Flask App Configuration ===
app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY')
if not app.secret_key:
    logger.error("❌ FLASK_SECRET_KEY not set")
    sys.exit(1)

# Configure PostgreSQL with connection pooling
DATABASE_URL = os.getenv('DATABASE_URL')
if not DATABASE_URL:
    logger.error("❌ DATABASE_URL environment variable not set")
    sys.exit(1)
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
if '?' in DATABASE_URL:
    DATABASE_URL += '&sslmode=require'
else:
    DATABASE_URL += '?sslmode=require'
app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
    'pool_size': 5,
    'max_overflow': 10,
    'pool_timeout': 30,
    'pool_pre_ping': True,
}

app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=30)  # Keep sessions alive for 30 days
app.config['SESSION_COOKIE_SECURE'] = True  # Use with HTTPS
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'  # Prevent cookie issues on mobile
app.config['REMEMBER_COOKIE_DURATION'] = timedelta(days=30)  # Match session lifetime



# Initialize SQLAlchemy
db = SQLAlchemy(app)

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Paystack setup
PAYSTACK_PUBLIC_KEY = os.getenv('PAYSTACK_PUBLIC_KEY', 'pk_test_3ab2fd3709c83c56dd600042ed0ea8690271f6c5')
PAYSTACK_SECRET_KEY = os.getenv('PAYSTACK_SECRET_KEY')

# === Database Model ===
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)
    is_vip = db.Column(db.Boolean, default=False)
    vip_expiry = db.Column(db.DateTime, nullable=True)

    def set_password(self, password):
        self.password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

    def check_password(self, password):
        return bcrypt.checkpw(password.encode('utf-8'), self.password_hash.encode('utf-8'))

# === Routes ===
@app.route('/init_db', methods=['GET'])
def init_db():
    with app.app_context():
        db.create_all()
    logger.info("✅ Database tables created")
    return "Database tables created!"

@app.route('/ping_db')
def ping_db():
    try:
        with db.engine.connect() as conn:
            conn.execute("SELECT 1")
        logger.info("✅ Database ping successful")
        return "Database ping successful"
    except Exception as e:
        logger.error(f"❌ Database ping failed: {e}")
        return "Database ping failed", 500

# New: Scheduler status route for debugging
@app.route('/scheduler_status')
def scheduler_status():
    global scheduler
    return jsonify({"scheduler_running": scheduler.running if 'scheduler' in globals() else False})

@retry(retry=retry_if_exception_type(OperationalError), stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
@login_manager.user_loader
def load_user(user_id):
    try:
        return User.query.get(int(user_id))
    except OperationalError as e:
        logger.error(f"❌ Database error in load_user: {e}")
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error in load_user: {e}")
        return None

# === Retry Logic for API Calls ===
api_call_count = 0

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10), retry=retry_if_exception_type(requests.exceptions.RequestException))
def fetch_with_retry(url):
    global api_call_count
    api_call_count += 1
    logger.info(f"API Call #{api_call_count}: {url}")
    response = requests.get(url, headers=HEADERS, timeout=30)
    logger.info(f"Rate Limit Headers: {response.headers.get('X-Rate-Limit-Limit', 'N/A')}, "
                f"Remaining: {response.headers.get('X-Rate-Limit-Remaining', 'N/A')}, "
                f"Reset: {response.headers.get('X-Rate-Limit-Reset', 'N/A')}")
    if response.status_code == 429:
        retry_after = response.headers.get('Retry-After', 10)
        logger.warning(f"Rate limit hit, suggested wait: {retry_after} seconds")
    response.raise_for_status()
    return response

# === Confidence Function ===
def favour_v6_confidence(row, HomeGoalList, HomeConcededList, AwayGoalList, AwayConcededList, HomeBTTS, AwayBTTS, poisson_prob):
    try:
        HomeGoalList = list(map(int, HomeGoalList))
        HomeConcededList = list(map(int, HomeConcededList))
        AwayGoalList = list(map(int, AwayGoalList))
        AwayConcededList = list(map(int, AwayConcededList))
        HomeBTTS = list(map(int, HomeBTTS))
        AwayBTTS = list(map(int, AwayBTTS))
    except ValueError as e:
        logger.warning(f"⚠️ Invalid input data format: {e}")
        return 0.0, 100.0, []

    required_cols = ['avg_home_scored', 'avg_away_scored', 'avg_home_conceded', 'avg_away_conceded']
    if not all(col in row for col in required_cols):
        missing_cols = [col for col in required_cols if col not in row]
        logger.warning(f"⚠️ Missing required columns in row: {missing_cols}")
        return 0.0, 100.0, []

    base_score = poisson_prob * 100
    triggered_rules = ["Poisson: Base score set to Poisson model probability"]

    scored_sum = row['avg_home_scored'] + row['avg_away_scored']
    conceded_sum = row['avg_home_conceded'] + row['avg_away_conceded']
    division_result = 1.0 if scored_sum == conceded_sum else max(scored_sum, conceded_sum) / min(scored_sum, conceded_sum) if min(scored_sum, conceded_sum) != 0 else float('inf')

    zero_count = sum(1 for g in HomeGoalList + AwayGoalList + HomeConcededList + AwayConcededList if g == 0)
    avg_conceded = (row['avg_home_conceded'] + row['avg_away_conceded']) / 2
    high_goal_count = sum(1 for g in HomeGoalList + AwayGoalList + HomeConcededList + AwayConcededList if g >= 2)

    if avg_conceded >= 1.8 and high_goal_count >= 10:
        base_score += 20
        triggered_rules.append("Rule 1: +20 to base_score (avg conceded >= 1.8 and high goal/conceded count >= 10)")
    if avg_conceded >= 1.5 and high_goal_count >= 8:
        base_score += 10
        triggered_rules.append("Rule 4: +10 to base_score (avg conceded >= 1.5 and high goal/conceded count >= 8)")
    if high_goal_count <= 9 and zero_count <= 6:  # Aligned with prediction1
        base_score -= 25
        triggered_rules.append("Rule 2: -25 to base_score (high goal/conceded count <= 9 and zero count <= 6)")
    if high_goal_count >= 8 and zero_count >= 7:
        base_score += 15
        triggered_rules.append("Rule 3: +15 to base_score (high goal/conceded count >= 8 and zero count >= 7)")
    if division_result >= 1.2 and zero_count in [6, 7, 8]:
        base_score += 15
        triggered_rules.append("Rule 7: +15 to base_score (division result >= 1.2 and zero count in [6, 7, 8])")

    base_score = max(0, min(base_score, 100))
    over_conf = max(0, min(base_score, 90))
    under_conf = max(0, min(100 - base_score, 90))

    return over_conf, under_conf, triggered_rules

# === Fuzzy Matching for Team Names ===
def is_team_match(api_team_name, expected_team_name, threshold=75):
    score = fuzz.token_set_ratio(api_team_name.lower(), expected_team_name.lower())
    logger.debug(f"is_team_match: Comparing '{api_team_name}' vs '{expected_team_name}', Score: {score}, Threshold: {threshold}")
    return score >= threshold

# === Fetch Match Data ===
def fetch_match_data(home_team_key, away_team_key, season_id, league_id, match_id, match_date, home_team_name, away_team_name):
    global api_call_count
    match_info = f"{home_team_name} vs {away_team_name} ({match_date})"
    logger.info(f"Processing match: {match_info}, LeagueID: {league_id}, MatchID: {match_id}")

    to_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    from_date = '2024-08-01'

    try:
        datetime.strptime(match_date, '%Y-%m-%d')
    except (ValueError, TypeError):
        logger.error(f"❌ Invalid match date format for {match_info}: {match_date}")
        return None

    HomeGoalList, HomeConcededList, HomeBTTS = [], [], []
    home_red_cards = []
    try:
        recent_url = f"{API_BASE_URL}?met=Fixtures&teamId={home_team_key}&leagueId={league_id}&APIkey={API_KEY}&season={season_id}&from={from_date}&to={to_date}&limit=20"
        response = fetch_with_retry(recent_url)
        response.encoding = 'utf-8'
        home_team_response = response.json()
        logger.debug(f"Home team {home_team_key} matches API response: {home_team_response}")
        if not home_team_response.get('success') == 1:
            raise ValueError(f"Invalid response for home team {home_team_key}")
        home_matches = home_team_response.get('result', [])
        if not isinstance(home_matches, list):
            raise ValueError(f"Incomplete response for home team {home_team_key}")
    except Exception as e:
        logger.error(f"❌ Error fetching home team {home_team_key} matches: {e}")
        return None

    home_filtered = []
    for match in sorted(home_matches, key=lambda x: x.get('event_date', '9999-12-31'), reverse=True):
        if match.get('event_key') == match_id:
            continue
        if match.get('event_status') != 'Finished':
            continue
        if str(match.get('league_key', '')) != str(league_id):
            continue
        if is_team_match(match.get('event_home_team', ''), home_team_name):
            result = match.get('event_final_result', '')
            if result and '-' in result and len(result.split('-')) == 2:
                try:
                    parts = result.replace(' ', '').split('-')
                    home_goals, away_goals = map(int, parts[:2])
                    cards = match.get('cards', [])
                    match_red_cards = [
                        {'time': card.get('time', '0'), 'card': card.get('card')}
                        for card in cards if card.get('card') == 'red card'
                    ]
                    home_filtered.append({'match': match, 'red_cards': match_red_cards})
                    if len(home_filtered) == 5:
                        break
                except (ValueError, TypeError):
                    continue
            else:
                logger.warning(f"⚠️ Invalid result format for home match {match.get('event_date')}: {result}")
                continue

    if len(home_filtered) != 5:
        logger.error(f"❌ Skipping {match_info}: Only {len(home_filtered)} home matches found")
        return None

    for item in home_filtered:
        match = item['match']
        match_red_cards = item['red_cards']
        result = match.get('event_final_result', '')
        home_goals, away_goals = (map(int, result.replace(' ', '').split('-')[:2]) 
                                  if result and '-' in result else (0, 0))
        HomeGoalList.append(home_goals)
        HomeConcededList.append(away_goals)
        HomeBTTS.append(1 if home_goals > 0 and away_goals > 0 else 0)
        home_red_cards.extend(match_red_cards)

    logger.debug(f"Home data: Goals={HomeGoalList}, Conceded={HomeConcededList}, BTTS={HomeBTTS}, RedCards={len(home_red_cards)}")

    AwayGoalList, AwayConcededList, AwayBTTS = [], [], []
    away_red_cards = []
    try:
        recent_url = f"{API_BASE_URL}?met=Fixtures&teamId={away_team_key}&leagueId={league_id}&APIkey={API_KEY}&season={season_id}&from={from_date}&to={to_date}&limit=20"
        response = fetch_with_retry(recent_url)
        response.encoding = 'utf-8'
        away_team_response = response.json()
        logger.debug(f"Away team {away_team_key} matches API response: {away_team_response}")
        if not away_team_response.get('success') == 1:
            raise ValueError(f"Invalid response for away team {away_team_key}")
        away_matches = away_team_response.get('result', [])
        if not isinstance(away_matches, list):
            raise ValueError(f"Incomplete response for away team {away_team_key}")
    except Exception as e:
        logger.error(f"❌ Error fetching away team {away_team_key} matches: {e}")
        return None

    away_filtered = []
    for match in sorted(away_matches, key=lambda x: x.get('event_date', '9999-12-31'), reverse=True):
        if match.get('event_key') == match_id:
            continue
        if match.get('event_status') != 'Finished':
            continue
        if str(match.get('league_key', '')) != str(league_id):
            continue
        if is_team_match(match.get('event_away_team', ''), away_team_name):
            result = match.get('event_final_result', '')
            if result and '-' in result and len(result.split('-')) == 2:
                try:
                    parts = result.replace(' ', '').split('-')
                    home_goals, away_goals = map(int, parts[:2])
                    cards = match.get('cards', [])
                    match_red_cards = [
                        {'time': card.get('time', '0'), 'card': card.get('card')}
                        for card in cards if card.get('card') == 'red card'
                    ]
                    away_filtered.append({'match': match, 'red_cards': match_red_cards})
                    if len(away_filtered) == 5:
                        break
                except (ValueError, TypeError):
                    continue
            else:
                logger.warning(f"⚠️ Invalid result format for away match {match.get('event_date')}: {result}")
                continue

    if len(away_filtered) != 5:
        logger.error(f"❌ Skipping {match_info}: Only {len(away_filtered)} away matches found")
        return None

    for item in away_filtered:
        match = item['match']
        match_red_cards = item['red_cards']
        result = match.get('event_final_result', '')
        home_goals, away_goals = (map(int, result.replace(' ', '').split('-')[:2]) 
                                  if result and '-' in result else (0, 0))
        AwayGoalList.append(away_goals)
        AwayConcededList.append(home_goals)
        AwayBTTS.append(1 if home_goals > 0 and away_goals > 0 else 0)
        away_red_cards.extend(match_red_cards)

    logger.debug(f"Away data: Goals={AwayGoalList}, Conceded={AwayConcededList}, BTTS={AwayBTTS}, RedCards={len(away_red_cards)}")

    total_red_cards = len(home_red_cards) + len(away_red_cards)
    if total_red_cards > 2:
        logger.error(f"❌ Skipping {match_info}: Total red cards ({total_red_cards}) exceeds 2")
        return None
    elif total_red_cards == 2:
        for item in home_filtered + away_filtered:
            match = item['match']
            match_red_cards = item['red_cards']
            if len(match_red_cards) == 2:
                red_card_times = []
                for card in match_red_cards:
                    try:
                        time_str = card.get('time', '0')
                        minute = int(time_str.split('+')[0]) if '+' in time_str else int(time_str)
                        red_card_times.append(minute)
                    except (ValueError, TypeError):
                        logger.warning(f"⚠️ Invalid red card time in match {match.get('event_date')}: {time_str}")
                        return None
                red_card_times.sort()
                if len(red_card_times) != 2 or red_card_times[1] <= 75:
                    logger.error(f"❌ Skipping {match_info}: Second red card at {red_card_times[1]} minutes")
                    return None
                result = match.get('event_final_result', '')
                if result and '-' in result:
                    try:
                        home_goals, away_goals = map(int, result.replace(' ', '').split('-')[:2])
                        if home_goals != 0 or away_goals != 0:
                            logger.error(f"❌ Skipping {match_info}: Match with 2 red cards has goals ({result})")
                            return None
                    except (ValueError, TypeError):
                        logger.warning(f"⚠️ Invalid result format in match {match.get('event_date')}: {result}")
                        return None

    return {
        'TotalHomeGoals': sum(HomeGoalList),
        'TotalHomeConceded': sum(HomeConcededList),
        'TotalAwayGoals': sum(AwayGoalList),
        'TotalAwayConceded': sum(AwayConcededList),
        'HomeGoalList': HomeGoalList,
        'HomeConcededList': HomeConcededList,
        'HomeBTTS': HomeBTTS,
        'AwayGoalList': AwayGoalList,
        'AwayConcededList': AwayConcededList,
        'AwayBTTS': AwayBTTS
    }

# === Prediction Logic ===
def make_prediction(data_dict, match_info):
    required_length = 5
    lists = [
        data_dict['HomeGoalList'], data_dict['HomeConcededList'],
        data_dict['AwayGoalList'], data_dict['AwayConcededList'],
        data_dict['HomeBTTS'], data_dict['AwayBTTS']
    ]
    if any(len(lst) != required_length for lst in lists):
        logger.error(f"❌ Skipping {match_info}: Lists have incorrect lengths: {[len(lst) for lst in lists]}")
        return {"error": "Incorrect list lengths"}

    avg_home_scored = data_dict['TotalHomeGoals'] / required_length
    avg_home_conceded = data_dict['TotalHomeConceded'] / required_length
    avg_away_scored = data_dict['TotalAwayGoals'] / required_length
    avg_away_conceded = data_dict['TotalAwayConceded'] / required_length
    btts_count = sum(data_dict['HomeBTTS']) + sum(data_dict['AwayBTTS'])
    high_scoring_matches = sum(1 for g in data_dict['HomeGoalList'] + data_dict['AwayGoalList'] if g >= 2)
    low_conceded_count = sum(1 for c in data_dict['HomeConcededList'] + data_dict['AwayConcededList'] if c <= 1)
    heavy_conceding_boost = int(avg_home_conceded >= 2.0 or avg_away_conceded >= 2.0)
    moderate_conceding_boost = int(1.5 <= avg_home_conceded < 2.0 or 1.5 <= avg_away_conceded < 2.0)
    btts_boost_flag = 2 if btts_count >= 8 else 1 if btts_count >= 6 else 0
    many_0_1_conceded_flag = int(low_conceded_count >= 6)
    defensive_strength_flag = int(avg_home_conceded <= 0.8 or avg_away_conceded <= 0.8)
    avoid_match_penalty_flag = int(high_scoring_matches <= 2 and btts_count <= 4)
    low_conceded_boost = int(low_conceded_count >= 5)
    defensive_threshold_flag = int((avg_home_conceded + avg_away_conceded) / 2 <= 1.0)
    home_goals_list_avg = np.mean(data_dict['HomeGoalList'])
    home_conceded_list_avg = np.mean(data_dict['HomeConcededList'])
    away_goals_list_avg = np.mean(data_dict['AwayGoalList'])
    away_conceded_list_avg = np.mean(data_dict['AwayConcededList'])

    data = pd.DataFrame([{
        'avg_home_scored': avg_home_scored,
        'avg_home_conceded': avg_home_conceded,
        'avg_away_scored': avg_away_scored,
        'avg_away_conceded': avg_away_conceded,
        'btts_count': btts_count,
        'high_scoring_matches': high_scoring_matches,
        'low_conceded_count': low_conceded_count,
        'heavy_conceding_boost': heavy_conceding_boost,
        'moderate_conceding_boost': moderate_conceding_boost,
        'btts_boost_flag': btts_boost_flag,
        'many_0_1_conceded_flag': many_0_1_conceded_flag,
        'defensive_strength_flag': defensive_strength_flag,
        'avoid_match_penalty_flag': avoid_match_penalty_flag,
        'low_conceded_boost': low_conceded_boost,
        'defensive_threshold_flag': defensive_threshold_flag,
        'home_goals_list_avg': home_goals_list_avg,
        'home_conceded_list_avg': home_conceded_list_avg,
        'away_goals_list_avg': away_goals_list_avg,
        'away_conceded_list_avg': away_conceded_list_avg
    }])

    feature_columns = [
        'avg_home_scored', 'avg_home_conceded', 'avg_away_scored', 'avg_away_conceded',
        'btts_count', 'high_scoring_matches', 'low_conceded_count',
        'heavy_conceding_boost', 'moderate_conceding_boost',
        'btts_boost_flag', 'many_0_1_conceded_flag',
        'defensive_strength_flag', 'avoid_match_penalty_flag',
        'low_conceded_boost', 'defensive_threshold_flag',
        'home_goals_list_avg', 'home_conceded_list_avg', 'away_goals_list_avg', 'away_conceded_list_avg'
    ]

    try:
        data_scaled = pd.DataFrame(
            scaler_base.transform(data[feature_columns]),
            columns=feature_columns,
            index=data.index
        )
    except Exception as e:
        logger.error(f"❌ Base scaler error for {match_info}: {e}")
        return {"error": f"Base scaler error: {e}"}

    # Poisson Model (aligned with prediction1)
    league_avg_goals = (data_dict['TotalHomeGoals'] + data_dict['TotalHomeConceded'] +
                        data_dict['TotalAwayGoals'] + data_dict['TotalAwayConceded']) / 10
    if league_avg_goals == 0:
        league_avg_goals = 1.0
    lambda_h = max(0.5, avg_home_scored * avg_away_conceded / league_avg_goals)
    lambda_a = max(0.5, avg_away_scored * avg_home_conceded / league_avg_goals)
    poisson_prob = 0
    for h in range(6):
        for a in range(6):
            prob_h = poisson.pmf(h, lambda_h)
            prob_a = poisson.pmf(a, lambda_a)
            if h + a >= 2:
                poisson_prob += prob_h * prob_a

    try:
        logistic_prob = logistic_model.predict_proba(data_scaled)[0, 1]
        gb_prob = gb_model.predict_proba(data_scaled)[0, 1]
        rf_prob = rf_model.predict_proba(data_scaled)[0, 1]
        nb_prob = nb_model.predict_proba(data_scaled)[0, 1]
        et_prob = et_model.predict_proba(data_scaled)[0, 1]
    except Exception as e:
        logger.error(f"❌ Prediction error for {match_info}: {e}")
        return {"error": f"Prediction error: {e}"}

    data['LogisticProb'] = logistic_prob
    data['GBProb'] = gb_prob
    data['RFProb'] = rf_prob
    data['NBProb'] = nb_prob
    data['ETProb'] = et_prob
    data['PoissonProb'] = poisson_prob

    over_conf, under_conf, triggered_rules = favour_v6_confidence(
        data.iloc[0], data_dict['HomeGoalList'], data_dict['HomeConcededList'],
        data_dict['AwayGoalList'], data_dict['AwayConcededList'],
        data_dict['HomeBTTS'], data_dict['AwayBTTS'], poisson_prob
    )

    meta_feature_columns = [
        'RuleOverConfidence', 'RuleUnderConfidence', 'LogisticProb', 'GBProb',
        'RFProb', 'NBProb', 'ETProb', 'PoissonProb'
    ]
    meta_data = pd.DataFrame([{
        'RuleOverConfidence': over_conf,
        'RuleUnderConfidence': under_conf,
        'LogisticProb': logistic_prob,
        'GBProb': gb_prob,
        'RFProb': rf_prob,
        'NBProb': nb_prob,
        'ETProb': et_prob,
        'PoissonProb': poisson_prob
    }], columns=meta_feature_columns)

    try:
        meta_data_scaled = pd.DataFrame(
            scaler_meta.transform(meta_data[meta_feature_columns]),
            columns=meta_feature_columns,
            index=meta_data.index
        )
        meta_probs = meta_model.predict_proba(meta_data_scaled)[0]
    except Exception as e:
        logger.error(f"❌ Meta-model prediction error for {match_info}: {e}")
        return {"error": f"Meta-model prediction error: {e}"}

    zero_count = sum(1 for g in data_dict['HomeGoalList'] + data_dict['AwayGoalList'] +
                     data_dict['HomeConcededList'] + data_dict['AwayConcededList'] if g == 0)
    recommendation = "NO BET"
    reason = ""
    meta_over_prob = meta_probs[1] * 100
    meta_under_prob = meta_probs[0] * 100

    if zero_count in [6, 7, 8] and meta_probs[0] > meta_probs[1]:
        recommendation = "NO BET"
        reason = f"Match rejected: {zero_count} zeros in goal/conceded lists and meta-model favors Under 3.5 ({meta_under_prob:.1f}% vs Over 1.5 {meta_over_prob:.1f}%)."
    elif meta_over_prob >= 75:
        recommendation = "Over 1.5"
        reason = f"Meta-Model Over 1.5 Probability ({meta_over_prob:.1f}%) exceeds 75% threshold."
    elif meta_under_prob >= 75:
        recommendation = "Under 3.5"
        reason = f"Meta-Model Under 3.5 Probability ({meta_under_prob:.1f}%) exceeds 75% threshold."
    else:
        reason = f"Neither Meta-Model Over 1.5 Probability ({meta_over_prob:.1f}%) nor Under 3.5 Probability ({meta_under_prob:.1f}%) exceeds 75% threshold. No bet recommended."

    return {
        'Match': match_info,
        'OverConfidence': over_conf,
        'UnderConfidence': under_conf,
        'MetaOverProb': meta_over_prob,
        'MetaUnderProb': meta_under_prob,
        'Recommendation': recommendation,
        'Reason': reason,
        'TriggeredRules': triggered_rules
    }

# === Filter Leagues ===
def filter_leagues(leagues):
    filtered = []
    for league in leagues:
        league_name = league['league_name'].lower()
        if any(keyword.lower() in league_name for keyword in EXCLUDED_KEYWORDS):
            logger.debug(f"Excluding league: {league['league_name']} (ID: {league['league_key']})")
            continue
        filtered.append(league)
    logger.info(f"Filtered {len(leagues)} leagues to {len(filtered)} after excluding cups, women's, and youth leagues")
    return filtered

# === Fetch All Leagues ===
def fetch_all_leagues():
    global api_call_count
    cache_file = 'leagues_cache.json'
    if os.path.exists(cache_file):
        with open(cache_file, 'r', encoding='utf-8') as f:
            leagues = json.load(f)
            logger.info("Loaded leagues from cache")
            return [(league['league_key'], league['league_name'], league.get('country_name', 'Unknown')) for league in leagues]
    logger.info("Fetching all leagues...")
    try:
        url = f"{API_BASE_URL}?met=Leagues&APIkey={API_KEY}"
        response = fetch_with_retry(url)
        response.encoding = 'utf-8'
        data = response.json()
        if data.get("success") != 1:
            logger.error(f"❌ API error fetching leagues: {data}")
            return []
        leagues = filter_leagues(data["result"])
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(leagues, f)
        logger.info(f"✅ Retrieved and cached {len(leagues)} eligible leagues")
        with open('leagues.txt', 'w', encoding='utf-8') as f:
            f.write("Filtered League List (Excluding Cups, Women's, and Youth Leagues):\n")
            for league in leagues:
                f.write(f"ID: {league['league_key']}, Name: {league['league_name']}, Country: {league.get('country_name', 'Unknown')}\n")
        return [(league['league_key'], league['league_name'], league.get('country_name', 'Unknown')) for league in leagues]
    except Exception as e:
        logger.error(f"❌ Error fetching leagues: {e}")
        return []

# === Fetch Upcoming Matches for a League ===
def fetch_upcoming_matches(league_id, league_name, country_name, season_id, date_from):
    global api_call_count
    logger.info(f"Fetching matches for {league_name} ({country_name}, ID: {league_id}) on {date_from}")
    try:
        url = f"{API_BASE_URL}?met=Fixtures&leagueId={league_id}&APIkey={API_KEY}&season={season_id}&from={date_from}&to={date_from}"
        response = fetch_with_retry(url)
        response.encoding = 'utf-8'
        data = response.json()
        if data.get("success") != 1 or not data.get("result"):
            logger.warning(f"⚠️ No matches found for {league_name} on {date_from}")
            return []
        matches = data["result"]
        for match in matches:
            match['league_name'] = league_name
            match['country_name'] = country_name
        return matches
    except Exception as e:
        logger.error(f"❌ Error fetching matches for {league_name}: {e}")
        return []

# === Modified Main Function ===
def main(date_from=None):
    global api_call_count
    api_call_count = 0
    wat_tz = pytz.timezone('Africa/Lagos')
    if date_from is None:
        date_from = datetime.now(wat_tz).strftime('%Y-%m-%d')
        logger.info(f"No date provided, defaulting to today: {date_from}")

    season_id = SEASON_ID
    logger.info(f"Using Season ID: {season_id} for date: {date_from}")

    target_league_ids = [
        211, 156, 155, 250, 244, 245, 251, 223, 329, 330, 653, 7097, 171, 175, 152, 302, 207, 168,
        308, 118, 253, 141, 593, 614, 352, 353, 362, 331, 329
    ]

    leagues = fetch_all_leagues()
    if not leagues:
        logger.error("❌ Aborting: No eligible leagues retrieved.")
        return

    leagues = [(league_id, league_name, country_name) for league_id, league_name, country_name in leagues
               if league_id in target_league_ids]

    if not leagues:
        logger.error("❌ Aborting: No matching leagues found for provided IDs after filtering.")
        return

    all_matches = []
    logger.info(f"\nFetching matches for {date_from} for {len(leagues)} selected leagues...")
    for league_id, league_name, country_name in tqdm(leagues, desc="Processing leagues"):
        matches = fetch_upcoming_matches(league_id, league_name, country_name, season_id, date_from)
        all_matches.extend(matches)
        time.sleep(0.5)

    if not all_matches:
        logger.error(f"❌ No matches found for selected leagues on {date_from}.")
        return

    logger.info(f"\nFound {len(all_matches)} matches for {date_from}:")
    for match in all_matches:
        match_info = f"{match.get('event_home_team', 'Unknown')} vs {match.get('event_away_team', 'Unknown')} ({match['league_name']}, {match['country_name']})"
        logger.info(f"- {match_info} (Match ID: {match.get('event_key')})")

    results = []
    skipped_matches = []
    logger.info("\nPredicting outcomes...")
    for match in tqdm(all_matches, desc="Predicting matches"):
        match_id = match.get('event_key')
        home_team_key = match.get('home_team_key')
        away_team_key = match.get('away_team_key')
        home_team_name = match.get('event_home_team', 'Unknown')
        away_team_name = match.get('event_away_team', 'Unknown')
        match_date = match.get('event_date')
        league_id = match.get('league_key')
        league_name = match['league_name']
        country_name = match['country_name']
        match_info = f"{home_team_name} vs {away_team_name} ({match_date}, {league_name}, {country_name})"

        if not home_team_key or not away_team_key:
            logger.error(f"❌ Skipping {match_info}: Missing team key(s)")
            skipped_matches.append(match_info)
            continue

        data_dict = fetch_match_data(home_team_key, away_team_key, season_id, league_id, match_id, match_date, home_team_name, away_team_name)
        if data_dict is None:
            skipped_matches.append(match_info)
            continue

        result = make_prediction(data_dict, match_info)
        if 'error' in result:
            skipped_matches.append(match_info)
            continue
        results.append(result)
        time.sleep(0.5)

    logger.info(f"Total API Calls Made: {api_call_count}")
    predictions_data = [
        {
            'Match': r['Match'],
            'MetaOverProb': r['MetaOverProb'],
            'MetaUnderProb': r['MetaUnderProb'],
            'Recommendation': r['Recommendation'],
            'OverConfidence': r['OverConfidence'],
            'UnderConfidence': r['UnderConfidence'],
            'Reason': r['Reason'],
            'TriggeredRules': r['TriggeredRules']
        } for r in results
    ]
    with open('predictions.json', 'w', encoding='utf-8') as f:
        json.dump(predictions_data, f)

    logger.info("\n=== Prediction Results ===")
    with open('predictions.txt', 'w', encoding='utf-8') as f:
        if not results:
            msg = f"❌ No valid predictions for {date_from}. Check prediction_log.txt."
            logger.error(msg)
            f.write(msg + "\n")
        for result in results:
            output = (f"\nMatch: {result['Match']}\n"
                      f"Over 1.5 Confidence: {result['OverConfidence']:.1f}%\n"
                      f"Under 3.5 Confidence: {result['UnderConfidence']:.1f}%\n"
                      f"Meta-Model Over 1.5 Probability: {result['MetaOverProb']:.1f}%\n"
                      f"Meta-Model Under 3.5 Probability: {result['MetaUnderProb']:.1f}%\n"
                      f"Recommendation: {result['Recommendation']}\n"
                      f"Reason: {result['Reason']}\n"
                      f"Triggered Rules:\n" + "\n".join(result['TriggeredRules']) + "\n" + f"{'='*50}")
            logger.info(output)
            f.write(output + "\n")
        if skipped_matches:
            logger.info("\n=== Skipped Matches ===")
            f.write("\nSkipped Matches:\n" + "\n".join(skipped_matches) + "\n")

# === Load Predictions ===
def load_predictions():
    try:
        with open('predictions.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"❌ Error loading predictions.json: {e}")
        return []

# === Schedule Predictions ===
scheduler = None  # Global for status route
def schedule_predictions():
    if RUN_MODE != 'worker':
        logger.info("Skipping scheduler initialization in web mode")
        return None

    wat_tz = pytz.timezone('Africa/Lagos')
    scheduler = BackgroundScheduler(timezone=wat_tz)

    def job_wrapper():
        try:
            logger.info(f"Starting scheduled prediction job at {datetime.now(wat_tz)}")
            main(date_from=datetime.now(wat_tz).strftime('%Y-%m-%d'))
            logger.info("Scheduled prediction job completed")
        except Exception as e:
            logger.error(f"Scheduled job failed: {e}")

    scheduler.add_job(
        job_wrapper,
        'cron',
        hour=0,
        minute=15,
        id='daily_prediction_job',
        replace_existing=True,
        misfire_grace_time=3600,  # 1 hour grace period
        timezone=wat_tz
    )
    scheduler.add_listener(
        lambda event: logger.info(f"Job {event.job_id} executed") if not event.exception else logger.error(f"Job {event.job_id} failed: {event.exception}"),
        EVENT_JOB_EXECUTED | EVENT_JOB_ERROR
    )
    logger.info("Scheduler started for 12:15 AM WAT daily predictions")
    scheduler.start()
    return scheduler

# === Routes ===
@app.route('/')
def home():
    try:
        if not os.path.exists('predictions.json'):
            wat_tz = pytz.timezone('Africa/Lagos')
            logger.info("No predictions.json found, running predictions...")
            main(date_from=datetime.now(wat_tz).strftime('%Y-%m-%d'))
        predictions = load_predictions()
        if not predictions:
            return render_template('home.html', predictions=[], error="No matches available for today.", user=current_user)
        free_preds = []
        for pred in predictions:
            pick = "Over 1.5" if pred['MetaOverProb'] > pred['MetaUnderProb'] else "Under 3.5"
            free_preds.append({
                'match': pred['Match'],
                'pick': pick
            })
        return render_template('home.html', predictions=free_preds, error=None, user=current_user)
    except Exception as e:
        logger.error(f"❌ Error rendering home: {e}")
        flash('Server error. Please try again.', 'danger')
        return render_template('home.html', predictions=[], error="Server error occurred.", user=current_user)

@app.route('/vip')
@login_required
def vip():
    try:
        if not current_user.is_vip or (current_user.vip_expiry and current_user.vip_expiry < datetime.utcnow()):
            current_user.is_vip = False
            current_user.vip_expiry = None
            db.session.commit()
            flash('Your VIP subscription has expired. Please renew.', 'warning')
            return redirect(url_for('pay'))
        predictions = load_predictions()
        vip_preds = [p for p in predictions if p['Recommendation'] != "NO BET"]
        return render_template('vip.html', predictions=vip_preds, user=current_user)
    except OperationalError as e:
        logger.error(f"❌ Database error in vip route: {e}")
        flash('Database connection issue. Please try again.', 'danger')
        return redirect(url_for('home'))
    except Exception as e:
        logger.error(f"❌ Unexpected error in vip route: {e}")
        flash('Server error. Please try again.', 'danger')
        return redirect(url_for('home'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if not username or not password:
            flash('Username and password are required.', 'danger')
            return render_template('register.html')
        try:
            if User.query.filter_by(username=username).first():
                flash('Username already exists.', 'danger')
                return render_template('register.html')
            user = User(username=username)
            user.set_password(password)
            db.session.add(user)
            db.session.commit()
            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('login'))
        except OperationalError as e:
            logger.error(f"❌ Database error during register: {e}")
            flash('Database connection issue. Please try again.', 'danger')
            return render_template('register.html')
        except Exception as e:
            logger.error(f"❌ Unexpected error during register: {e}")
            flash('Server error. Please try again.', 'danger')
            return render_template('register.html')
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if not username or not password:
            flash('Username and password are required.', 'danger')
            return render_template('login.html')
        try:
            user = User.query.filter_by(username=username).first()
            if user and user.check_password(password):
                login_user(user)
                flash('Logged in successfully!', 'success')
                return redirect(url_for('home'))
            flash('Invalid username or password.', 'danger')
        except OperationalError as e:
            logger.error(f"❌ Database error during login: {e}")
            flash('Database connection issue. Please try again.', 'danger')
            return render_template('login.html')
        except Exception as e:
            logger.error(f"❌ Unexpected error during login: {e}")
            flash('Server error. Please try again.', 'danger')
            return render_template('login.html')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    try:
        logout_user()
        flash('Logged out successfully.', 'success')
        return redirect(url_for('home'))
    except OperationalError as e:
        logger.error(f"❌ Database error during logout: {e}")
        flash('Database connection issue. Please try again.', 'danger')
        return redirect(url_for('home'))
    except Exception as e:
        logger.error(f"❌ Unexpected error during logout: {e}")
        flash('Server error. Please try again.', 'danger')
        return redirect(url_for('home'))

@app.route('/pay')
@login_required
def pay():
    try:
        return render_template('pay.html', paystack_key=PAYSTACK_PUBLIC_KEY, user=current_user)
    except Exception as e:
        logger.error(f"❌ Error rendering pay page: {e}")
        flash('Server error. Please try again.', 'danger')
        return redirect(url_for('home'))

@app.route('/paystack/callback')
@login_required
def paystack_callback():
    ref = request.args.get('reference')
    if not ref:
        logger.error("❌ No payment reference provided")
        flash('Payment failed: No reference provided.', 'danger')
        return redirect(url_for('pay'))
    try:
        ps = Paystack(PAYSTACK_SECRET_KEY)
        data = ps.transaction.verify(ref)
        logger.info(f"Paystack verification response: {data}")
        if data.get('status') and data.get('data', {}).get('status') == 'success':
            current_user.is_vip = True
            current_user.vip_expiry = datetime.utcnow() + timedelta(days=7)
            db.session.commit()
            logger.info(f"✅ Payment verified for ref: {ref}, VIP granted to {current_user.username} until {current_user.vip_expiry}")
            flash('VIP subscription activated for 7 days!', 'success')
            return redirect(url_for('vip'))
        else:
            logger.error(f"❌ Payment verification failed for ref: {ref}, response: {data}")
            flash('Payment verification failed.', 'danger')
            return redirect(url_for('pay'))
    except OperationalError as e:
        logger.error(f"❌ Database error during paystack callback: {e}")
        flash('Database connection issue. Please try again.', 'danger')
        return redirect(url_for('pay'))
    except Exception as e:
        logger.error(f"❌ Error verifying payment: {e}")
        flash('Payment issue – try again.', 'danger')
        return redirect(url_for('pay'))

# === CLI Command for Manual Testing ===
@app.cli.command("run-predictions")
def run_predictions():
    """Run predictions manually or via cron."""
    wat_tz = pytz.timezone('Africa/Lagos')
    logger.info("Running predictions via CLI command")
    main(date_from=datetime.now(wat_tz).strftime('%Y-%m-%d'))

# === Main Entry Point ===
if __name__ == '__main__':
    if RUN_MODE == 'worker':
        logger.info("Starting in worker mode")
        scheduler = schedule_predictions()
        try:
            while True:
                time.sleep(60)  # Keep worker alive
        except KeyboardInterrupt:
            if scheduler:
                scheduler.shutdown()
                logger.info("Worker shutdown")
    else:
        logger.info("Starting in web mode")
        app.run(debug=False)
