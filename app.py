import asyncio
import aiohttp
from aiohttp import ClientSession
from cachetools import TTLCache
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
from sqlalchemy.exc import OperationalError
from sqlalchemy.sql import text
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
import aiohttp_retry

# Load environment variables
load_dotenv()

# Configure Logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('prediction_log.txt', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger()
logger.info("Starting Flask application...")

# API Configuration
API_KEY = os.getenv('SPORTS_API_KEY', 'your_api_key_here')  # Replace with your actual key
API_BASE_URL = "https://apiv2.allsportsapi.com/football"
HEADERS = {'Content-Type': 'application/json'}
SEASON_ID = "2024-2025"

# Cache Configuration
match_cache = TTLCache(maxsize=1000, ttl=3600)  # Cache matches for 1 hour
odds_cache = TTLCache(maxsize=1000, ttl=3600)   # Cache odds for 1 hour
league_cache = TTLCache(maxsize=100, ttl=86400) # Cache leagues for 24 hours

# Initialize ThreadPoolExecutor for CPU-bound tasks
executor = ThreadPoolExecutor(max_workers=4)

# Load Models and Scalers
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

# Flask App Configuration
app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY')
if not app.secret_key:
    logger.error("❌ FLASK_SECRET_KEY not set")
    sys.exit(1)

DATABASE_URL = os.getenv('DATABASE_URL')
if not DATABASE_URL:
    logger.error("❌ DATABASE_URL environment variable not set")
    sys.exit(1)
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
if '?' not in DATABASE_URL:
    DATABASE_URL += '?sslmode=prefer'
app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
    'pool_size': 5,
    'max_overflow': 10,
    'pool_timeout': 30,
    'pool_pre_ping': True,
}
db = SQLAlchemy(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

PAYSTACK_PUBLIC_KEY = os.getenv('PAYSTACK_PUBLIC_KEY', 'pk_test_3ab2fd3709c83c56dd600042ed0ea8690271f6c5')
PAYSTACK_SECRET_KEY = os.getenv('PAYSTACK_SECRET_KEY')

# Database Model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)
    is_vip = db.Column(db.Boolean, default=False)
    vip_expiry = db.Column(db.DateTime, nullable=True)
    is_admin = db.Column(db.Boolean, default=False)

    def set_password(self, password):
        if not password:
            logger.error("❌ set_password: No password provided")
            raise ValueError("Password cannot be empty")
        self.password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        logger.debug(f"Password hash set for user {self.username}")

    def check_password(self, password):
        if not password or not self.password_hash:
            logger.error(f"❌ check_password: Missing password or password_hash for {self.username}")
            return False
        try:
            return bcrypt.checkpw(password.encode('utf-8'), self.password_hash.encode('utf-8'))
        except Exception as e:
            logger.error(f"❌ Error checking password for {self.username}: {e}")
            return False

# Admin User Creation
def create_admin_user():
    admin_username = os.getenv('ADMIN_USERNAME', 'admin')
    admin_password = os.getenv('ADMIN_PASSWORD')
    if not admin_password:
        logger.error("❌ ADMIN_PASSWORD not set, cannot create admin user")
        return False
    try:
        with app.app_context():
            admin_user = User.query.filter_by(username=admin_username).first()
            if not admin_user:
                admin_user = User(username=admin_username, is_admin=True, is_vip=True)
                admin_user.set_password(admin_password)
                db.session.add(admin_user)
                db.session.commit()
                logger.info(f"✅ Admin user '{admin_username}' created with is_admin=True, is_vip=True")
                return True
            else:
                updates = []
                if not admin_user.is_admin:
                    admin_user.is_admin = True
                    updates.append("is_admin=True")
                if not admin_user.is_vip:
                    admin_user.is_vip = True
                    updates.append("is_vip=True")
                admin_user.set_password(admin_password)
                if updates:
                    db.session.commit()
                    logger.info(f"✅ Admin user '{admin_username}' updated: {', '.join(updates)}")
                else:
                    logger.info(f"Admin user '{admin_username}' already exists with correct permissions")
                return True
    except Exception as e:
        logger.error(f"❌ Error creating/updating admin user '{admin_username}': {e}")
        return False

# Routes for Database Initialization
@app.route('/init_db', methods=['GET'])
@retry(retry=retry_if_exception_type(OperationalError), stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def init_db():
    try:
        with app.app_context():
            db.create_all()
        logger.info("✅ Database tables created")
        return "Database tables created!"
    except Exception as e:
        logger.error(f"❌ Error initializing database: {e}")
        return f"Error: {str(e)}", 500

@app.route('/update_db', methods=['GET'])
@retry(retry=retry_if_exception_type(OperationalError), stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def update_db():
    try:
        with app.app_context():
            db.create_all()
            with db.engine.connect() as conn:
                result = conn.execute(text("SELECT column_name FROM information_schema.columns WHERE table_name = 'user' AND column_name = 'is_admin'"))
                if not result.fetchone():
                    logger.info("Adding is_admin column to user table")
                    conn.execute(text("ALTER TABLE \"user\" ADD COLUMN is_admin BOOLEAN DEFAULT FALSE"))
                    conn.commit()
                    logger.info("✅ is_admin column added")
                else:
                    logger.info("is_admin column already exists")
            success = create_admin_user()
            if success:
                logger.info("✅ Database schema updated and admin user created/updated")
                return "Database schema updated and admin user created!"
            else:
                logger.error("❌ Admin user creation failed, but schema updated")
                return "Database schema updated, but admin user creation failed", 500
    except Exception as e:
        logger.error(f"❌ Error updating database schema: {e}")
        return f"Error: {str(e)}", 500

@app.route('/ping_db')
@retry(retry=retry_if_exception_type(OperationalError), stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def ping_db():
    try:
        with db.engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("✅ Database ping successful")
        return "Database ping successful"
    except Exception as e:
        logger.error(f"❌ Database ping failed: {e}")
        return f"Database ping failed: {str(e)}", 500

@app.route('/scheduler_status')
def scheduler_status():
    global scheduler
    return jsonify({"scheduler_running": scheduler.running if 'scheduler' in globals() else False})

@retry(retry=retry_if_exception_type(OperationalError), stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
@login_manager.user_loader
def load_user(user_id):
    try:
        user = User.query.get(int(user_id))
        logger.debug(f"User loaded: {user.username if user else 'None'}")
        return user
    except OperationalError as e:
        logger.error(f"❌ Database error in load_user: {e}")
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error in load_user: {e}")
        return None

# Team Name Matching
def is_team_match(api_team_name, expected_team_name, threshold=60):  # Lowered from 65
    score = fuzz.token_set_ratio(api_team_name.lower(), expected_team_name.lower())
    logger.debug(f"is_team_match: Comparing '{api_team_name}' vs '{expected_team_name}', Score: {score}, Threshold: {threshold}")
    return score >= threshold

# Asynchronous API Call with Retry
async def fetch_with_retry_async(session: ClientSession, url: str, semaphore: asyncio.Semaphore):
    global api_call_count
    async with semaphore:
        api_call_count += 1
        logger.info(f"API Call #{api_call_count}: {url}")
        retry_options = aiohttp_retry.ExponentialRetry(
            attempts=3,
            start_timeout=4,
            max_timeout=10,
            factor=2.0,
            exceptions={aiohttp.ClientError}
        )
        retry_client = aiohttp_retry.RetryClient(client_session=session, retry_options=retry_options)
        async with retry_client.get(url, headers=HEADERS, timeout=30) as response:
            logger.info(f"Rate Limit Headers: Limit={response.headers.get('X-Rate-Limit-Limit', 'N/A')}, "
                        f"Remaining={response.headers.get('X-Rate-Limit-Remaining', 'N/A')}, "
                        f"Reset={response.headers.get('X-Rate-Limit-Reset', 'N/A')}")
            if response.status == 429:
                retry_after = int(response.headers.get('Retry-After', 10))
                logger.warning(f"Rate limit hit, waiting {retry_after} seconds")
                await asyncio.sleep(retry_after)
                raise aiohttp.ClientError("Rate limit hit")
            response.raise_for_status()
            return await response.json(content_type=None)

# Fetch Odds Asynchronously
async def fetch_odds_async(session: ClientSession, match_id: str, semaphore: asyncio.Semaphore):
    cache_key = f"odds_{match_id}"
    if cache_key in odds_cache:
        logger.info(f"Cache hit for odds: Match ID {match_id}")
        return odds_cache[cache_key]
    
    logger.info(f"Fetching odds for Match ID: {match_id}")
    try:
        url = f"{API_BASE_URL}?met=Odds&matchId={match_id}&APIkey={API_KEY}"
        data = await fetch_with_retry_async(session, url, semaphore)
        logger.debug(f"Odds API response for Match ID {match_id}: {data}")
        if data.get("success") != 1 or not data.get("result"):
            logger.warning(f"⚠️ No odds data found for Match ID: {match_id}")
            return None
        odds_data = data["result"].get(str(match_id), [])
        if not odds_data:
            logger.warning(f"⚠️ Empty odds data for Match ID: {match_id}")
            return None
        for bookmaker in odds_data:
            over_1_5 = float(bookmaker.get('o+1.5', 0)) if bookmaker.get('o+1.5') else None
            under_3_5 = float(bookmaker.get('u+3.5', 0)) if bookmaker.get('u+3.5') else None
            if over_1_5 and under_3_5:
                result = {'over_1_5': over_1_5, 'under_3_5': under_3_5}
                odds_cache[cache_key] = result
                return result
        logger.warning(f"⚠️ No valid Over 1.5 or Under 3.5 odds found for Match ID: {match_id}")
        return None
    except Exception as e:
        logger.error(f"❌ Error fetching odds for Match ID {match_id}: {e}")
        return None

# Fetch All Leagues
async def fetch_all_leagues_async(session: ClientSession, semaphore: asyncio.Semaphore):
    cache_key = "leagues"
    if cache_key in league_cache:
        logger.info("Loaded leagues from cache")
        return league_cache[cache_key]
    
    logger.info("Fetching all leagues...")
    try:
        url = f"{API_BASE_URL}?met=Leagues&APIkey={API_KEY}"
        data = await fetch_with_retry_async(session, url, semaphore)
        logger.debug(f"Leagues API response: {data}")
        if data.get("success") != 1:
            logger.error(f"❌ API error fetching leagues: {data}")
            return []
        leagues = data["result"]
        league_cache[cache_key] = [(league['league_key'], league['league_name'], league.get('country_name', 'Unknown')) for league in leagues]
        with open('leagues.txt', 'w', encoding='utf-8') as f:
            f.write("League List:\n")
            for league in leagues:
                f.write(f"ID: {league['league_key']}, Name: {league['league_name']}, Country: {league.get('country_name', 'Unknown')}\n")
        logger.info(f"✅ Retrieved and cached {len(leagues)} leagues")
        return league_cache[cache_key]
    except Exception as e:
        logger.error(f"❌ Error fetching leagues: {e}")
        return []

# Fetch Upcoming Matches for a League
async def fetch_upcoming_matches_async(session: ClientSession, league_id: int, league_name: str, country_name: str, season_id: str, date_from: str, semaphore: asyncio.Semaphore):
    cache_key = f"matches_{league_id}_{date_from}"
    if cache_key in match_cache:
        logger.info(f"Cache hit for matches: {league_name} on {date_from}")
        return match_cache[cache_key]
    
    date_to = (datetime.strptime(date_from, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
    logger.info(f"Fetching matches for {league_name} ({country_name}, ID: {league_id}) from {date_from} to {date_to}")
    try:
        url = f"{API_BASE_URL}?met=Fixtures&leagueId={league_id}&APIkey={API_KEY}&season={season_id}&from={date_from}&to={date_to}"
        data = await fetch_with_retry_async(session, url, semaphore)
        logger.debug(f"Matches API response for {league_name}: {data}")
        if data.get("success") != 1 or not data.get("result"):
            logger.warning(f"⚠️ No matches found for {league_name} from {date_from} to {date_to}")
            return []
        matches = data["result"]
        for match in matches:
            match['league_name'] = league_name
            match['country_name'] = country_name
        match_cache[cache_key] = matches
        logger.info(f"Retrieved {len(matches)} matches for {league_name}")
        return matches
    except Exception as e:
        logger.error(f"❌ Error fetching matches for {league_name}: {e}")
        return []

# Fetch Match Data
async def fetch_match_data_async(session: ClientSession, home_team_key: str, away_team_key: str, season_id: str, league_id: str, match_id: str, match_date: str, home_team_name: str, away_team_name: str, semaphore: asyncio.Semaphore):
    match_info = f"{home_team_name} vs {away_team_name} ({match_date})"
    logger.info(f"Processing match: {match_info}, LeagueID: {league_id}, MatchID: {match_id}")

    cache_key = f"match_data_{home_team_key}_{away_team_key}_{league_id}_{season_id}"
    if cache_key in match_cache:
        logger.info(f"Cache hit for match data: {match_info}")
        return match_cache[cache_key]

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
        home_team_response = await fetch_with_retry_async(session, recent_url, semaphore)
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
                    for card in match_red_cards:
                        try:
                            time_str = card.get('time', '0')
                            minute = int(time_str.split('+')[0]) if '+' in time_str else int(time_str)
                            if minute <= 70:
                                logger.error(f"❌ Skipping {match_info}: Red card at {minute} minutes in match {match.get('event_date')}")
                                return None
                        except (ValueError, TypeError):
                            logger.warning(f"⚠️ Invalid red card time match {match.get('event_date')}: {time_str}")
                            return None
                    home_filtered.append({'match': match, 'red_cards': match_red_cards})
                    if len(home_filtered) == 3:  # Reduced from 5
                        break
                except (ValueError, TypeError):
                    continue
            else:
                logger.warning(f"⚠️ Invalid result format for home match {match.get('event_date')}: {result}")
                continue

    if len(home_filtered) < 3:
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
        away_team_response = await fetch_with_retry_async(session, recent_url, semaphore)
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
                    for card in match_red_cards:
                        try:
                            time_str = card.get('time', '0')
                            minute = int(time_str.split('+')[0]) if '+' in time_str else int(time_str)
                            if minute <= 70:
                                logger.error(f"❌ Skipping {match_info}: Red card at {minute} minutes in match {match.get('event_date')}")
                                return None
                        except (ValueError, TypeError):
                            logger.warning(f"⚠️ Invalid red card time in match {match.get('event_date')}: {time_str}")
                            return None
                    away_filtered.append({'match': match, 'red_cards': match_red_cards})
                    if len(away_filtered) == 3:  # Reduced from 5
                        break
                except (ValueError, TypeError):
                    continue
            else:
                logger.warning(f"⚠️ Invalid result format for away match {match.get('event_date')}: {result}")
                continue

    if len(away_filtered) < 3:
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

    result = {
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
    match_cache[cache_key] = result
    return result

# Optimized Confidence Function
def favour_v6_confidence(row, HomeGoalList, HomeConcededList, AwayGoalList, AwayConcededList, HomeBTTS, AwayBTTS, poisson_prob):
    try:
        HomeGoalList = np.array(HomeGoalList, dtype=int)
        HomeConcededList = np.array(HomeConcededList, dtype=int)
        AwayGoalList = np.array(AwayGoalList, dtype=int)
        AwayConcededList = np.array(AwayConcededList, dtype=int)
        HomeBTTS = np.array(HomeBTTS, dtype=int)
        AwayBTTS = np.array(AwayBTTS, dtype=int)
    except ValueError as e:
        logger.warning(f"⚠️ Invalid input data format: {e}")
        return 0.0, 100.0, []

    required_cols = ['avg_home_scored', 'avg_away_scored', 'avg_home_conceded', 'avg_away_conceded']
    if not all(col in row for col in required_cols):
        missing_cols = [col for col in required_cols if col not in row]
        logger.warning(f"⚠️ Missing required columns in row: {missing_cols}")
        return 0.0, 100.0, []

    home_outcomes = np.where(HomeGoalList > HomeConcededList, 'w', np.where(HomeGoalList == HomeConcededList, 'd', 'l'))
    away_outcomes = np.where(AwayGoalList > AwayConcededList, 'w', np.where(AwayGoalList == AwayConcededList, 'd', 'l'))
    total_wins = np.sum(home_outcomes == 'w') + np.sum(away_outcomes == 'w')
    total_draws = np.sum(home_outcomes == 'd') + np.sum(away_outcomes == 'd')
    total_losses = np.sum(home_outcomes == 'l') + np.sum(away_outcomes == 'l')
    wins_plus_draws = total_wins + total_draws

    all_goals = np.concatenate([HomeGoalList, HomeConcededList, AwayGoalList, AwayConcededList])
    zero_count = np.sum(all_goals == 0)
    high_goal_count = np.sum(all_goals >= 2)
    one_count = np.sum(all_goals == 1)
    scored_sum = row['avg_home_scored'] + row['avg_away_scored']
    conceded_sum = row['avg_home_conceded'] + row['avg_away_conceded']
    division_result = 1.0 if scored_sum == conceded_sum else max(scored_sum, conceded_sum) / min(scored_sum, conceded_sum) if min(scored_sum, conceded_sum) != 0 else float('inf')
    avg_conceded = (row['avg_home_conceded'] + row['avg_away_conceded']) / 2
    avg_conceded_both = (np.sum(HomeConcededList) + np.sum(AwayConcededList)) / 6  # Adjusted for 3 matches
    home_clean_sheets = np.sum(HomeConcededList == 0)
    away_clean_sheets = np.sum(AwayConcededList == 0)
    clean_sheet_rate_home = home_clean_sheets / len(HomeConcededList)
    clean_sheet_rate_away = away_clean_sheets / len(AwayConcededList)
    total_goals_per_match = np.sum(all_goals) / 6  # Adjusted for 3 matches
    low_scoring_matches = np.sum(HomeGoalList + HomeConcededList <= 2) + np.sum(AwayGoalList + AwayConcededList <= 2)
    low_scoring_rate = low_scoring_matches / (len(HomeGoalList) + len(AwayGoalList))

    base_score = poisson_prob * 100
    triggered_rules = ["Poisson: Base score set to Poisson model probability"]

    if total_goals_per_match >= 3.5 and (clean_sheet_rate_home >= 0.3 or clean_sheet_rate_away >= 0.3) and low_scoring_rate >= 0.2:
        base_score *= 0.80
        triggered_rules.append(f"Defensive Outlier Check: -20% to base_score (total goals per match = {total_goals_per_match:.2f} >= 3.5, "
                              f"clean sheet rate home = {clean_sheet_rate_home:.2f} or away = {clean_sheet_rate_away:.2f} >= 0.3, "
                              f"low scoring rate = {low_scoring_rate:.2f} >= 0.2)")

    if avg_conceded >= 1.8 and high_goal_count >= 6:  # Adjusted for fewer matches
        base_score += 20
        triggered_rules.append("Rule 1: +20 to base_score (avg conceded >= 1.8 and high goal/conceded count >= 6)")
    if avg_conceded >= 1.5 and high_goal_count >= 5:  # Adjusted
        base_score += 10
        triggered_rules.append("Rule 4: +10 to base_score (avg conceded >= 1.5 and high goal/conceded count >= 5)")
    if high_goal_count <= 6 and zero_count <= 4:  # Adjusted
        base_score -= 5
        triggered_rules.append("Rule 2: -5 to base_score (high goal/conceded count <= 6 and zero count <= 4)")
    if high_goal_count >= 5 and zero_count >= 5:  # Adjusted
        base_score += 15
        triggered_rules.append("Rule 3: +15 to base_score (high goal/conceded count >= 5 and zero count >= 5)")
    if division_result >= 1.2 and zero_count in [4, 5, 6]:  # Adjusted
        base_score += 15
        triggered_rules.append("Rule 7: +15 to base_score (division result >= 1.2 and zero count in [4, 5, 6])")
    if one_count >= 6:  # Adjusted
        base_score *= 0.9
        triggered_rules.append("Rule: -10% to base_score (count of 1s in HomeGoalList, AwayGoalList, HomeConcededList, AwayConcededList >= 6)")
    if wins_plus_draws >= 5 and avg_conceded <= 1.5:  # Adjusted
        base_score *= 0.80
        triggered_rules.append(f"New Rule: -15% to base_score (wins + draws = {wins_plus_draws} >= 5 and avg conceded = {avg_conceded:.2f} <= 1.5)")
    if total_losses >= 4 and avg_conceded_both >= 1.6:  # Adjusted
        base_score *= 1.20
        triggered_rules.append(f"New Loss Rule: +15% to base_score (losses = {total_losses} >= 4 and avg conceded both teams = {avg_conceded_both:.2f} >= 1.6)")
    if total_losses >= 3 and high_goal_count >= 4:  # Adjusted
        base_score *= 1.25
        triggered_rules.append(f"New Rule: +25% to base_score (losses = {total_losses} >= 3 and high goal/conceded count = {high_goal_count} >= 4)")

    base_score = max(0, min(base_score, 100))
    over_conf = max(0, min(base_score, 90))
    under_conf = max(0, min(100 - base_score, 90))

    return over_conf, under_conf, triggered_rules

# Prediction Logic
def make_prediction(data_dict, match_info, match_id):
    required_length = 3  # Reduced from 5
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
    btts_boost_flag = 2 if btts_count >= 5 else 1 if btts_count >= 4 else 0  # Adjusted
    many_0_1_conceded_flag = int(low_conceded_count >= 4)  # Adjusted
    defensive_strength_flag = int(avg_home_conceded <= 0.8 or avg_away_conceded <= 0.8)
    avoid_match_penalty_flag = int(high_scoring_matches <= 2 and btts_count <= 3)  # Adjusted
    low_conceded_boost = int(low_conceded_count >= 3)  # Adjusted
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

    league_avg_goals = (data_dict['TotalHomeGoals'] + data_dict['TotalHomeConceded'] +
                        data_dict['TotalAwayGoals'] + data_dict['TotalAwayConceded']) / 6  # Adjusted for 3 matches
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

    async def fetch_odds_wrapper():
        async with aiohttp.ClientSession() as session:
            return await fetch_odds_async(session, match_id, semaphore=asyncio.Semaphore(10))

    odds = asyncio.run(fetch_odds_wrapper())
    over_1_5_odds = odds['over_1_5'] if odds else None
    under_3_5_odds = odds['under_3_5'] if odds else None
    if over_1_5_odds is None or under_3_5_odds is None:
        logger.warning(f"No odds available for {match_info}, proceeding with probability-based recommendation")

    if zero_count in [4, 5, 6] and meta_probs[0] > meta_probs[1]:  # Adjusted for 3 matches
        recommendation = "NO BET"
        reason = f"Match rejected: {zero_count} zeros in goal/conceded lists and meta-model favors Under 3.5 ({meta_under_prob:.1f}% vs Over 1.5 {meta_over_prob:.1f}%)."
    elif meta_over_prob >= 60.0:  # Lowered from 70.0
        if over_1_5_odds and 1.05 <= over_1_5_odds <= 1.30:  # Relaxed range
            recommendation = "Over 1.5"
            reason = f"Meta-Model Over 1.5 Probability ({meta_over_prob:.1f}%) meets or exceeds 60% threshold, and Over 1.5 odds ({over_1_5_odds:.2f}) are within 1.05-1.30 range."
        else:
            recommendation = "Over 1.5 (No Odds)"
            reason = f"Meta-Model Over 1.5 Probability ({meta_over_prob:.1f}%) meets 60% threshold, but Over 1.5 odds ({over_1_5_odds:.2f if over_1_5_odds else 'N/A'}) are unavailable or outside 1.05-1.30 range."
    elif meta_under_prob >= 60.0:  # Lowered from 70.0
        if under_3_5_odds and 1.05 <= under_3_5_odds <= 1.30:  # Relaxed range
            recommendation = "Under 3.5"
            reason = f"Meta-Model Under 3.5 Probability ({meta_under_prob:.1f}%) meets or exceeds 60% threshold, and Under 3.5 odds ({under_3_5_odds:.2f if under_3_5_odds else 'N/A'}) are within 1.05-1.30 range."
        else:
            recommendation = "Under 3.5 (No Odds)"
            reason = f"Meta-Model Under 3.5 Probability ({meta_under_prob:.1f}%) meets 60% threshold, but Under 3.5 odds ({under_3_5_odds:.2f if under_3_5_odds else 'N/A'}) are unavailable or outside 1.05-1.30 range."
    else:
        reason = f"Neither Meta-Model Over 1.5 Probability ({meta_over_prob:.1f}%) nor Under 3.5 Probability ({meta_under_prob:.1f}%) meets the 60% threshold. No bet recommended."

    return {
        'Match': match_info,
        'OverConfidence': over_conf,
        'UnderConfidence': under_conf,
        'MetaOverProb': meta_over_prob,
        'MetaUnderProb': meta_under_prob,
        'Recommendation': recommendation,
        'Reason': reason,
        'TriggeredRules': triggered_rules,
        'Over1_5Odds': over_1_5_odds,
        'Under3_5Odds': under_3_5_odds
    }

# Main Function
async def main_async(date_from=None):
    global api_call_count
    api_call_count = 0
    wat_tz = pytz.timezone('Africa/Lagos')
    if date_from is None:
        date_from = datetime.now(wat_tz).strftime('%Y-%m-%d')
        logger.info(f"No date provided, defaulting to today: {date_from}")

    season_id = SEASON_ID
    logger.info(f"Using Season ID: {season_id} for date: {date_from}")

    match_cache.clear()
    odds_cache.clear()
    league_cache.clear()

    target_league_ids = [
        152, 302, 207, 175, 168, 266, 244, 332, 322, 279, 56, 135, 308, 307, 171, 312, 155, 156, 614, 593, 160, 352, 353, 192, 
        395, 223, 245, 251, 253, 362, 307, 118
    ]

    async with aiohttp.ClientSession() as session:
        semaphore = asyncio.Semaphore(10)
        leagues = await fetch_all_leagues_async(session, semaphore)
        if not leagues:
            logger.error("❌ Aborting: No leagues retrieved.")
            return

        leagues = [(league_id, league_name, country_name) for league_id, league_name, country_name in leagues
                   if league_id in target_league_ids]

        if not leagues:
            logger.error("❌ Aborting: No matching leagues found for provided IDs.")
            return

        all_matches = []
        logger.info(f"\nFetching matches for {date_from} for {len(leagues)} selected leagues...")
        tasks = [
            fetch_upcoming_matches_async(session, league_id, league_name, country_name, season_id, date_from, semaphore)
            for league_id, league_name, country_name in leagues
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for matches in results:
            if isinstance(matches, list):
                all_matches.extend(matches)

        if not all_matches:
            logger.error(f"❌ No matches found for selected leagues on {date_from}. Trying tomorrow...")
            tomorrow = (datetime.now(wat_tz) + timedelta(days=1)).strftime('%Y-%m-%d')
            tasks = [
                fetch_upcoming_matches_async(session, league_id, league_name, country_name, season_id, tomorrow, semaphore)
                for league_id, league_name, country_name in leagues
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for matches in results:
                if isinstance(matches, list):
                    all_matches.extend(matches)
            if not all_matches:
                logger.error(f"❌ No matches found for {tomorrow} either. Aborting.")
                return

        logger.info(f"\nFound {len(all_matches)} matches for {date_from}:")
        for match in all_matches:
            match_info = f"{match.get('event_home_team', 'Unknown')} vs {match.get('event_away_team', 'Unknown')} ({match['league_name']}, {match['country_name']})"
            logger.info(f"- {match_info} (Match ID: {match.get('event_key')})")

        results = []
        skipped_matches = []
        logger.info("\nPredicting outcomes...")
        
        match_data_tasks = []
        odds_tasks = []
        for match in all_matches:
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

            match_data_tasks.append(
                fetch_match_data_async(
                    session, home_team_key, away_team_key, season_id, league_id, match_id, match_date, home_team_name, away_team_name, semaphore
                )
            )
            odds_tasks.append(fetch_odds_async(session, match_id, semaphore))

        match_data_results = await asyncio.gather(*match_data_tasks, return_exceptions=True)
        odds_results = await asyncio.gather(*odds_tasks, return_exceptions=True)

        def process_match(args):
            data_dict, odds, match_info, match_id = args
            if data_dict is None or isinstance(data_dict, Exception):
                return None, match_info
            result = make_prediction(data_dict, match_info, match_id)
            if 'error' in result:
                return None, match_info
            result['Over1_5Odds'] = odds['over_1_5'] if odds else None
            result['Under3_5Odds'] = odds['under_3_5'] if odds else None
            return result, None

        tasks = [
            (data_dict, odds, f"{match.get('event_home_team', 'Unknown')} vs {match.get('event_away_team', 'Unknown')} ({match.get('event_date')}, {match['league_name']}, {match['country_name']})", match.get('event_key'))
            for data_dict, odds, match in zip(match_data_results, odds_results, all_matches)
        ]
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            prediction_results = list(tqdm(executor.map(process_match, tasks), total=len(tasks), desc="Predicting matches"))

        for result, skipped in prediction_results:
            if result:
                results.append(result)
            elif skipped:
                skipped_matches.append(skipped)

        logger.info(f"Total API Calls Made: {api_call_count}")
        predictions_data = [
            {
                'Match': r['Match'],
                'MetaOverProb': r['MetaOverProb'],
                'MetaUnderProb': r['MetaUnderProb'],
                'Recommendation': r['Recommendation'],
                'Reason': r['Reason'],
                'OverConfidence': r['OverConfidence'],
                'UnderConfidence': r['UnderConfidence'],
                'TriggeredRules': r['TriggeredRules'],
                'Over1_5Odds': r['Over1_5Odds'],
                'Under3_5Odds': r['Under3_5Odds']
            } for r in results
        ]
        with open('predictions.json', 'w', encoding='utf-8') as f:
            json.dump(predictions_data, f)

        logger.info("\n=== Prediction Results ===")
        with open('predictions.txt', 'w', encoding='utf-8') as f:
            if not results:
                msg = f"❌ No valid predictions for {date_from}. Check prediction_log.txt for details."
                logger.error(msg)
                f.write(msg + "\n")
            for result in results:
                over_odds = f"{result['Over1_5Odds']:.2f}" if result['Over1_5Odds'] is not None else 'N/A'
                under_odds = f"{result['Under3_5Odds']:.2f}" if result['Under3_5Odds'] is not None else 'N/A'
                output = (f"\nMatch: {result['Match']}\n"
                          f"Over 1.5 Confidence: {result['OverConfidence']:.1f}%\n"
                          f"Under 3.5 Confidence: {result['UnderConfidence']:.1f}%\n"
                          f"Meta-Model Over 1.5 Probability: {result['MetaOverProb']:.1f}%\n"
                          f"Meta-Model Under 3.5 Probability: {result['MetaUnderProb']:.1f}%\n"
                          f"Over 1.5 Odds: {over_odds}\n"
                          f"Under 3.5 Odds: {under_odds}\n"
                          f"Recommendation: {result['Recommendation']}\n"
                          f"Reason: {result['Reason']}\n"
                          f"Triggered Rules:\n" + "\n".join(result['TriggeredRules']) + "\n" + f"{'='*50}")
                logger.info(output)
                f.write(output + "\n")
            if skipped_matches:
                logger.info("\n=== Skipped Matches ===")
                f.write("\nSkipped Matches:\n" + "\n".join(skipped_matches) + "\n")

# Wrapper for main function
def main(date_from=None):
    asyncio.run(main_async(date_from))

# Load Predictions
def load_predictions():
    try:
        with open('predictions.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"❌ Error loading predictions.json: {e}")
        return []

# Schedule Predictions
scheduler = None
def schedule_predictions():
    if os.getenv('RUN_MODE', 'web') != 'worker':
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
        misfire_grace_time=3600,
        timezone=wat_tz
    )
    scheduler.add_listener(
        lambda event: logger.info(f"Job {event.job_id} executed") if not event.exception else logger.error(f"Job {event.job_id} failed: {event.exception}"),
        EVENT_JOB_EXECUTED | EVENT_JOB_ERROR
    )
    logger.info("Scheduler started for 12:15 AM WAT daily predictions")
    scheduler.start()
    return scheduler

# Flask Routes
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
        if current_user.is_admin:
            predictions = load_predictions()
            vip_preds = [p for p in predictions if p['Recommendation'] != "NO BET"]
            return render_template('vip.html', predictions=vip_preds, user=current_user, admin_message="Admin Access: Full VIP privileges")
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

@app.route('/predictions')
@login_required
def predictions():
    try:
        if not current_user.is_vip and not current_user.is_admin:
            flash('VIP access required for predictions.', 'danger')
            return redirect(url_for('pay'))
        predictions = load_predictions()
        vip_preds = [p for p in predictions if p['Recommendation'] != "NO BET"]
        return render_template('predictions.html', predictions=vip_preds, user=current_user)
    except OperationalError as e:
        logger.error(f"❌ Database error in predictions route: {e}")
        flash('Database connection issue. Please try again.', 'danger')
        return redirect(url_for('home'))
    except Exception as e:
        logger.error(f"❌ Unexpected error in predictions route: {e}")
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
            logger.info(f"User {username} registered successfully")
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
        logger.debug(f"Authenticated user {current_user.username} redirected from /login")
        return redirect(url_for('home'))
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        logger.debug(f"Login attempt: username={username}")
        if not username or not password:
            logger.warning("Login: Missing username or password")
            flash('Username and password are required.', 'danger')
            return render_template('login.html')
        try:
            user = User.query.filter_by(username=username).first()
            if not user:
                logger.warning(f"Login: User {username} not found")
                flash('Invalid username or password.', 'danger')
                return render_template('login.html')
            logger.debug(f"User found: {user.username}, is_admin={user.is_admin}, is_vip={user.is_vip}")
            if not user.password_hash:
                logger.error(f"Login: No password hash for {username}")
                flash('Account error: No password set.', 'danger')
                return render_template('login.html')
            if user.check_password(password):
                login_user(user)
                logger.info(f"User {username} logged in successfully")
                flash('Logged in successfully!', 'success')
                return redirect(url_for('home'))
            else:
                logger.warning(f"Login: Incorrect password for {username}")
                flash('Invalid username or password.', 'danger')
                return render_template('login.html')
        except OperationalError as e:
            logger.error(f"❌ Database error during login: {e}")
            flash('Database connection issue. Please try again later.', 'danger')
            return render_template('login.html', error="Database connection issue")
        except Exception as e:
            logger.error(f"❌ Unexpected error during login: {e}")
            flash('Server error. Please try again later.', 'danger')
            return render_template('login.html', error="Server error")
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

# CLI Command for Manual Testing
@app.cli.command("run-predictions")
def run_predictions():
    wat_tz = pytz.timezone('Africa/Lagos')
    logger.info("Running predictions via CLI command")
    main(date_from=datetime.now(wat_tz).strftime('%Y-%m-%d'))

# Main Entry Point
if __name__ == '__main__':
    if os.getenv('RUN_MODE', 'web') == 'worker':
        logger.info("Starting in worker mode")
        scheduler = schedule_predictions()
        try:
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            if scheduler:
                scheduler.shutdown()
                logger.info("Worker shutdown")
    else:
        logger.info("Starting in web mode")
        with app.app_context():
            db.create_all()
            create_admin_user()
        app.run(debug=False)
