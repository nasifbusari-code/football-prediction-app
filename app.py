import logging
import os
import json
import requests
from datetime import datetime, timedelta
from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from flask_session import Session
import bcrypt
from paystackapi.paystack import Paystack
from paystackapi.transaction import Transaction
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
import joblib
from rapidfuzz import process, fuzz
from tqdm import tqdm
from apscheduler.schedulers.background import BackgroundScheduler
import pytz

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('prediction_log.txt', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Flask app setup
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-secret-key')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

# Initialize database and login manager
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Paystack configuration
paystack_secret_key = os.environ.get('PAYSTACK_SECRET_KEY')
paystack = Paystack(secret_key=paystack_secret_key)

# API configuration
API_KEY = os.environ.get('SPORTS_API_KEY')
API_BASE_URL = "https://apiv2.allsportsapi.com/football/"
HEADERS = {'Content-Type': 'application/json; charset=utf-8'}

# Load models and scalers
logger.info("Loading models...")
start_time = datetime.now()
scaler = joblib.load('favour_v6_base_scaler.pkl')
gb_model = joblib.load('favour_v6_gb_model.pkl')
xgb_model = joblib.load('favour_v6_xgb_model.pkl')
meta_scaler = joblib.load('favour_v6_meta_scaler.pkl')
meta_model = joblib.load('favour_v6_meta_model.pkl')
logger.info(f"✅ Models and scalers loaded in {(datetime.now() - start_time).total_seconds():.2f} seconds")

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    is_premium = db.Column(db.Boolean, default=False)
    subscription_date = db.Column(db.DateTime)
    payment_ref = db.Column(db.String(100))

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def fetch_all_leagues():
    try:
        url = f"{API_BASE_URL}?met=Leagues&APIkey={API_KEY}"
        response = requests.get(url, headers=HEADERS, timeout=30)
        response.raise_for_status()
        data = response.json()
        if data.get("success") != 1:
            logger.error("❌ Failed to fetch leagues")
            return []
        leagues = data.get("result", [])
        filtered_leagues = []
        exclude_keywords = ['cup', 'women', 'u17', 'u19', 'u21', 'youth']
        for league in leagues:
            league_name = league.get('league_name', '').lower()
            if not any(keyword in league_name for keyword in exclude_keywords):
                filtered_leagues.append((
                    league.get('league_id'),
                    league.get('league_name'),
                    league.get('country_name')
                ))
        logger.info(f"Filtered {len(leagues)} leagues to {len(filtered_leagues)} after excluding cups, women's, and youth leagues")
        logger.info(f"✅ Retrieved {len(filtered_leagues)} eligible leagues")
        return filtered_leagues
    except Exception as e:
        logger.error(f"❌ Error fetching leagues: {e}")
        return []

def fetch_upcoming_matches(league_id, league_name, country_name, season_id, date_from, max_retries=3):
    logger.info(f"Fetching matches for {league_name} ({country_name}, ID: {league_id}) on {date_from}")
    url = f"{API_BASE_URL}?met=Fixtures&leagueId={league_id}&APIkey={API_KEY}&season={season_id}&from={date_from}&to={date_from}"
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=HEADERS, timeout=30)
            response.encoding = 'utf-8'
            response.raise_for_status()
            data = response.json()
            if data.get("success") != 1 or not data.get("result"):
                logger.warning(f"⚠️ No matches found for {league_name} on {date_from}")
                return []
            matches = data["result"]
            for match in matches:
                match['league_name'] = league_name
                match['country_name'] = country_name
            return matches
        except requests.exceptions.RequestException as e:
            logger.warning(f"⚠️ Attempt {attempt + 1}/{max_retries} failed for {league_name}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                logger.error(f"❌ Failed to fetch matches for {league_name} after {max_retries} attempts: {e}")
                return []

def load_predictions():
    try:
        with open('/opt/render/project/src/data/predictions.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"❌ Error loading predictions.json: {e}")
        return []

def main(date_from):
    logger.info(f"Using Season ID: 2024-2025 for date: {date_from}")
    season_id = "2024-2025"
    leagues = [
        (152, "Premier League", "England"),
        (302, "La Liga", "Spain"),
        (207, "Serie A", "Italy"),
        (175, "Bundesliga", "Germany"),
        (168, "Ligue 1", "France"),
        (244, "Eredivisie", "Netherlands"),
        (332, "MLS", "USA"),
        (322, "Süper Lig", "Turkey"),
        (118, "Chinese Super League", "China"),
        (245, "Eerste Divisie", "Netherlands"),
        (223, "Virsliga", "Latvia"),
        (250, "Championship", "Northern Ireland"),
        (251, "Premiership", "Northern Ireland"),
        (329, "USL League Two", "USA"),
        (330, "USL Championship", "USA"),
        (172, "2. Bundesliga", "Germany"),
        (300, "Segunda División", "Spain"),
        (278, "Primeira Liga", "Portugal")
    ]
    problematic_leagues = [332, 322]  # MLS and Süper Lig
    filtered_leagues = [(lid, lname, cname) for lid, lname, cname in leagues if lid not in problematic_leagues]
    logger.info(f"Excluding problematic leagues: {problematic_leagues}, processing {len(filtered_leagues)} leagues")

    all_matches = []
    logger.info(f"\nFetching matches for {date_from} for {len(filtered_leagues)} selected leagues...")
    for league_id, league_name, country_name in tqdm(filtered_leagues, desc="Processing leagues"):
        matches = fetch_upcoming_matches(league_id, league_name, country_name, season_id, date_from, max_retries=3)
        all_matches.extend(matches)

    logger.info(f"✅ Retrieved {len(all_matches)} matches across {len(filtered_leagues)} leagues")
    if not all_matches:
        logger.warning("⚠️ No matches found for the given date")
        return

    # Placeholder for prediction logic (replace with your actual prediction logic)
    results = []
    for match in all_matches:
        # Example: Mock prediction logic (replace with your actual model predictions)
        match_data = {
            'Match': f"{match['home_team']} vs {match['away_team']}",
            'MetaOverProb': np.random.uniform(0, 100),
            'MetaUnderProb': np.random.uniform(0, 100),
            'Recommendation': "Over 1.5" if np.random.uniform(0, 100) > 50 else "Under 3.5",
            'OverConfidence': np.random.uniform(0, 100),
            'UnderConfidence': np.random.uniform(0, 100),
            'Reason': "Based on model analysis",
            'TriggeredRules': ["Rule1", "Rule2"]
        }
        results.append(match_data)

    predictions_data = results
    try:
        with open('/opt/render/project/src/data/predictions.json', 'w', encoding='utf-8') as f:
            json.dump(predictions_data, f)
        with open('/opt/render/project/src/data/predictions.txt', 'w', encoding='utf-8') as f:
            for pred in predictions_data:
                f.write(f"{pred['Match']}: {pred['Recommendation']}\n")
        logger.info(f"✅ Predictions saved to /opt/render/project/src/data/predictions.json and predictions.txt")
    except Exception as e:
        logger.error(f"❌ Error saving predictions: {e}")

def schedule_predictions():
    wat_tz = pytz.timezone('Africa/Lagos')
    scheduler = BackgroundScheduler(timezone=wat_tz)
    scheduler.add_job(
        main,
        'cron',
        hour=22,
        minute=30,
        args=[(datetime.now(wat_tz) + timedelta(days=1)).strftime('%Y-%m-%d')],
        timezone=wat_tz
    )
    logger.info("Scheduler started for 10:30 PM WAT daily predictions")
    scheduler.start()

@app.route('/')
def home():
    predictions = load_predictions()
    if not predictions:
        return render_template('home.html', predictions=[], error="No predictions available yet. Please check back later.")
    free_preds = []
    for pred in predictions:
        high_prob = max(pred['MetaOverProb'], pred['MetaUnderProb'])
        if high_prob > 50:
            pick = "Over 1.5" if pred['MetaOverProb'] > pred['MetaUnderProb'] else "Under 3.5"
            free_preds.append({
                'match': pred['Match'],
                'pick': pred['Recommendation'] if pred['Recommendation'] != "NO BET" else "No Bet"
            })
    return render_template('home.html', predictions=free_preds, error=None, user=current_user)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email).first()
        if user and bcrypt.checkpw(password.encode('utf-8'), user.password.encode('utf-8')):
            login_user(user)
            return redirect(url_for('home'))
        flash('Invalid email or password')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        new_user = User(email=email, password=hashed_password.decode('utf-8'))
        db.session.add(new_user)
        db.session.commit()
        flash('Registration successful! Please log in.')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

@app.route('/subscribe', methods=['GET', 'POST'])
@login_required
def subscribe():
    if request.method == 'POST':
        amount = 500000  # Amount in kobo (5000 NGN)
        email = current_user.email
        try:
            response = Transaction.initialize(
                email=email,
                amount=amount,
                callback_url=url_for('verify_payment', _external=True)
            )
            return redirect(response['data']['authorization_url'])
        except Exception as e:
            flash(f"Payment initialization failed: {e}")
    return render_template('subscribe.html')

@app.route('/verify_payment/<reference>')
@login_required
def verify_payment(reference):
    try:
        response = Transaction.verify(reference=reference)
        if response['data']['status'] == 'success':
            current_user.is_premium = True
            current_user.subscription_date = datetime.utcnow()
            current_user.payment_ref = reference
            db.session.commit()
            flash('Subscription successful!')
        else:
            flash('Payment verification failed.')
    except Exception as e:
        flash(f"Payment verification failed: {e}")
    return redirect(url_for('home'))

@app.route('/vip')
@login_required
def vip():
    if not current_user.is_premium:
        flash('Please subscribe to access VIP predictions.')
        return redirect(url_for('subscribe'))
    predictions = load_predictions()
    return render_template('vip.html', predictions=predictions, user=current_user)

# Temporary route for testing predictions
@app.route('/run-predictions')
def run_predictions():
    wat_tz = pytz.timezone('Africa/Lagos')
    main(date_from=(datetime.now(wat_tz) + timedelta(days=1)).strftime('%Y-%m-%d'))
    return "Predictions generated. Check /opt/render/project/src/data/predictions.json."

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    schedule_predictions()
    app.run(debug=True)
