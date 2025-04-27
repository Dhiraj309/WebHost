import os
from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, current_user
from flask_migrate import Migrate
from flask.cli import with_appcontext

from prediction import generate_labels, predict_summary
from models import db, User
from auth import auth  # Ensure auth blueprint is imported

# Path to store the generated key
secret_key_path = "secret.key"

# Check if a secret key already exists
if os.path.exists(secret_key_path):
    # Read the key from the file
    with open(secret_key_path, "rb") as f:
        secret_key = f.read()
else:
    # Generate a new key and save it
    secret_key = os.urandom(24)
    with open(secret_key_path, "wb") as f:
        f.write(secret_key)

# Set Flask secret key
app = Flask(__name__)
app.config["SECRET_KEY"] = secret_key

# Configuration for PostgreSQL
app.config["SQLALCHEMY_DATABASE_URI"] = (
    "postgresql://postgres:kQznCzQIRXpBDZxdStlskAdXGDsytkSb@shuttle.proxy.rlwy.net:34298/railway"
)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config['REMEMBER_COOKIE_DURATION'] = 86400  # Cookie/session stays for 1 day

# Initialize DB and Migrations
db.init_app(app)
migrate = Migrate(app, db)

# Login Manager
login_manager = LoginManager()
login_manager.login_view = "auth.login"  # Use the correct blueprint route
login_manager.init_app(app)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Register Blueprints
app.register_blueprint(auth, url_prefix="/auth")

@app.route("/")
def landing():
    if current_user.is_authenticated:
        return redirect(url_for('welcome'))  # Go to /welcome if authenticated
    return redirect(url_for('auth.login'))  # Otherwise, redirect to login page

@app.route("/welcome")
def welcome():
    if current_user.is_authenticated:
        return render_template("welcome.html", user=current_user)
    return redirect(url_for('auth.login'))

# Your other routes remain unchanged...

@app.route("/home")
def home():
    if current_user.is_authenticated:
        return render_template("index.html", user=current_user)
    return redirect(url_for('auth.login'))

@app.route("/features")
def features():
    if current_user.is_authenticated:
        return render_template("features.html", user=current_user)
    return redirect(url_for('auth.login'))

@app.route("/text_model")
def text_model():
    if current_user.is_authenticated:
        return render_template("text_model.html", user=current_user)
    return redirect(url_for('auth.login'))

@app.route("/image_model")
def image_model():
    if current_user.is_authenticated:
        return render_template("image_model.html", user=current_user)
    return redirect(url_for('auth.login'))

@app.route("/video_model")
def video_model():
    if current_user.is_authenticated:
        return render_template("video_model.html", user=current_user)
    return redirect(url_for('auth.login'))

@app.route("/files")
def files():
    if current_user.is_authenticated:
        return render_template("files.html", user=current_user)
    return redirect(url_for('auth.login'))

@app.route("/predict_category", methods=["POST"])
def predict_category_endpoint():
    try:
        data = request.get_json()
        text_input = data.get("text", "").strip()

        if not text_input:
            return jsonify({"error": "No text provided"}), 400

        category = generate_labels(text_input)
        return jsonify({"Category": category})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict_summary", methods=["POST"])
def predict_summary_endpoint():
    try:
        data = request.get_json()
        text_input = data.get("text", "").strip()

        if not text_input:
            return jsonify({"error": "No text provided"}), 400

        summary = predict_summary(text_input)
        return jsonify({"Summary": summary})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Flask CLI commands for database management
@app.cli.command("db_init")
@with_appcontext
def db_init():
    from flask_migrate import init
    init()

@app.cli.command("db_migrate")
@with_appcontext
def db_migrate():
    from flask_migrate import migrate
    migrate(message="Initial migration")

@app.cli.command("db_upgrade")
@with_appcontext
def db_upgrade():
    from flask_migrate import upgrade
    upgrade()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
