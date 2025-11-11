# app.py
import os
import pickle
import json
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from rapidfuzz import process, fuzz

# ---------------- Configuration ----------------
app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = os.environ.get("FLASK_SECRET", "super_secret_key")
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///health_assistant.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

# ---------------- Database Models ----------------
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120))
    email = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(300), nullable=False)

    def set_password(self, pw):
        self.password_hash = generate_password_hash(pw)

    def check_password(self, pw):
        return check_password_hash(self.password_hash, pw)

class History(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    symptoms = db.Column(db.Text)
    prediction = db.Column(db.String(200))
    top3_json = db.Column(db.Text)
    created_at = db.Column(db.DateTime, server_default=db.func.now())

# ---------------- Load Model Artifacts ----------------
import os
import pickle

# Define paths relative to the project root
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "artifacts", "model.pkl")
LE_PATH = os.path.join(BASE_DIR, "artifacts", "label_encoder.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "artifacts", "top_features.pkl")

# Initialize variables
model = None
le = None
top_features = None

def load_artifacts():
    global model, le, top_features

    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        print("✅ Model loaded")
    except Exception as e:
        print("⚠️ Could not load model:", e)

    try:
        with open(LE_PATH, "rb") as f:
            le = pickle.load(f)
        print("✅ LabelEncoder loaded")
    except Exception as e:
        print("⚠️ Could not load label encoder:", e)

    try:
        with open(FEATURES_PATH, "rb") as f:
            top_features = pickle.load(f)
        print(f"✅ {len(top_features)} features loaded")
    except Exception as e:
        print("⚠️ Could not load features:", e)

# Load everything
load_artifacts()


# ---------------- Helper: fuzzy mapping ----------------
def map_free_text_to_symptoms(text, symptom_pool, score_cutoff=65):
    if not text or not symptom_pool:
        return []
    text = text.lower()
    tokens = [t.strip() for t in text.replace("/", ",").replace(";", ",").split(",") if t.strip()]
    matches = set()
    for token in tokens:
        if token in symptom_pool:
            matches.add(token)
        else:
            best = process.extract(token, symptom_pool, scorer=fuzz.WRatio, limit=3)
            for cand, score, _ in best:
                if score >= score_cutoff:
                    matches.add(cand)
    return list(matches)

# ---------------- Routes ----------------
@app.route("/")
def index():
    if "user_id" in session:
        return redirect(url_for("home"))
    return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        if not (name and email and password):
            flash("Please fill all fields", "warning")
            return redirect(url_for("register"))
        if User.query.filter_by(email=email).first():
            flash("Email already exists", "danger")
            return redirect(url_for("register"))
        u = User(name=name, email=email)
        u.set_password(password)
        db.session.add(u)
        db.session.commit()
        flash("Registration successful! Please log in.", "success")
        return redirect(url_for("index"))
    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        user = User.query.filter_by(email=email).first()
        if user and user.check_password(password):
            session["user_id"] = user.id
            flash("Logged in successfully", "success")
            return redirect(url_for("home"))
        flash("Invalid credentials", "danger")
        return render_template("login.html")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.pop("user_id", None)
    flash("Logged out", "info")
    return redirect(url_for("index"))

@app.route("/home", methods=["GET", "POST"])
def home():
    if "user_id" not in session:
        return redirect(url_for("index"))
    missing = model is None or le is None or top_features is None
    symptoms_list = top_features if top_features else []
    if request.method == "POST":
        selected = request.form.getlist("symptoms")
        text = request.form.get("symptom_text", "").strip()
        mapped = map_free_text_to_symptoms(text, symptoms_list)
        all_selected = list(dict.fromkeys(selected + mapped))
        if not all_selected:
            flash("Please select or enter symptoms.", "warning")
            return redirect(url_for("home"))
        x = np.zeros(len(symptoms_list))
        for s in all_selected:
            if s in symptoms_list:
                x[symptoms_list.index(s)] = 1
        try:
            pred_raw = model.predict([x])[0]
            try:
                pred_label = le.inverse_transform([pred_raw])[0]
            except:
                pred_label = str(pred_raw)
            top3 = []
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba([x])[0]
                classes = getattr(model, "classes_", [])
                try:
                    labels = list(le.inverse_transform(classes))
                except:
                    labels = [str(c) for c in classes]
                pairs = sorted(zip(labels, probs), key=lambda t: t[1], reverse=True)[:3]
                top3 = [{"label": l, "prob": float(p)} for l, p in pairs]
            else:
                top3 = [{"label": pred_label, "prob": None}]
            hist = History(user_id=session["user_id"], symptoms=",".join(all_selected),
                           prediction=pred_label, top3_json=json.dumps(top3))
            db.session.add(hist)
            db.session.commit()
            return render_template("result.html", prediction=pred_label, top3=top3, symptoms=all_selected)
        except Exception as e:
            flash(f"Prediction error: {e}", "danger")
            return redirect(url_for("home"))
    return render_template("home.html", symptoms=symptoms_list, missing=missing)

@app.route("/history")
def history():
    if "user_id" not in session:
        return redirect(url_for("index"))
    recs = History.query.filter_by(user_id=session["user_id"]).order_by(History.created_at.desc()).all()
    for r in recs:
        try:
            r.top3 = json.loads(r.top3_json)
        except:
            r.top3 = []
    return render_template("history.html", records=recs)

# ---------------- Main ----------------
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
        load_artifacts()  # ✅ ensure model reloads after Flask restart
    app.run(debug=True, use_reloader=False)

