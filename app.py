# app.py (cleaned & hardened for local + Render deployment)
import os
import pickle
import json
import numpy as np
from datetime import datetime
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
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "artifacts", "model.pkl")
LE_PATH = os.path.join(BASE_DIR, "artifacts", "label_encoder.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "artifacts", "top_features.pkl")

model = None
le = None
top_features = []

def load_artifacts():
    """
    Loads model, label encoder, and features from artifacts/ directory.
    Ensures top_features is always a Python list (never a numpy array).
    """
    global model, le, top_features

    # load model
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        print("✅ Model loaded from", MODEL_PATH)
    except Exception as e:
        model = None
        print("⚠️ Could not load model:", e)

    # load label encoder
    try:
        with open(LE_PATH, "rb") as f:
            le = pickle.load(f)
        print("✅ Label encoder loaded from", LE_PATH)
    except Exception as e:
        le = None
        print("⚠️ Could not load label encoder:", e)

    # load features
    try:
        with open(FEATURES_PATH, "rb") as f:
            loaded = pickle.load(f)
        # Convert to plain Python list (safe for numpy arrays, pandas Index, etc.)
        if isinstance(loaded, np.ndarray):
            top_features = loaded.tolist()
        elif isinstance(loaded, (list, tuple)):
            top_features = list(loaded)
        else:
            # try to coerce to list
            top_features = list(loaded)
        print(f"✅ {len(top_features)} features loaded from", FEATURES_PATH)
    except Exception as e:
        top_features = []
        print("⚠️ Could not load features:", e)

# load on startup
load_artifacts()

# ---------------- Helper: fuzzy mapping ----------------
def map_free_text_to_symptoms(text, symptom_pool, score_cutoff=65):
    """
    Convert free text like "fever, cough" to a list of matching symptom tokens from symptom_pool.
    symptom_pool is expected to be a list of lowercased symptom strings.
    """
    if not text:
        return []
    if not symptom_pool:
        return []

    # normalize symptom_pool to list of lowercase strings
    pool = [str(s).strip().lower() for s in symptom_pool if s is not None]
    text = text.strip().lower()
    # split by common separators
    tokens = [t.strip() for t in text.replace("/", ",").replace(";", ",").split(",") if t.strip()]
    matches = set()
    for token in tokens:
        if token in pool:
            matches.add(token)
        else:
            # fuzzy search top 3 candidates
            best = process.extract(token, pool, scorer=fuzz.WRatio, limit=3)
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

    # missing indicates whether artifacts are present
    missing = model is None or le is None or (not top_features)

    # ensure symptoms_list is a plain list of strings (lowercased for mapping)
    symptoms_list = [str(s) for s in (top_features or [])]

    if request.method == "POST":
        # if artifacts missing, short-circuit with friendly message
        if missing:
            flash("Model artifacts are not loaded. Please add files to the artifacts/ folder.", "danger")
            return redirect(url_for("home"))

        selected = request.form.getlist("symptoms")
        text = request.form.get("symptom_text", "").strip()
        # pass a lowercased pool to fuzzy mapper
        mapped = map_free_text_to_symptoms(text, [s.lower() for s in symptoms_list])
        # normalize selected to lowercase too (templates present original tokens)
        selected_norm = [s.strip().lower() for s in selected if s]
        all_selected = list(dict.fromkeys(selected_norm + mapped))

        if not all_selected:
            flash("Please select or enter symptoms.", "warning")
            return redirect(url_for("home"))

        # build feature vector using symptoms_list matching (case-insensitive)
        X_len = len(symptoms_list)
        x = np.zeros(X_len, dtype=int)
        pool_lower = [s.lower() for s in symptoms_list]
        for s in all_selected:
            if s in pool_lower:
                idx = pool_lower.index(s)
                x[idx] = 1

        try:
            # model expects 2D array
            preds_raw = model.predict([x])
            pred_raw = preds_raw[0]
            # attempt to map encoded label back to original name
            try:
                pred_label = le.inverse_transform([pred_raw])[0]
            except Exception:
                pred_label = str(pred_raw)

            top3 = []
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba([x])[0]
                classes = getattr(model, "classes_", [])
                try:
                    # if classes are encoded ints, inverse_transform will work
                    labels = list(le.inverse_transform(classes))
                except Exception:
                    # fallback: convert classes to strings
                    labels = [str(c) for c in classes]
                pairs = sorted(zip(labels, probs), key=lambda t: t[1], reverse=True)[:3]
                top3 = [{"label": l, "prob": float(p)} for l, p in pairs]
            else:
                top3 = [{"label": pred_label, "prob": None}]

            # save history (store original symptom tokens)
            hist = History(
                user_id=session["user_id"],
                symptoms=",".join(all_selected),
                prediction=pred_label,
                top3_json=json.dumps(top3)
            )
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
        except Exception:
            r.top3 = []
    return render_template("history.html", records=recs)

# ---------------- Main ----------------
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
        # reload artifacts on start (safe)
        load_artifacts()
        print("Server starting - model loaded:", model is not None, "label encoder:", le is not None, "features:", len(top_features))
    # Local dev server
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True, use_reloader=False)
