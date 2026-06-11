# ================================================================
# app.py — PetPulse AI Flask Backend
# CSE 3811 Artificial Intelligence | UIU | Student ID: 011320065
#
# Usage:
#   pip install -r requirements.txt
#   python app.py
#
# Requires:  pet_disease_full_merged.csv in the same directory.
# API Base:  http://localhost:5000/api
# ================================================================

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import heapq, warnings, os, sys

warnings.filterwarnings("ignore")
np.random.seed(65)   # Derived from student ID 011320065

app  = Flask(__name__)
CORS(app)            # Allow cross-origin requests from the HTML frontend


# ================================================================
# MODEL INITIALIZATION  (runs once at startup)
# ================================================================

def load_and_train():
    """
    Loads pet_disease_full_merged.csv, preprocesses it (CO1),
    trains a Decision Tree (CO4), sets up Naive Bayes tables (CO3),
    and loads A* treatment graphs (CO2).

    Returns a dict of all artefacts needed at inference time.
    """
    csv_path = os.path.join(os.path.dirname(__file__),
                            "pet_disease_full_merged.csv")
    if not os.path.exists(csv_path):
        print(f"ERROR: Dataset not found at {csv_path}")
        print("Place pet_disease_full_merged.csv in the same folder as app.py")
        sys.exit(1)

    print("PetPulse AI — Loading dataset...")
    df = pd.read_csv(csv_path)
    print(f"  Loaded: {df.shape[0]} rows × {df.shape[1]} columns")

    # ── CO1: Preprocessing ──────────────────────────────────
    df_ml = df.drop(columns=["Treatment", "Advice Prevention"]).copy()

    # Fill missing values
    for col in df_ml.select_dtypes(include="object").columns:
        df_ml[col].fillna(df_ml[col].mode()[0], inplace=True)
    for col in df_ml.select_dtypes(include="number").columns:
        df_ml[col].fillna(df_ml[col].median(), inplace=True)

    # Parse "39.2°C" → float
    df_ml["Body Temperature"] = (
        df_ml["Body Temperature"]
        .astype(str)
        .str.replace("°C", "", regex=False)
        .str.replace("C",  "", regex=False)
        .str.strip()
        .astype(float)
    )

    # Binary flag encoding  Yes → 1 / No → 0
    binary_cols = ["Appetite Loss", "Vomiting", "Diarrhea",
                   "Coughing", "Labored Breathing", "Lameness", "Skin Lesions"]
    for col in binary_cols:
        df_ml[col] = df_ml[col].map({"Yes": 1, "No": 0}).fillna(0).astype(int)

    # Label-encode multi-value categorical columns
    cat_cols = ["Animal Type", "Breed", "Gender",
                "Symptom 1", "Symptom 2", "Symptom 3", "Symptom 4"]
    le_dict = {}
    for col in cat_cols:
        le = LabelEncoder()
        df_ml[col] = le.fit_transform(df_ml[col].astype(str))
        le_dict[col] = le

    # Encode target
    le_target = LabelEncoder()
    df_ml["Disease_Encoded"] = le_target.fit_transform(df_ml["Disease Prediction"])

    # StandardScale numerical features
    num_cols = ["Age", "Heart Rate", "Body Temperature"]
    scaler = StandardScaler()
    df_ml[num_cols] = scaler.fit_transform(df_ml[num_cols])

    # ── CO4: Decision Tree ───────────────────────────────────
    feature_cols = [c for c in df_ml.columns
                    if c not in ["Disease Prediction", "Disease_Encoded"]]
    X = df_ml[feature_cols].values
    y = df_ml["Disease_Encoded"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=65
    )
    dt_model = DecisionTreeClassifier(
        criterion="gini", max_depth=5, min_samples_leaf=1, random_state=65
    )
    dt_model.fit(X_train, y_train)
    print(f"  Decision Tree trained  (max_depth=5, gini criterion)")

    # ── CO3: Naive Bayes tables ──────────────────────────────
    symptom_cols  = ["Symptom 1", "Symptom 2", "Symptom 3", "Symptom 4"]
    all_diseases  = df["Disease Prediction"].unique()
    priors        = (df["Disease Prediction"].value_counts() / len(df)).to_dict()

    conditionals = {}
    for disease, group in df.groupby("Disease Prediction"):
        conditionals[disease] = {}
        for col in symptom_cols:
            vc       = group[col].value_counts()
            total    = len(group)
            all_vals = df[col].unique()
            conditionals[disease][col] = {
                val: (vc.get(val, 0) + 1) / (total + len(all_vals))
                for val in all_vals
            }
        for col in binary_cols:
            orig_col = df[col] if col in df.columns else None
            vc    = group[col].value_counts() if col in group.columns else pd.Series()
            total = len(group)
            # Work with original string values from df
            grp_orig = df.loc[group.index, col] if col in df.columns else group[col]
            vc_str   = grp_orig.value_counts()
            conditionals[disease][col] = {
                "Yes": (vc_str.get("Yes", 0) + 1) / (total + 2),
                "No":  (vc_str.get("No",  0) + 1) / (total + 2),
            }
    print(f"  Naive Bayes tables built  ({len(all_diseases)} disease classes, Laplace smoothing)")

    return dict(
        df=df, df_ml=df_ml, le_dict=le_dict, le_target=le_target,
        scaler=scaler, dt_model=dt_model, feature_cols=feature_cols,
        binary_cols=binary_cols, symptom_cols=symptom_cols,
        all_diseases=all_diseases, priors=priors, conditionals=conditionals,
    )


M = load_and_train()


# ================================================================
# A*  TREATMENT GRAPHS  (CO2)
# ================================================================

TREATMENT_GRAPHS = {
    "Canine Parvovirus": {
        "start": "Triage & Isolation",
        "goal":  "Recovery & Discharge",
        "edges": {
            "Triage & Isolation":         [("IV Fluid Therapy", 1),
                                           ("Broad-Spectrum Antibiotics", 3)],
            "IV Fluid Therapy":           [("Antiemetics (Maropitant)", 2)],
            "Antiemetics (Maropitant)":   [("Broad-Spectrum Antibiotics", 1),
                                           ("Nutritional Support", 2)],
            "Broad-Spectrum Antibiotics": [("Nutritional Support", 1)],
            "Nutritional Support":        [("Viral Load Monitoring", 2)],
            "Viral Load Monitoring":      [("Recovery & Discharge", 1)],
            "Recovery & Discharge":       [],
        },
        "heuristics": {
            "Triage & Isolation": 6, "IV Fluid Therapy": 5,
            "Antiemetics (Maropitant)": 4, "Broad-Spectrum Antibiotics": 3,
            "Nutritional Support": 2, "Viral Load Monitoring": 1,
            "Recovery & Discharge": 0,
        },
    },
    "Bovine Tuberculosis": {
        "start": "Quarantine Animal",
        "goal":  "Re-Testing & Clearance",
        "edges": {
            "Quarantine Animal":          [("Tuberculin Skin Test", 1)],
            "Tuberculin Skin Test":       [("Confirmatory PCR/Culture", 2),
                                           ("Notify Authorities", 4)],
            "Confirmatory PCR/Culture":   [("Notify Authorities", 1)],
            "Notify Authorities":         [("Herd Depopulation Protocol", 2)],
            "Herd Depopulation Protocol": [("Premises Disinfection", 1)],
            "Premises Disinfection":      [("Re-Testing & Clearance", 2)],
            "Re-Testing & Clearance":     [],
        },
        "heuristics": {
            "Quarantine Animal": 6, "Tuberculin Skin Test": 5,
            "Confirmatory PCR/Culture": 4, "Notify Authorities": 3,
            "Herd Depopulation Protocol": 2, "Premises Disinfection": 1,
            "Re-Testing & Clearance": 0,
        },
    },
    "Upper Respiratory Infection": {
        "start": "Clinical Assessment",
        "goal":  "Follow-Up & Discharge",
        "edges": {
            "Clinical Assessment":     [("Nasal Swab & Culture", 1),
                                        ("Antibiotic Therapy", 3)],
            "Nasal Swab & Culture":    [("Antibiotic Therapy", 1),
                                        ("Antiviral (Famciclovir)", 2)],
            "Antibiotic Therapy":      [("Nebulization Therapy", 1)],
            "Antiviral (Famciclovir)": [("Nebulization Therapy", 1)],
            "Nebulization Therapy":    [("Nutritional Support", 1)],
            "Nutritional Support":     [("Follow-Up & Discharge", 1)],
            "Follow-Up & Discharge":   [],
        },
        "heuristics": {
            "Clinical Assessment": 6, "Nasal Swab & Culture": 5,
            "Antibiotic Therapy": 4, "Antiviral (Famciclovir)": 4,
            "Nebulization Therapy": 3, "Nutritional Support": 2,
            "Follow-Up & Discharge": 0,
        },
    },
    "Feline Infectious Peritonitis": {
        "start": "Isolation & Supportive Care",
        "goal":  "Discharge & Long-term Care",
        "edges": {
            "Isolation & Supportive Care": [("Effusion Drainage", 2),
                                             ("GS-441524 Antiviral", 3)],
            "Effusion Drainage":           [("GS-441524 Antiviral", 1)],
            "GS-441524 Antiviral":         [("Prednisolone Therapy", 2)],
            "Prednisolone Therapy":        [("Blood Panel Monitoring", 2)],
            "Blood Panel Monitoring":      [("Response Assessment", 1)],
            "Response Assessment":         [("Discharge & Long-term Care", 1)],
            "Discharge & Long-term Care":  [],
        },
        "heuristics": {
            "Isolation & Supportive Care": 6, "Effusion Drainage": 5,
            "GS-441524 Antiviral": 4, "Prednisolone Therapy": 3,
            "Blood Panel Monitoring": 2, "Response Assessment": 1,
            "Discharge & Long-term Care": 0,
        },
    },
    "Bovine Respiratory Disease": {
        "start": "Isolation & Rest",
        "goal":  "Recovery & Return to Herd",
        "edges": {
            "Isolation & Rest":                     [("Temperature & Vitals Check", 1)],
            "Temperature & Vitals Check":           [("Antibiotic Therapy (Tulathromycin)", 1)],
            "Antibiotic Therapy (Tulathromycin)":   [("NSAID Pain Management", 1)],
            "NSAID Pain Management":                [("Bronchodilator Support", 1)],
            "Bronchodilator Support":               [("Nutritional & Hydration Support", 1)],
            "Nutritional & Hydration Support":      [("Recovery & Return to Herd", 1)],
            "Recovery & Return to Herd":            [],
        },
        "heuristics": {
            "Isolation & Rest": 6, "Temperature & Vitals Check": 5,
            "Antibiotic Therapy (Tulathromycin)": 4, "NSAID Pain Management": 3,
            "Bronchodilator Support": 2, "Nutritional & Hydration Support": 1,
            "Recovery & Return to Herd": 0,
        },
    },
    "_generic_": {
        "start": "Emergency Triage",
        "goal":  "Recovery & Discharge",
        "edges": {
            "Emergency Triage":          [("Diagnostic Workup", 1)],
            "Diagnostic Workup":         [("Symptomatic Stabilisation", 2)],
            "Symptomatic Stabilisation": [("Targeted Treatment", 1)],
            "Targeted Treatment":        [("Progress Monitoring", 1)],
            "Progress Monitoring":       [("Recovery & Discharge", 1)],
            "Recovery & Discharge":      [],
        },
        "heuristics": {
            "Emergency Triage": 5, "Diagnostic Workup": 4,
            "Symptomatic Stabilisation": 3, "Targeted Treatment": 2,
            "Progress Monitoring": 1, "Recovery & Discharge": 0,
        },
    },
}


# ================================================================
# HELPER FUNCTIONS
# ================================================================

def a_star(start, goal, edges, heuristics):
    """
    A* Search (CO2).  Returns (path: list[str], cost: int).
    f(n) = g(n) + h(n) — guaranteed optimal when h is admissible.
    """
    open_heap  = [(heuristics.get(start, 0), 0, start, [start])]
    closed_set = set()
    while open_heap:
        f, g, node, path = heapq.heappop(open_heap)
        if node in closed_set:
            continue
        closed_set.add(node)
        if node == goal:
            return path, g
        for neighbour, cost in edges.get(node, []):
            if neighbour not in closed_set:
                new_g = g + cost
                heapq.heappush(
                    open_heap,
                    (new_g + heuristics.get(neighbour, 0), new_g,
                     neighbour, path + [neighbour])
                )
    return [], 0


def get_treatment_path(disease_name):
    """Selects disease-specific or generic A* graph and runs search."""
    g    = TREATMENT_GRAPHS.get(disease_name, TREATMENT_GRAPHS["_generic_"])
    path, cost = a_star(g["start"], g["goal"], g["edges"], g["heuristics"])
    return path, cost


def encode_input_for_ml(session):
    """
    CO1: Converts plain-English user input into the feature vector
    that dt_model expects (same encoding / scaling as training).
    """
    row = {}
    for col in ["Animal Type", "Breed", "Gender",
                "Symptom 1", "Symptom 2", "Symptom 3", "Symptom 4"]:
        val = str(session.get(col, "Unknown"))
        le  = M["le_dict"][col]
        row[col] = int(le.transform([val])[0]) if val in le.classes_ else 0

    for col in M["binary_cols"]:
        row[col] = 1 if session.get(col, "No") == "Yes" else 0

    age_v  = float(session.get("Age", 3))
    hr_v   = float(session.get("Heart Rate", 90))
    temp_v = float(session.get("Body Temperature", 39.2))
    scaled = M["scaler"].transform(
        pd.DataFrame([[age_v, hr_v, temp_v]],
                     columns=["Age", "Heart Rate", "Body Temperature"])
    )[0]
    row["Age"]              = scaled[0]
    row["Heart Rate"]       = scaled[1]
    row["Body Temperature"] = scaled[2]

    return np.array([row[c] for c in M["feature_cols"]]).reshape(1, -1)


def is_input_complete(session):
    """
    True when ALL 4 symptom slots are filled AND ≥4 binary flags
    are present  →  routes to Decision Tree.
    Otherwise  →  routes to Naive Bayes.
    """
    symptoms_ok = sum(
        1 for s in ["Symptom 1", "Symptom 2", "Symptom 3", "Symptom 4"]
        if str(session.get(s, "")).strip()
        not in ("", "Unknown", "None", "— Not sure")
    )
    flags_ok = sum(1 for f in M["binary_cols"] if f in session)
    return (symptoms_ok >= 4) and (flags_ok >= 4)


def bayesian_symptom_reasoning(observed_symptoms, top_n=5):
    """
    CO3: Naive Bayes inference with Laplace smoothing.
    Handles incomplete input naturally — missing features are skipped.
    Log-space arithmetic prevents floating-point underflow.
    """
    log_scores = {}
    for disease in M["all_diseases"]:
        log_score = np.log(M["priors"].get(disease, 1e-9))
        for feature, value in observed_symptoms.items():
            cond = M["conditionals"].get(disease, {}).get(feature, {})
            p    = cond.get(value, 1e-9)
            log_score += np.log(max(p, 1e-9))
        log_scores[disease] = log_score

    vals  = np.array(list(log_scores.values()))
    vals -= vals.max()
    probs  = np.exp(vals)
    probs /= probs.sum()

    ranked = sorted(zip(log_scores.keys(), probs), key=lambda x: -x[1])
    return ranked[:top_n]


def get_clinical_text(disease_name, col_name):
    """Retrieves Treatment or Advice Prevention text from the dataset."""
    match = M["df"][M["df"]["Disease Prediction"] == disease_name]
    if not match.empty:
        return str(match.iloc[0][col_name])
    return "Please consult a licensed veterinarian for tailored advice."


# ================================================================
# API ROUTES
# ================================================================

@app.route("/api/health", methods=["GET"])
def health():
    """Ping endpoint — confirms backend is running."""
    return jsonify({
        "status":   "ok",
        "diseases": len(M["all_diseases"]),
        "records":  len(M["df"]),
        "model":    "DecisionTree (CO4) + NaiveBayes (CO3) + A* (CO2)",
    })


@app.route("/api/diagnose", methods=["POST"])
def diagnose():
    """
    Main inference endpoint.

    Body (JSON): session dict from the HTML frontend.
    Returns JSON with:
      - primary_disease
      - top_diseases  [{disease, probability}]
      - engine_used
      - treatment_path  {path: [{step, action, is_goal}], total_cost}
      - treatment_text
      - prevention_text
      - input_complete
    """
    session = request.get_json(force=True)
    if not session:
        return jsonify({"error": "No JSON payload received."}), 400

    complete = is_input_complete(session)

    # ── Route to inference engine ────────────────────────────
    if complete:
        # CO4: Decision Tree
        fv   = encode_input_for_ml(session)
        prob = M["dt_model"].predict_proba(fv)[0]
        top5 = prob.argsort()[-5:][::-1]
        top_diseases = [
            {"disease":     M["le_target"].classes_[i],
             "probability": float(prob[i])}
            for i in top5
        ]
        engine = "Decision Tree (CO4) — max_depth=5, Gini criterion"
    else:
        # CO3: Naive Bayes
        obs = {
            k: v for k, v in session.items()
            if k in M["symptom_cols"] + M["binary_cols"]
            and str(v).strip() not in ("", "Unknown", "None", "— Not sure")
        }
        bayes = bayesian_symptom_reasoning(obs, top_n=5)
        top_diseases = [
            {"disease": d, "probability": float(p)}
            for d, p in bayes
        ]
        engine = "Naive Bayes (CO3) — Laplace smoothing, log-space arithmetic"

    primary = top_diseases[0]["disease"]

    # ── CO2: A* Treatment Path ───────────────────────────────
    path, cost = get_treatment_path(primary)
    treatment_path = {
        "path": [
            {"step": i + 1, "action": node, "is_goal": (i == len(path) - 1)}
            for i, node in enumerate(path)
        ],
        "total_cost": int(cost),
    }

    return jsonify({
        "primary_disease":  primary,
        "top_diseases":     top_diseases,
        "engine_used":      engine,
        "treatment_path":   treatment_path,
        "treatment_text":   get_clinical_text(primary, "Treatment"),
        "prevention_text":  get_clinical_text(primary, "Advice Prevention"),
        "input_complete":   complete,
    })


@app.route("/api/symptoms", methods=["GET"])
def list_symptoms():
    """Returns all unique symptom values in the dataset (for autocomplete)."""
    syms = set()
    for col in M["symptom_cols"]:
        syms.update(M["df"][col].dropna().unique().tolist())
    return jsonify({"symptoms": sorted(syms)})


@app.route("/api/diseases", methods=["GET"])
def list_diseases():
    """Returns all disease names in the dataset."""
    return jsonify({"diseases": sorted(M["all_diseases"].tolist())})


# ================================================================
# ENTRY POINT
# ================================================================

if __name__ == "__main__":
    print("=" * 58)
    print("  PetPulse AI Backend — http://localhost:5000")
    print("  Endpoints:")
    print("    GET  /api/health")
    print("    POST /api/diagnose  (JSON body)")
    print("    GET  /api/symptoms")
    print("    GET  /api/diseases")
    print("=" * 58)
    app.run(debug=True, port=5000, host="0.0.0.0")
