import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier
import os

# 🧠 Define constants
FEATURE_ORDER = ['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate']
CSV_PATH = "data/pregnancy_dataset.csv"

# Global variables for lazy loading
scaler = None
ensemble_model = None

def load_model():
    """
    Loads and trains the model (only once) from CSV file.
    Returns the trained ensemble model and fitted scaler.
    """
    global ensemble_model, scaler

    if ensemble_model is not None and scaler is not None:
        return ensemble_model, scaler

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV file not found at: {CSV_PATH}")

    # Load and preprocess data
    df = pd.read_csv(CSV_PATH)

    if 'RiskLevel' not in df.columns:
        raise ValueError("Missing 'RiskLevel' column in CSV.")

    df['risk'] = df['RiskLevel'].map({
        'low risk': 0,
        'mid risk': 1,
        'high risk': 2
    })

    if df['risk'].isnull().any():
        raise ValueError("Found unmapped 'RiskLevel' in dataset.")

    X = df[FEATURE_ORDER]
    y = df['risk']

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Define models
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    lgbm = LGBMClassifier(
        num_leaves=31,
        min_child_samples=5,
        min_split_gain=0.0,
        max_depth=-1,
        learning_rate=0.1,
        n_estimators=100,
        random_state=42
    )
    dt = DecisionTreeClassifier(random_state=42)

    ensemble_model = VotingClassifier(
        estimators=[('rf', rf), ('lgbm', lgbm), ('dt', dt)],
        voting='hard'
    )

    ensemble_model.fit(X_train_scaled, y_train)

    return ensemble_model, scaler

def predict_risk(instance: dict) -> str:
    """
    Accepts a dict of features, scales the input, and predicts risk level.
    """
    model, fitted_scaler = load_model()

    try:
        x_input = np.array([[instance[feat] for feat in FEATURE_ORDER]])
    except KeyError as e:
        raise ValueError(f"Missing expected feature in input: {e}")

    x_scaled = fitted_scaler.transform(x_input)
    pred_class = model.predict(x_scaled)[0]

    label_map = {0: "low risk", 1: "medium risk", 2: "high risk"}
    return label_map.get(pred_class, "unknown risk")
