import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier

# 🧠 Define expected input features
FEATURE_ORDER = ['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate']
CSV_PATH = "data/pregnancy_dataset.csv"  # Make sure this exists!

# 🧼 STEP 1: Load & Prepare Data
df = pd.read_csv(CSV_PATH)

# Map labels to numeric
df['risk'] = df['RiskLevel'].map({
    'low risk': 0,
    'mid risk': 1,
    'high risk': 2
})

X = df[FEATURE_ORDER]
y = df['risk']

# 🧪 STEP 2: Train/Test Split
X_train, _, y_train, _ = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 📏 STEP 3: Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# 🤖 STEP 4: Define Ensemble Model
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

ensemble = VotingClassifier(
    estimators=[('rf', rf), ('lgbm', lgbm), ('dt', dt)],
    voting='hard'
)

# 🏋️ STEP 5: Train
ensemble.fit(X_train_scaled, y_train)

# 🎯 STEP 6: Prediction Function
def predict_risk(instance: dict) -> str:
    # Ensure correct input order
    x_input = np.array([[instance[feat] for feat in FEATURE_ORDER]])
    x_scaled = scaler.transform(x_input)
    pred_class = ensemble.predict(x_scaled)[0]

    # Map back to risk level
    label_map = {0: "low risk", 1: "medium risk", 2: "high risk"}
    return label_map[pred_class]
