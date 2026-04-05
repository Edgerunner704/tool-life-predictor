"""
Fatigue Life Predictor for Mechanical Components
Uses Random Forest Regression trained on synthetic S-N curve data
Author: Prabhu S J
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder

# ─────────────────────────────────────────────
# 1. GENERATE SYNTHETIC TRAINING DATA
#    Based on modified Goodman + Basquin's equation
# ─────────────────────────────────────────────

np.random.seed(42)
N = 1000  # samples

materials = np.random.choice(["Steel", "Aluminium", "Titanium"], N)
stress_amp = np.random.uniform(50, 400, N)       # MPa
mean_stress = np.random.uniform(0, 200, N)        # MPa
surface_factor = np.random.uniform(0.5, 1.0, N)  # Ks (finish quality)
kt = np.random.uniform(1.0, 3.5, N)              # stress concentration factor

# Material properties (UTS in MPa, fatigue exponent b)
mat_props = {
    "Steel":     {"UTS": 600, "b": -0.085},
    "Aluminium": {"UTS": 310, "b": -0.100},
    "Titanium":  {"UTS": 900, "b": -0.070},
}

# Compute fatigue life using physics-based formula + noise
log_cycles = np.zeros(N)
for i in range(N):
    props = mat_props[materials[i]]
    Se = 0.5 * props["UTS"] * surface_factor[i] / kt[i]  # endurance limit
    Se_eff = Se * (1 - mean_stress[i] / props["UTS"])     # Goodman correction
    Se_eff = max(Se_eff, 1)                                # avoid divide by zero
    ratio = stress_amp[i] / Se_eff
    b = props["b"]
    log_N = np.log10(max((ratio ** (1 / b)) * 1e6, 1e2))  # Basquin
    log_cycles[i] = log_N + np.random.normal(0, 0.1)      # add slight noise

# ─────────────────────────────────────────────
# 2. ENCODE & SPLIT
# ─────────────────────────────────────────────

le = LabelEncoder()
mat_encoded = le.fit_transform(materials)

X = np.column_stack([stress_amp, mean_stress, surface_factor, kt, mat_encoded])
y = log_cycles

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ─────────────────────────────────────────────
# 3. TRAIN MODEL
# ─────────────────────────────────────────────

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ─────────────────────────────────────────────
# 4. EVALUATE
# ─────────────────────────────────────────────

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("=" * 50)
print("   FATIGUE LIFE PREDICTOR — MODEL REPORT")
print("=" * 50)
print(f"  Model       : Random Forest Regressor (100 trees)")
print(f"  Training set: {len(X_train)} samples")
print(f"  Test set    : {len(X_test)} samples")
print(f"  R² Score    : {r2:.4f}  (1.0 = perfect)")
print(f"  MAE (log N) : {mae:.4f}")
print("=" * 50)

# ─────────────────────────────────────────────
# 5. PREDICT ON NEW COMPONENT
# ─────────────────────────────────────────────

def predict_fatigue(stress_amp, mean_stress, surface_factor, kt, material):
    mat_code = le.transform([material])[0]
    X_new = np.array([[stress_amp, mean_stress, surface_factor, kt, mat_code]])
    log_N = model.predict(X_new)[0]
    cycles = 10 ** log_N

    if log_N >= 6:
        risk = "LOW     ✅  Component likely safe for long-term use"
    elif log_N >= 4.5:
        risk = "MEDIUM  ⚠️   Monitor — moderate fatigue expected"
    else:
        risk = "HIGH    ❌  Risk of early fatigue failure"

    print(f"\n{'─'*50}")
    print(f"  INPUT COMPONENT")
    print(f"{'─'*50}")
    print(f"  Material         : {material}")
    print(f"  Stress Amplitude : {stress_amp} MPa")
    print(f"  Mean Stress      : {mean_stress} MPa")
    print(f"  Surface Factor   : {surface_factor}")
    print(f"  Stress Conc. Kt  : {kt}")
    print(f"{'─'*50}")
    print(f"  Predicted Life   : {cycles:,.0f} cycles  (10^{log_N:.2f})")
    print(f"  Risk Level       : {risk}")
    print(f"{'─'*50}\n")

# ── INPUT for-predictions ── #

def get_float(prompt):
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("Invalid input. Enter a number.")

stress_amp = get_float("Stress Amplitude (MPa): ")
mean_stress = get_float("Mean Stress (MPa): ")
surface_factor = get_float("Surface Factor (0–1): ")
kt = get_float("Stress Concentration Factor (Kt): ")
material = input("Material: ")

predict_fatigue(stress_amp, mean_stress, surface_factor, kt, material)