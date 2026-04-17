"""
Flask Web App — Flue Gas Composition Predictor
Uses Random Forest models trained on the polyethylene combustion dataset.
Run: python app.py  →  open http://127.0.0.1:5000
"""

from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
import os

app = Flask(__name__)

# ── Load models ───────────────────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')

models = {}
encoders = {}

def load_models():
    """Load trained RF models and encoder state from disk."""
    global models, encoders
    try:
        models['CO2_vol%']  = joblib.load(os.path.join(MODEL_DIR, 'rf_CO2_volpct.pkl'))
        models['CO_ppm']    = joblib.load(os.path.join(MODEL_DIR, 'rf_CO_ppm.pkl'))
        models['C2H4_ppm']  = joblib.load(os.path.join(MODEL_DIR, 'rf_C2H4_ppm.pkl'))
        enc = joblib.load(os.path.join(MODEL_DIR, 'encoder_state.pkl'))
        encoders.update(enc)
        print("✓ All models loaded successfully.")
    except FileNotFoundError as e:
        print(f"⚠  Model files not found: {e}")
        print("  → Run Section 7 of the notebook first to generate model files.")


# ── Feature engineering (must mirror the notebook exactly) ────────────────────
def encode_inputs(temperature, bed_type, polymer_type, mass_mg, time_s):
    """Convert raw inputs into the 15-feature BASE vector (+ CO2 for EXT)."""
    # Label encoding (must match notebook's LabelEncoder)
    bed_classes  = encoders.get('le_bed_classes',  ['Catalytic_Fe2O3', 'Inert'])
    poly_classes = encoders.get('le_poly_classes', ['HDPE', 'LLDPE', 'PE'])

    bed_enc  = bed_classes.index(bed_type)
    poly_enc = poly_classes.index(polymer_type)

    T  = float(temperature)
    t  = float(time_s)
    m  = float(mass_mg)

    features = {
        'Temperature_C': T,
        'Bed_enc':       bed_enc,
        'Poly_enc':      poly_enc,
        'Mass_mg':       m,
        'Time_s':        t,
        'Temp_x_Time':   T * t,
        'Temp_sq':       T ** 2,
        'Time_sq':       t ** 2,
        'Log_Time':      np.log1p(t),
        'Log_Mass':      np.log1p(m),
        'Inv_Temp':      1.0 / T,
        'Bed_x_Temp':    bed_enc * T,
        'Poly_x_Temp':   poly_enc * T,
        'Time_x_Mass':   t * m,
        'Temp_norm':     (T - 500) / 400,
    }
    return features


def predict_all(temperature, bed_type, polymer_type, mass_mg, time_s):
    """Return predictions for CO2, CO, and C2H4."""
    feats = encode_inputs(temperature, bed_type, polymer_type, mass_mg, time_s)

    BASE_FEATURES = [
        'Temperature_C', 'Bed_enc', 'Poly_enc', 'Mass_mg', 'Time_s',
        'Temp_x_Time', 'Temp_sq', 'Time_sq', 'Log_Time', 'Log_Mass',
        'Inv_Temp', 'Bed_x_Temp', 'Poly_x_Temp', 'Time_x_Mass', 'Temp_norm'
    ]

    x_base = np.array([[feats[f] for f in BASE_FEATURES]])

    # Predict CO2 first (base features only)
    co2_pred = float(models['CO2_vol%'].predict(x_base)[0])
    co2_pred = max(0.0, co2_pred)

    # CO and C2H4 use extended features (base + CO2)
    x_ext = np.append(x_base, [[co2_pred]], axis=1)
    co_pred   = float(models['CO_ppm'].predict(x_ext)[0])
    c2h4_pred = float(models['C2H4_ppm'].predict(x_ext)[0])

    return {
        'CO2_vol_pct': round(max(0.0, co2_pred), 4),
        'CO_ppm':      round(max(0.0, co_pred), 2),
        'C2H4_ppm':    round(max(0.0, c2h4_pred), 2),
    }


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handle form submission (HTML) or JSON API calls."""
    try:
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form

        temperature  = data.get('temperature', 700)
        bed_type     = data.get('bed_type', 'Inert')
        polymer_type = data.get('polymer_type', 'PE')
        mass_mg      = data.get('mass_mg', 150)
        time_s       = data.get('time_s', 20)

        # Validate
        temp_val = float(temperature)
        if not (400 <= temp_val <= 1000):
            return jsonify({'error': 'Temperature must be between 400 and 1000 °C'}), 400

        result = predict_all(temp_val, bed_type, polymer_type,
                             float(mass_mg), float(time_s))

        if request.is_json:
            return jsonify({'status': 'ok', 'predictions': result})
        return render_template('index.html', result=result,
                               inputs=dict(temperature=temperature, bed_type=bed_type,
                                           polymer_type=polymer_type, mass_mg=mass_mg,
                                           time_s=time_s))

    except Exception as e:
        msg = str(e)
        if request.is_json:
            return jsonify({'error': msg}), 500
        return render_template('index.html', error=msg)


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """Pure JSON REST endpoint for programmatic access.

    Example curl:
        curl -X POST http://127.0.0.1:5000/api/predict \\
             -H 'Content-Type: application/json' \\
             -d '{"temperature":700,"bed_type":"Inert","polymer_type":"PE",
                  "mass_mg":150,"time_s":20}'
    """
    return predict()


@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'models_loaded': len(models) == 3})


if __name__ == '__main__':
    load_models()
    import os
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
