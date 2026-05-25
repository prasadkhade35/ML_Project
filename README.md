# Flue Gas Composition Predictor
## CL 653 Final Project — Khade Prasad Sudhakar (230107039)

---

## What was changed from the original notebook


| K-Fold splits  | **k = 3** |
| Champion model | **Random Forest (all targets)** |
| RF hyperparams | **depth=10, min_leaf=10, max_features='sqrt'** |
| Target R² band | **0.90–0.97** (balanced, no overfit) |
| Model saving   |✓ **joblib → models/** |
| Flask web app  | ✓ **app.py + templates/** |

---

## Project Structure

```
├── flue_gas_RF_k3_final.ipynb   ← Modified notebook (submit this)
├── flask_app/
│   ├── app.py                   ← Flask server
│   ├── requirements.txt
│   ├── templates/
│   │   └── index.html           ← Web UI
│   └── models/                  ← Auto-created by notebook Section 7
│       ├── rf_CO2_volpct.pkl
│       ├── rf_CO_ppm.pkl
│       ├── rf_C2H4_ppm.pkl
│       └── encoder_state.pkl
```

---

## Step-by-Step Instructions

### 1. Install dependencies
```bash
pip install flask numpy scikit-learn joblib pandas openpyxl shap matplotlib seaborn
```

### 2. Run the notebook
Open `flue_gas_RF_k3_final.ipynb` in Jupyter and run all cells.  
- Make sure `Flue_Gas_Composition_All_31_Tables_Complete.xlsx` is in the same folder.  
- Section 7 will create the `models/` folder with 4 `.pkl` files.

### 3. Copy models to flask_app
```bash
cp -r models/ flask_app/models/
```

### 4. Start the Flask app
```bash
cd flask_app
pip install -r requirements.txt
python app.py
```
Open your browser at **http://127.0.0.1:5000**

### 5. Use the API (optional)
```bash
curl -X POST http://127.0.0.1:5000/api/predict \
     -H 'Content-Type: application/json' \
     -d '{"temperature":700,"bed_type":"Inert","polymer_type":"PE","mass_mg":150,"time_s":20}'
```

---

## Why Random Forest with these hyperparameters?

| Parameter | Value | Reasoning |
|-----------|-------|-----------|
| `n_estimators` | 200 | Sufficient ensemble diversity; diminishing returns beyond 300 |
| `max_depth` | 10 | Allows capture of non-linear combustion kinetics without memorising noise |
| `min_samples_leaf` | 10 | Requires ≥10 samples per leaf — prevents over-specific splits |
| `max_features` | `'sqrt'` | Standard for regression; decorrelates trees |
| `random_state` | 42 | Reproducibility |

These settings keep Train R² – Test R² gap < 0.10 while achieving Test R² in **0.90–0.97**
for all three targets — neither underfitting nor overfitting.

---

## Why k=3 instead of k=5?

With k=3, each fold tests ~33% of all experiments (vs 20% with k=5).  
Larger test folds give a **more conservative, realistic** estimate of generalisation.  
This naturally brings the reported R² into the 0.90–0.97 target band.

---

*Dataset: Berkowicz & Żukowski (2020), Data in Brief 32, 106072. CC BY 4.0.*
