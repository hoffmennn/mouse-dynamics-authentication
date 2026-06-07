# Používateľská príručka — Mouse Dynamics Authentication

Systém na autentifikáciu používateľov na základe biometrie pohybu myši. V tomto projekte sa nachádzajú zdrojové kódy k celému procesu spracovania surových dát, extrakcie behaviorálnych príznakov a tréning binárneho XGBoost klasifikátora pre každého používateľa a následnú analýzu rozhodnutí modelu.

---
## 1. Inštalácia závislostí

Odporúčaný spôsob: virtuálne prostredie Python 3.10+.

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux / macOS
```

Inštalácia balíčkov:

```bash
pip install pandas numpy scikit-learn xgboost lightgbm shap matplotlib seaborn scipy
```

Alebo vytvorte súbor `requirements.txt` s nasledovným obsahom a spustite `pip install -r requirements.txt`:

```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
xgboost>=2.0
lightgbm>=4.0
shap>=0.44
matplotlib>=3.7
seaborn>=0.12
scipy>=1.11
```

---

## 2. Spustenie projektu

### 2.1 Štruktúra dát

Surové dáta musia byť umiestnené v:

```
data/
   └── sapimouse/
        ├── user1/
        │   ├── session_1_3min.csv
        │   └── session_2_3min.csv
        ├── user2/
        │   └── ...
        └── ...
```

Každý CSV súbor obsahuje stĺpce udalostí myši: `timestamp`, `x`, `y`, `button`, `state`.

### 2.2 Extrakcia príznakov



```bash
python main.py
```


### 2.3 Tréning a vyhodnotenie klasifikátora

```bash
python src/classifier/xgb.py
```

Skript načíta trénovacie a testovacie CSV súbory, natrénuje jeden binárny XGBoost model pre každého používateľa a vypíše metriky AUC a EER.

### 2.4 Explainability analýza (SHAP)

```bash
python src/classifier/binary_xgb-xai.py
```

Generuje SHAP grafy dôležitosti príznakov uložené do priečinka `images/`.



