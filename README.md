# NLP_P1 - Hate-Speech-Klassifikation (Binary)

Dieses Projekt implementiert eine reproduzierbare NLP-Pipeline für **binäre Hate-Speech-Erkennung** auf Basis des Jigsaw-Toxicity-Datensatzes.
Aus mehreren Toxicity-Labels wird ein einzelnes Ziel-Label `label` erzeugt (`0 = nicht-toxisch`, `1 = toxisch`), Texte werden bereinigt und anschliessend mit klassischen ML-Modellen trainiert.

Die Feature-Erzeugung ist modular:

* `tfidf`
* `embeddings` (SentenceTransformer)

---

# Inhalt

* [Projektziel](#projektziel)
* [Pipeline auf einen Blick](#pipeline-auf-einen-blick)
* [Projektstruktur](#projektstruktur)
* [Setup](#setup)
* [Daten und Vorverarbeitung](#daten-und-vorverarbeitung)
* [Training und Evaluation](#training-und-evaluation)
* [Outputs und Artefakte](#outputs-und-artefakte)
* [Reproduzierbarkeit](#reproduzierbarkeit)
* [Troubleshooting](#troubleshooting)

## Direkte Links

* [README](README.md)
* [Konfiguration (`src/config.py`)](src/config.py)
* [End-to-End Pipeline (`src/main.py`)](src/main.py)
* [Batch-Experiment-Runner (`src/run_experiment.py`)](src/run_experiment.py)
* [Datenvorbereitung (`src/data/load_data.py`)](src/data/load_data.py)
* [Preprocessing Pipeline (`src/preprocessing/preprocess_pipeline.py`)](src/preprocessing/preprocess_pipeline.py)
* [Feature Pipeline (`src/features/feature_pipeline.py`)](src/features/feature_pipeline.py)
* [Feature Factory (`src/features/factory.py`)](src/features/factory.py)
* [Model Registry (`src/models/__init__.py`)](src/models/__init__.py)
* [Requirements (`requirements.txt`)](requirements.txt)

---

# Projektziel

Ziel ist ein klarer Experiment-Workflow für Textklassifikation:

1. Rohdaten laden (`data/raw/train.csv`)
2. Multi-Label → Binary Label umwandeln
3. Text normalisieren und bereinigen
4. Features mit gewählter Methode erzeugen (`tfidf` oder `embeddings`)
5. Modell trainieren und evaluieren
6. Metriken, Vorhersagen, Plots und Fehleranalyse speichern

---

# Pipeline auf einen Blick

Die Pipeline in `src/main.py` umfasst:

1. **Features vorbereiten** (`src/features/feature_pipeline.py`)
2. **Train/Test-Split** (stratifiziert)
3. **Modelltraining**
4. **Evaluation** (Accuracy, Precision, Recall, F1, Konfusionsmatrix, Classification Report)
5. **Plots und Fehleranalyse**

Unterstützte Modelle (Registry in `src/models/__init__.py`):

* `logreg`
* `svm`
* `naive_bayes`
* `random_forest`

Unterstützte Feature-Methoden (Factory in `src/features/factory.py`):

* `tfidf`
* `embeddings`

---

# Projektstruktur

```text
NLP_P1/
|- data/
|  |- raw/                # Originaldaten (train/test)
|  |- interim/            # Zwischenstand mit binary label
|  `- processed/          # Vorverarbeitete Daten
|- models/
|  |- trained/            # Gespeicherte Modelle (*.joblib)
|  `- vectorizers/        # TF-IDF- oder Embedding-Artefakte
|- outputs/
|  |- metrics/            # JSON-Metriken
|  |- plots/              # PNG-Plots
|  `- predictions/        # CSV-Vorhersagen + Fehleranalyse
|- src/
|  |- data/               # Laden / Label-Generierung
|  |- preprocessing/      # Textbereinigung
|  |- features/           # Feature-Methoden (tfidf/embeddings)
|  |- models/             # Modell-Definitionen
|  |- evaluation/         # Metriken, Plots, Error Analysis
|  `- main.py             # End-to-End Pipeline
|- requirements.txt
`- README.md

```

---

# Setup

## 1) Voraussetzungen

* Python 3.11+ (empfohlen)
* `pip`

---

## 2) Virtuelle Umgebung und Abhängigkeiten

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

```

### Windows (PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt

```

### Windows (CMD)

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
pip install --upgrade pip
pip install -r requirements.txt

```

---

# NLTK-Ressource installieren

Die Vorverarbeitung nutzt englische Stopwords (`nltk.corpus.stopwords`).

### macOS / Linux

```bash
python3 -c "import nltk; nltk.download('stopwords')"

```

### Windows

```cmd
python -c "import nltk; nltk.download('stopwords')"

```

---

# Daten und Vorverarbeitung

## Erwartete Rohdaten

Mindestens erforderlich:

* `data/raw/train.csv`

---

# Schritt 1: Binary-Label erzeugen

Aus den Spalten

```
toxic
severe_toxic
obscene
threat
insult
identity_hate

```

wird ein einzelnes Label `label` gebildet:

* `label = 1`, wenn mindestens eine Toxicity-Spalte > 0 ist
* sonst `label = 0`

### Ausführen

macOS / Linux

```bash
python3 -m src.data.load_data

```

Windows

```cmd
python -m src.data.load_data

```

Output:

```
data/interim/train_binary.csv

```

---

# Schritt 2: Text vorverarbeiten

Die Vorverarbeitung in `src/preprocessing/preprocess_pipeline.py` umfasst unter anderem:

* Lowercasing
* Entfernen von URLs, HTML und Zahlen
* optionale Zeichennormalisierung (z. B. obfuskiertes Profanity-Mapping)
* Tokenisierung (Regex)
* optionale Stopword-Entfernung
* optionales Stemming
* optionale Datenbereinigung auf Datensatzebene (Duplikate / leere Texte)

### Ausführen

macOS / Linux

```bash
python3 -m src.preprocessing.preprocess_pipeline
```

Windows

```cmd
python -m src.preprocessing.preprocess_pipeline
```

Die Pipeline nutzt die Defaults aus `src/config.py` (z. B. `REMOVE_PUNCTUATION`,
`REMOVE_STOPWORDS`, `STEM_WORDS`, `MIN_TOKEN_LENGTH`).

Output:

```
data/processed/train_binary_preprocessed.csv
```

### Hinweis zur Ablation

`src/preprocessing/preprocess_pipeline.py` enthält zusätzlich `run_ablation_experiment()`
für mehrere `exp_*.csv`-Varianten. Im direkten Script-Start wird aktuell jedoch
`run_standard_pipeline()` ausgeführt.

---

# Training und Evaluation

## End-to-End Lauf

macOS / Linux

```bash
python3 -m src.main --model logreg --feature-method tfidf

```

Windows

```cmd
python -m src.main --model logreg --feature-method tfidf

```

---

# Alternative Modelle

macOS / Linux

```bash
python3 -m src.main --model svm --feature-method tfidf
python3 -m src.main --model naive_bayes --feature-method tfidf
python3 -m src.main --model random_forest --feature-method tfidf

```

Windows

```cmd
python -m src.main --model svm --feature-method tfidf
python -m src.main --model naive_bayes --feature-method tfidf
python -m src.main --model random_forest --feature-method tfidf

```

---

# Alle Modelle in einem Lauf

`src.run_experiment` ist der Batch-Runner für Experiment-Serien.
Er ermittelt automatisch **alle gültigen** Kombinationen aus Modell-Registry
(`src/models/__init__.py`) und Feature-Factory (`src/features/factory.py`) und
führt sie nacheinander als separaten `src.main`-Subprozess aus.
Inkompatible Kombinationen (z. B. `naive_bayes + embeddings`) werden
automatisch übersprungen und am Ende gemeldet.

macOS / Linux

```bash
python3 -m src.run_experiment

```

Windows

```cmd
python -m src.run_experiment

```

Der Runner gibt am Ende eine Zusammenfassung aus:

```
============================================================
FINAL SUMMARY
============================================================
Total Successful: 7
Total Failed    : 0

All experiments completed successfully.

```

Schlägt mindestens ein Lauf fehl, endet der Prozess mit Exit-Code 1 und listet
alle fehlgeschlagenen Kombinationen auf.

> **Hinweis:** Einzelne Läufe können weiterhin direkt über `src.main`
> angestossen werden (siehe Abschnitt *Training und Evaluation*).
> `src.run_experiment` unterstützt keine eigenen CLI-Argumente – alle
> Konfigurationsdefaults werden aus `src/config.py` bezogen.

---

# Beispiel mit Embeddings

macOS / Linux

```bash
python3 -m src.main --model logreg --feature-method embeddings

```

Windows

```cmd
python -m src.main --model logreg --feature-method embeddings

```

---

# Wichtige CLI-Parameter

* `--model` (Default: `logreg`)
* `--feature-method` (Default: `tfidf`)
* `--input-file`
* `--test-size`
* `--random-state`

---

# Outputs und Artefakte

`src/main.py` verwendet das Experiment-Muster:

```
experiment_name = <model>_<feature_method>

```

Beispiele für Dateinamen bei `--model logreg --feature-method tfidf`:

```
models/trained/logreg_tfidf_model.joblib
outputs/metrics/logreg_tfidf_metrics.json
outputs/predictions/logreg_tfidf_val_predictions.csv
outputs/predictions/logreg_tfidf_errors.csv
outputs/plots/confusion_matrix_logreg_tfidf.png
outputs/plots/class_distribution_logreg_tfidf.png
outputs/plots/prediction_confidence_logreg_tfidf.png
outputs/plots/learning_curve_logreg_tfidf.png

```

Feature-Artefakte in `models/vectorizers/` und `models/trained/`:

```
models/vectorizers/tfidf_vectorizer.joblib
models/trained/embedding_model.joblib
models/vectorizers/tfidf_feature_object.joblib
models/vectorizers/embeddings_feature_object.joblib
```

---

# Reproduzierbarkeit

Zentrale Defaults in `src/config.py`:

```
RANDOM_STATE = 42
TEST_SIZE = 0.2
MODEL_NAME = "logreg"
FEATURE_METHOD = "tfidf"
TRAIN_PROCESSED_FILE = "train_binary_preprocessed.csv"

```

Damit sind Experimente konsistent wiederholbar, solange Datenstand und Abhängigkeiten unverändert bleiben.

---

# Troubleshooting

## `LookupError: Resource stopwords not found`

macOS / Linux

```bash
python3 -c "import nltk; nltk.download('stopwords')"

```

Windows

```cmd
python -c "import nltk; nltk.download('stopwords')"

```

---

## `FileNotFoundError` bei Input-Dateien

Prüfe, ob diese Schritte vorher ausgeführt wurden:

macOS / Linux

```bash
python3 -m src.data.load_data
python3 -m src.preprocessing.preprocess_pipeline

```

Windows

```cmd
python -m src.data.load_data
python -m src.preprocessing.preprocess_pipeline

```

---

## `Model '...' not supported`

Gültige Modellnamen:

* `logreg`
* `svm`
* `naive_bayes`
* `random_forest`

---

## `Unknown feature method '...'`

Gültige Feature-Methoden:

* `tfidf`
* `embeddings`
