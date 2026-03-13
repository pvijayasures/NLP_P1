# NLP_P1 - Hate Speech Klassifikation (Binary)

Dieses Projekt implementiert eine vollstaendige, reproduzierbare NLP-Pipeline fuer **binare Hate-Speech-Erkennung** auf Basis des Jigsaw-Toxicity-Datensatzes. 
Aus mehreren Toxicity-Labels wird ein einziges Ziel-Label `label` gebildet (`0 = nicht-toxisch`, `1 = toxisch`), anschliessend werden Texte bereinigt, mit TF-IDF vektorisiert und mit klassischen ML-Modellen trainiert.

## Inhalt

- [Projektziel](#projektziel)
- [Pipeline auf einen Blick](#pipeline-auf-einen-blick)
- [Projektstruktur](#projektstruktur)
- [Setup](#setup)
- [Daten und Vorverarbeitung](#daten-und-vorverarbeitung)
- [Training und Evaluation](#training-und-evaluation)
- [Outputs und Artefakte](#outputs-und-artefakte)
- [Reproduzierbarkeit](#reproduzierbarkeit)
- [Troubleshooting](#troubleshooting)
- [Tests](#tests)
- [Lizenz](#lizenz)

## Projektziel

Ziel ist ein klar strukturierter Experiment-Workflow fuer Textklassifikation:

1. Rohdaten laden (`data/raw/train.csv`)
2. Multi-Label -> Binary Label umwandeln
3. Text normalisieren und bereinigen
4. TF-IDF Features erzeugen
5. Modell trainieren und evaluieren
6. Metriken, Vorhersagen, Plots und Fehleranalyse speichern

## Pipeline auf einen Blick

Die Pipeline ist in `src/main.py` orchestriert und umfasst:

1. **Feature-Erzeugung** (`src/features/feature_pipeline.py`)
2. **Train/Test-Split** (stratifiziert)
3. **Modelltraining**
4. **Evaluation** (Accuracy, Precision, Recall, F1, Konfusionsmatrix)
5. **Fehleranalyse und Visualisierung**

Unterstuetzte Modelle (Model Registry in `src/models/__init__.py`):

- `logreg`
- `svm`
- `naive_bayes`
- `random_forest`

## Projektstruktur

```text
NLP_P1/
|- data/
|  |- raw/                # Originaldaten (train/test)
|  |- interim/            # Zwischenstand mit binary label
|  `- processed/          # Vorverarbeitete Daten
|- models/
|  |- trained/            # Gespeicherte Modelle (*.joblib)
|  `- vectorizers/        # TF-IDF-Vektorisierer
|- outputs/
|  |- metrics/            # JSON-Metriken
|  |- plots/              # PNG-Plots
|  `- predictions/        # CSV-Vorhersagen + Fehleranalyse
|- src/
|  |- data/               # Laden/Label-Generierung
|  |- preprocessing/      # Textbereinigung
|  |- features/           # TF-IDF Pipeline
|  |- models/             # Modell-Definitionen
|  |- evaluation/         # Metriken, Plots, Error Analysis
|  `- main.py             # End-to-End Pipeline
|- requirements.txt
`- README.md
```

## Setup

### 1) Voraussetzungen

- Python 3.11+ (empfohlen)
- `pip`

> **Hinweis fuer Windows-Nutzer:** Stelle sicher, dass Python zum `PATH` hinzugefuegt wurde (Checkbox waehrend der Installation). Alle Befehle funktionieren sowohl in **PowerShell** als auch in der **Eingabeaufforderung (CMD)**, sofern nicht anders angegeben.

### 2) Virtuelle Umgebung und Abhaengigkeiten

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

**Windows (PowerShell):**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

**Windows (CMD):**

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
pip install --upgrade pip
pip install -r requirements.txt
```

> **Hinweis:** Falls PowerShell die Skript-Ausfuehrung verweigert, fuehre einmalig als Administrator aus:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

### 3) NLTK-Ressource installieren

Die Vorverarbeitung nutzt englische Stopwords (`nltk.corpus.stopwords`):

**macOS / Linux:**

```bash
python3 -c "import nltk; nltk.download('stopwords')"
```

**Windows:**

```cmd
python -c "import nltk; nltk.download('stopwords')"
```

## Daten und Vorverarbeitung

### Erwartete Rohdaten

Die Pipeline erwartet mindestens:

- `data/raw/train.csv`

Im Repository sind zusaetzlich typische Jigsaw-Dateien vorhanden (z. B. `test.csv`, `test_labels.csv`).

### Schritt 1: Binary-Label erzeugen

Aus den Spalten
`toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`
wird ein einzelnes Label `label` gebildet:

- `label = 1`, wenn mindestens eine Toxicity-Spalte > 0 ist
- sonst `label = 0`

Ausfuehren:

**macOS / Linux:**
```bash
python3 -m src.data.load_data
```

**Windows:**
```cmd
python -m src.data.load_data
```

Output:

- `data/interim/train_binary.csv`

### Schritt 2: Text vorverarbeiten

Die Vorverarbeitung in `src/preprocessing/clean_text.py` umfasst:

- Lowercasing
- Entfernen von URLs, HTML, Zahlen, Satzzeichen
- Tokenisierung (Regex)
- Entfernen von Stopwords (konfigurierbar)
- Stemming mit SnowballStemmer (konfigurierbar)

Ausfuehren:

**macOS / Linux:**
```bash
python3 -m src.preprocessing.preprocess_pipeline
```

**Windows:**
```cmd
python -m src.preprocessing.preprocess_pipeline
```

Output:

- `data/processed/train_binary_preprocessed.csv`

## Training und Evaluation

### End-to-End Lauf

**macOS / Linux:**
```bash
python3 -m src.main --model logreg
```

**Windows:**
```cmd
python -m src.main --model logreg
```

Alternative Modelle:

**macOS / Linux:**
```bash
python3 -m src.main --model svm
python3 -m src.main --model naive_bayes
python3 -m src.main --model random_forest
```

**Windows:**
```cmd
python -m src.main --model svm
python -m src.main --model naive_bayes
python -m src.main --model random_forest
```

Wichtige CLI-Parameter (`src/main.py`):

- `--model`: Modellname
- `--input-file`: Dateiname in `data/processed/` (Default: `train_binary_preprocessed.csv`)
- `--test-size`: Anteil Validierungssplit (Default: `0.2`)
- `--random-state`: Seed (Default: `42`)

Beispiel mit geaendertem Split:

**macOS / Linux:**
```bash
python3 -m src.main --model logreg --test-size 0.25 --random-state 42
```

**Windows:**
```cmd
python -m src.main --model logreg --test-size 0.25 --random-state 42
```

## Outputs und Artefakte

Nach einem Lauf findest du Ergebnisse in:

- `models/trained/`
  - z. B. `logreg_model.joblib`
- `models/vectorizers/`
  - `tfidf_vectorizer.joblib`
- `outputs/metrics/`
  - z. B. `logreg_metrics.json`
- `outputs/predictions/`
  - z. B. `logreg_val_predictions.csv`
  - z. B. `logreg_errors.csv`
- `outputs/plots/`
  - `confusion_matrix_<model>.png`
  - `class_distribution_<model>.png`
  - `prediction_confidence_<model>.png`

## Reproduzierbarkeit

Zentrale Defaults sind in `src/config.py` definiert:

- `RANDOM_STATE = 42`
- `TEST_SIZE = 0.2`
- `TFIDF_CONFIG` (z. B. `max_features=10000`, `ngram_range=(1, 2)`)

Damit sind Experimente konsistent wiederholbar, solange Datenstand und Abhaengigkeiten unveraendert bleiben.

## Troubleshooting

### `LookupError: Resource stopwords not found`

**macOS / Linux:**
```bash
python3 -c "import nltk; nltk.download('stopwords')"
```

**Windows:**
```cmd
python -c "import nltk; nltk.download('stopwords')"
```

### `FileNotFoundError` bei Input-Dateien

Pruefe, ob diese Schritte vorher ausgefuehrt wurden:

**macOS / Linux:**
```bash
python3 -m src.data.load_data
python3 -m src.preprocessing.preprocess_pipeline
```

**Windows:**
```cmd
python -m src.data.load_data
python -m src.preprocessing.preprocess_pipeline
```

### `Model '...' not supported`

Gueltige Modellnamen sind:

- `logreg`
- `svm`
- `naive_bayes`
- `random_forest`
