# AraBERT Sentiment on Arabic Hotel Reviews

Fine-tuning AraBERT to classify the sentiment of Arabic hotel reviews, with a PySpark medallion pipeline (bronze, silver, gold) preparing the data.

## Pipeline

| Stage | Notebook | What it does |
|---|---|---|
| Inspect | [`notebook/00_inspect.ipynb`](notebook/00_inspect.ipynb) | Inspect the raw file: encoding, delimiter, columns, types |
| Bronze | [`src/01_bronze.ipynb`](src/01_bronze.ipynb) | Load the raw data as-is with an explicit schema, add ingestion metadata, save as Parquet |
| Silver | [`src/02_silver.ipynb`](src/02_silver.ipynb) | Deduplicate, label sentiment from ratings, remove label-leaking titles, normalise Arabic text, split train / val / test |
| Baseline | [`src/03_baseline.ipynb`](src/03_baseline.ipynb) | TF-IDF + logistic regression in Spark ML, tuned on val; exports the splits for AraBERT and saves test predictions |
| Fine-tune | [`src/04_finetune_kaggle.ipynb`](src/04_finetune_kaggle.ipynb) | Fine-tune `aubmindlab/bert-base-arabertv02` on a Kaggle T4 GPU; push the model to the Hugging Face Hub |
| Inference | [`src/05_inference.ipynb`](src/05_inference.ipynb) | Batch-score the test set with the fine-tuned model on CPU, distributed with Spark `mapInPandas` |
| Gold | [`src/06_gold.ipynb`](src/06_gold.ipynb) | Join both models' test predictions; compare accuracy by review length, rating and user type; check calibration |
| Mixed sentiment | [`src/07_mixed.ipynb`](src/07_mixed.ipynb) | Score each sentence separately to label reviews positive / negative / mixed / neutral (rule-based, checked against star ratings) |

Shared Spark settings and data paths live in [`src/config.py`](src/config.py).

## Results

Held-out test set (10,217 reviews, balanced positive / negative):

| Model | Val accuracy | Test accuracy |
|---|---|---|
| TF-IDF + logistic regression | 85.15% | 85.76% |
| AraBERT v02 (fine-tuned) | 91.91% | **91.85%** |

AraBERT is ahead across every review length, rating and user type, with the largest gain on short reviews (≤10 words: 81.7% → 89.8%). Its confidence is well calibrated: the 82% of reviews it predicts with ≥90% confidence are right 96.8% of the time.

## Data

HARD (Hotel Arabic Reviews Dataset), balanced version: 105,698 reviews in `balanced-reviews.txt` (tab-separated, UTF-16) with hotel name, rating, user type, room type, length of stay and review text.

The data is not included in this repository. Place the file at:

```
data/raw/balanced-reviews.txt
```

## Project layout

```
data/
  raw/       source file (not committed)
  bronze/    raw data as Parquet + ingestion metadata
  silver/    cleaned, labelled data split into train / val / test
  gold/      test predictions from both models, joined for analysis
notebook/    exploration
src/         pipeline steps and shared config
tools/       Windows Hadoop helpers (not committed)
```

## Setup

Requires Python 3.12 and Java 17+ (for PySpark).

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

On Windows, Spark needs Hadoop's `winutils.exe` and `hadoop.dll` to write files. Put both in `tools/hadoop/bin/`; `src/config.py` sets `HADOOP_HOME` to that folder automatically.
