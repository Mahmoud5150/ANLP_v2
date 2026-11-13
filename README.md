# AraBERT Sentiment on Arabic Hotel Reviews

Fine-tuning AraBERT to classify the sentiment of Arabic hotel reviews, with a PySpark medallion pipeline (bronze, silver, gold) preparing the data.

## Pipeline

| Stage | Notebook | What it does |
|---|---|---|
| Inspect | [`notebook/00_inspect.ipynb`](notebook/00_inspect.ipynb) | Inspect the raw file: encoding, delimiter, columns, types |
| Bronze | [`src/01_bronze.ipynb`](src/01_bronze.ipynb) | Load the raw data as-is with an explicit schema, add ingestion metadata, save as Parquet |
| Silver | [`src/02_silver.ipynb`](src/02_silver.ipynb) | Deduplicate, label sentiment from ratings, remove label-leaking titles, normalise Arabic text, split train / val / test |
| Baseline | [`src/03_baseline.ipynb`](src/03_baseline.ipynb) | TF-IDF + logistic regression in Spark ML, tuned on val (85.15% val accuracy); exports the splits for AraBERT |
| Gold | — | Model-ready data for AraBERT |
| Model | — | Fine-tune AraBERT and evaluate |

Shared Spark settings and data paths live in [`src/config.py`](src/config.py).

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
  gold/      model-ready data
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
