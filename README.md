# AraBERT Sentiment on Arabic Hotel Reviews

Fine-tuning **AraBERT** to classify the sentiment of Arabic hotel reviews, with a **PySpark** medallion pipeline (bronze → silver → gold) preparing the data.

> Work in progress — currently at the bronze stage.

## Pipeline

| Stage | Notebook | What it does |
|---|---|---|
| Inspect | [`notebook/00_inspect.ipynb`](notebook/00_inspect.ipynb) | Inspect the raw file: encoding, delimiter, columns, types |
| Bronze | [`src/01_bronze.ipynb`](src/01_bronze.ipynb) | Load the raw data as-is with an explicit schema, add ingestion metadata, save as Parquet |
| Silver | — | Clean and normalise text, handle missing values, derive sentiment labels from ratings |
| Gold | — | Train / validation / test splits ready for AraBERT |
| Model | — | Fine-tune AraBERT and evaluate |

## Data

`balanced-reviews.txt`: 105,698 Arabic hotel reviews (tab-separated, UTF-16) with hotel name, rating, user type, room type, length of stay and review text.

The data is **not included** in this repository. Place the file at:

```
data/raw/balanced-reviews.txt
```

## Project layout

```
data/
  raw/       source file (not committed)
  bronze/    raw data as Parquet + ingestion metadata
  silver/    cleaned data
  gold/      model-ready splits
notebook/    exploration
src/         pipeline steps
```

## Setup

Requires Python 3.12 and Java 17+ (for PySpark).

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

**Windows note:** Spark's native Parquet writer needs Hadoop's `winutils.exe` and `HADOOP_HOME` to be set. Until that's configured, the bronze step writes Parquet through pandas/pyarrow instead.
