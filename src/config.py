# config.py - shared settings for every pipeline step
import os
from pathlib import Path

from pyspark.sql import SparkSession

# project root = one folder up from this file (src/ -> AraBERT/)
ROOT = Path(__file__).resolve().parents[1]

# ---------- Windows: point Spark at winutils.exe + hadoop.dll ----------
# Spark needs these to WRITE files on Windows. They live inside the project
# (tools/hadoop/bin), so no system-wide setup is needed.
# Must run BEFORE the first SparkSession is created in the kernel.
HADOOP_HOME = ROOT / "tools" / "hadoop"
os.environ["HADOOP_HOME"] = str(HADOOP_HOME)
os.environ["PATH"] = str(HADOOP_HOME / "bin") + os.pathsep + os.environ.get("PATH", "")

# ---------- paths (one place -> change once, every notebook updates) ----------
RAW_PATH        = str(ROOT / "data" / "raw" / "balanced-reviews.txt")
BRONZE_PATH     = str(ROOT / "data" / "bronze" / "hard_reviews")
SILVER_PATH     = str(ROOT / "data" / "silver" / "hard_reviews")
QUARANTINE_PATH = str(ROOT / "data" / "quarantine" / "hard_reviews")
GOLD_PATH       = str(ROOT / "data" / "gold")


def get_spark(app_name: str) -> SparkSession:
    """Create (or reuse) the Spark session with the same settings everywhere."""
    return (SparkSession.builder
            .appName(app_name)
            .master("local[*]")
            .getOrCreate())
