# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · Clean — Silver Layer
# MAGIC **Pipeline:** PySpark Coding Assistant Fine-Tuning  
# MAGIC **Layer:** Silver — deduplicated, quality-filtered, PySpark-relevance scored  
# MAGIC **Key transforms:**
# MAGIC - Remove rows with empty instruction or output
# MAGIC - Deduplicate on instruction text
# MAGIC - Score PySpark relevance (keyword density)
# MAGIC - Filter out rows below relevance threshold
# MAGIC - Train / validation split (90 / 10)

# COMMAND ----------

# MAGIC %pip install mlflow --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import mlflow
import re
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, length, lower, trim, udf, monotonically_increasing_id,
    when, lit, hash, abs as spark_abs
)
from pyspark.sql.types import FloatType, BooleanType

spark = SparkSession.builder.getOrCreate()

mlflow.set_experiment("/Shared/pyspark-coding-assistant-finetune")

CATALOG       = "main"
SCHEMA        = "pyspark_finetune"
BRONZE_TABLE  = f"{CATALOG}.{SCHEMA}.bronze_raw"
SILVER_TABLE  = f"{CATALOG}.{SCHEMA}.silver_clean"

# COMMAND ----------

# MAGIC %md ## 1 · Load Bronze

# COMMAND ----------

df_bronze = spark.table(BRONZE_TABLE)
bronze_count = df_bronze.count()
print(f"Bronze rows: {bronze_count:,}")

# COMMAND ----------

# MAGIC %md ## 2 · Basic quality filters

# COMMAND ----------

df_filtered = (
    df_bronze
    # Non-null, non-empty instruction and output
    .filter(col("instruction").isNotNull() & (trim(col("instruction")) != ""))
    .filter(col("output").isNotNull()      & (trim(col("output"))      != ""))
    # Minimum length guards
    .filter(length(col("instruction")) >= 20)
    .filter(length(col("output"))      >= 30)
    # Max length guard — avoid context overflow during tokenization
    .filter(length(col("instruction")) <= 1500)
    .filter(length(col("output"))      <= 4000)
)

print(f"After basic filters: {df_filtered.count():,} rows")

# COMMAND ----------

# MAGIC %md ## 3 · PySpark relevance scoring

# COMMAND ----------

PYSPARK_KEYWORDS = [
    # Core API
    "pyspark", "sparksession", "sparkcontext", "dataframe", "rdd",
    "spark.read", "spark.sql", "writestream", "readstream",
    # Transforms
    "withcolumn", "groupby", "agg", "filter", "select", "join",
    "orderby", "partitionby", "window", "over", "alias",
    "withcolumnrenamed", "dropduplicates", "distinct", "union",
    "unionbyname", "crossjoin", "broadcast", "explode", "pivot",
    "unpivot", "fillna", "replace", "cast",
    # Functions module
    "from pyspark.sql.functions", "from pyspark.sql import",
    "from pyspark.sql.types", "from pyspark.sql.window",
    "col(", "lit(", "when(", "udf(", "struct(", "array(",
    "current_timestamp", "date_format", "to_date", "datediff",
    "row_number", "rank", "dense_rank", "lag", "lead",
    "sum(", "avg(", "count(", "max(", "min(", "collect_list",
    # Delta / Databricks
    "delta", "deltatables", "merge", "vacuum", "optimize",
    "zorder", "liquid clustering", "change data feed",
    "unity catalog", "databricks", "dbutils", "display(",
    # Performance
    "cache", "persist", "unpersist", "broadcast", "coalesce",
    "repartition", "checkpoint", "explain(", "storagelevel",
    # Streaming
    "readstream", "writestream", "trigger", "checkpointlocation",
    "foreachbatch", "outputmode",
    # MLlib
    "mllib", "pipeline", "vectorassembler", "stringindexer",
    "randomforestclassifier", "logisticregression",
]


@udf(returnType=FloatType())
def pyspark_relevance_score(instruction: str, output: str) -> float:
    """Return fraction of PySpark keywords found in instruction+output."""
    if not instruction and not output:
        return 0.0
    combined = ((instruction or "") + " " + (output or "")).lower()
    hits = sum(1 for kw in PYSPARK_KEYWORDS if kw in combined)
    return round(hits / len(PYSPARK_KEYWORDS), 4)


df_scored = df_filtered.withColumn(
    "pyspark_score",
    pyspark_relevance_score(col("instruction"), col("output"))
)

# COMMAND ----------

# Inspect score distribution
display(
    df_scored
    .groupBy(
        when(col("pyspark_score") == 0,      "0 — no match")
        .when(col("pyspark_score") < 0.05,   "0.01–0.04 — weak")
        .when(col("pyspark_score") < 0.15,   "0.05–0.14 — moderate")
        .otherwise(                           "0.15+ — strong")
        .alias("score_bucket")
    )
    .count()
    .orderBy("score_bucket")
)

# COMMAND ----------

# MAGIC %md ## 4 · Apply relevance threshold + deduplicate

# COMMAND ----------

# Curated seed rows always pass regardless of score
RELEVANCE_THRESHOLD = 0.02  # tunable — lower = more data, higher = higher purity

df_relevant = df_scored.filter(
    (col("pyspark_score") >= RELEVANCE_THRESHOLD) |
    (col("source") == "curated_pyspark_seed")
)
print(f"After relevance filter (>= {RELEVANCE_THRESHOLD}): {df_relevant.count():,}")

# Deduplicate on instruction hash — keeps first occurrence
df_deduped = df_relevant.dropDuplicates(["instruction"])
print(f"After deduplication: {df_deduped.count():,}")

# COMMAND ----------

# MAGIC %md ## 5 · Train / validation split

# COMMAND ----------

# Deterministic split using instruction hash — reproducible across runs
df_with_id = df_deduped.withColumn(
    "row_hash", spark_abs(hash(col("instruction"))) % 100
)

df_train = df_with_id.filter(col("row_hash") <  90).drop("row_hash")
df_val   = df_with_id.filter(col("row_hash") >= 90).drop("row_hash")

train_count = df_train.count()
val_count   = df_val.count()
print(f"Train: {train_count:,}  |  Val: {val_count:,}")

# COMMAND ----------

# MAGIC %md ## 6 · Write Silver table

# COMMAND ----------

with mlflow.start_run(run_name="02_clean", nested=True):

    df_silver = df_train.unionByName(
        df_val.withColumn("split", lit("val"))
    ).withColumn(
        "split",
        when(col("split").isNull(), lit("train")).otherwise(col("split"))
    )

    # Add split column before union
    df_train_tagged = df_train.withColumn("split", lit("train"))
    df_val_tagged   = df_val.withColumn("split",   lit("val"))
    df_silver_final = df_train_tagged.unionByName(df_val_tagged)

    (
        df_silver_final.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .option("delta.enableChangeDataFeed", "true")
        .partitionBy("split")
        .saveAsTable(SILVER_TABLE)
    )

    mlflow.log_params({
        "bronze_input_rows":     bronze_count,
        "relevance_threshold":   RELEVANCE_THRESHOLD,
        "silver_table":          SILVER_TABLE,
    })
    mlflow.log_metrics({
        "silver_train_count": train_count,
        "silver_val_count":   val_count,
        "silver_total_count": train_count + val_count,
        "filter_retention_pct": round((train_count + val_count) / bronze_count * 100, 2),
    })

    print(f"\n✓ Silver table written: {SILVER_TABLE}")
    print(f"  Train: {train_count:,} | Val: {val_count:,}")

# COMMAND ----------

# MAGIC %md ## 7 · Validate Silver

# COMMAND ----------

display(spark.table(SILVER_TABLE).groupBy("split", "source").count().orderBy("split", "count", ascending=False))
display(spark.table(SILVER_TABLE).filter(col("split") == "train").orderBy(col("pyspark_score").desc()).limit(5))
