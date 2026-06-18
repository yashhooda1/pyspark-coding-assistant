# Databricks notebook source
# MAGIC %md
# MAGIC # 03 · Prepare — Gold Layer
# MAGIC **Pipeline:** PySpark Coding Assistant Fine-Tuning  
# MAGIC **Layer:** Gold — model-ready prompt/completion pairs  
# MAGIC **Output format:** Mistral Instruct `[INST]` / Llama 3 `<|user|>` chat template  
# MAGIC **Note:** Toggle `MODEL_FAMILY` below to switch between Mistral and Llama 3 formatting.

# COMMAND ----------

# MAGIC %pip install mlflow tiktoken --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import mlflow
import tiktoken
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, length, lit, current_timestamp
from pyspark.sql.types import StringType, IntegerType

spark = SparkSession.builder.getOrCreate()

mlflow.set_experiment("/Shared/pyspark-coding-assistant-finetune")

CATALOG       = "main"
SCHEMA        = "pyspark_finetune"
SILVER_TABLE  = f"{CATALOG}.{SCHEMA}.silver_clean"
GOLD_TABLE    = f"{CATALOG}.{SCHEMA}.gold_training"

# Toggle: "mistral" | "llama3"
MODEL_FAMILY  = "mistral"

# COMMAND ----------

# MAGIC %md ## 1 · Prompt templates

# COMMAND ----------

SYSTEM_PROMPT = (
    "You are an expert PySpark and Databricks engineer. "
    "When asked a question or given a task, write clean, idiomatic PySpark code "
    "with concise inline comments. Always prefer DataFrame API over RDD API. "
    "Use Delta Lake best practices where relevant."
)


def format_mistral(instruction: str, input_ctx: str, output: str) -> str:
    """Mistral Instruct v0.3 format."""
    user_content = instruction
    if input_ctx and input_ctx.strip():
        user_content = f"{instruction}\n\nContext:\n{input_ctx.strip()}"
    return (
        f"<s>[INST] {SYSTEM_PROMPT}\n\n{user_content} [/INST] "
        f"{output} </s>"
    )


def format_llama3(instruction: str, input_ctx: str, output: str) -> str:
    """Llama 3 chat template format."""
    user_content = instruction
    if input_ctx and input_ctx.strip():
        user_content = f"{instruction}\n\nContext:\n{input_ctx.strip()}"
    return (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n\n"
        f"{SYSTEM_PROMPT}<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_content}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
        f"{output}<|eot_id|>"
    )


FORMAT_FN = format_mistral if MODEL_FAMILY == "mistral" else format_llama3

# Print a sample to verify format
sample_text = FORMAT_FN(
    "How do I read a Delta table in PySpark?",
    "table name: catalog.schema.events",
    "df = spark.read.table('catalog.schema.events')"
)
print("=== Sample formatted row ===")
print(sample_text[:600])

# COMMAND ----------

# MAGIC %md ## 2 · Token counting UDF

# COMMAND ----------

# Use cl100k_base as a proxy — close enough for Mistral/Llama token estimates
_enc = tiktoken.get_encoding("cl100k_base")


@udf(returnType=IntegerType())
def count_tokens(text: str) -> int:
    if not text:
        return 0
    try:
        return len(_enc.encode(text))
    except Exception:
        return 0


# COMMAND ----------

# MAGIC %md ## 3 · Format Silver → Gold

# COMMAND ----------

@udf(returnType=StringType())
def format_row(instruction: str, input_ctx: str, output: str) -> str:
    fn = format_mistral if MODEL_FAMILY == "mistral" else format_llama3
    return fn(instruction or "", input_ctx or "", output or "")


df_silver = spark.table(SILVER_TABLE)

df_gold = (
    df_silver
    .withColumn(
        "text",
        format_row(col("instruction"), col("input"), col("output"))
    )
    .withColumn("token_count", count_tokens(col("text")))
    # Filter out rows that exceed model context window (Mistral/Llama 3 = 8192 tokens)
    .filter(col("token_count") <= 3072)   # conservative: leave room for generation
    .filter(col("token_count") >= 50)     # remove trivially short examples
    .select(
        col("text"),
        col("token_count"),
        col("instruction"),
        col("output"),
        col("source"),
        col("pyspark_score"),
        col("split"),
        lit(MODEL_FAMILY).alias("model_family"),
        current_timestamp().alias("prepared_at"),
    )
)

train_count = df_gold.filter(col("split") == "train").count()
val_count   = df_gold.filter(col("split") == "val").count()
print(f"Gold rows — Train: {train_count:,}  |  Val: {val_count:,}")

# COMMAND ----------

# MAGIC %md ## 4 · Token distribution check

# COMMAND ----------

from pyspark.sql.functions import percentile_approx, avg as spark_avg, max as spark_max

display(
    df_gold.agg(
        spark_avg("token_count").alias("avg_tokens"),
        percentile_approx("token_count", 0.5).alias("median_tokens"),
        percentile_approx("token_count", 0.95).alias("p95_tokens"),
        spark_max("token_count").alias("max_tokens"),
    )
)

# COMMAND ----------

# MAGIC %md ## 5 · Write Gold table

# COMMAND ----------

with mlflow.start_run(run_name="03_prepare", nested=True):

    (
        df_gold.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .partitionBy("split")
        .saveAsTable(GOLD_TABLE)
    )

    mlflow.log_params({
        "gold_table":     GOLD_TABLE,
        "model_family":   MODEL_FAMILY,
        "max_tokens":     3072,
    })
    mlflow.log_metrics({
        "gold_train_count": train_count,
        "gold_val_count":   val_count,
    })

    # Log sample rows as an artifact
    sample_rows = df_gold.filter(col("split") == "train").limit(10).toPandas()
    sample_path = "/tmp/gold_sample.csv"
    sample_rows.to_csv(sample_path, index=False)
    mlflow.log_artifact(sample_path, "gold_samples")

    print(f"\n✓ Gold table written: {GOLD_TABLE}")
    print(f"  Train: {train_count:,} | Val: {val_count:,} | Model: {MODEL_FAMILY}")

# COMMAND ----------

# MAGIC %md ## 6 · Validate Gold — spot check formatted rows

# COMMAND ----------

for row in spark.table(GOLD_TABLE).filter(col("split") == "train").limit(3).collect():
    print("─" * 80)
    print(row["text"][:800])
    print(f"  tokens: {row['token_count']}  |  score: {row['pyspark_score']}")
