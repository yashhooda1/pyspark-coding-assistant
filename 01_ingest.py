# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Ingest — Bronze Layer
# MAGIC **Pipeline:** PySpark Coding Assistant Fine-Tuning  
# MAGIC **Layer:** Bronze — raw data landed as-is into Delta  
# MAGIC **Sources:**
# MAGIC - HuggingFace: `iamtarun/python_code_instructions_18k_alpaca` (Python/PySpark code instructions)
# MAGIC - HuggingFace: `TokenBender/code_instructions_122k_alpaca_style` (code Q&A, filtered to PySpark)
# MAGIC - Custom JSONL seed file with hand-curated PySpark patterns

# COMMAND ----------

# MAGIC %pip install datasets mlflow delta-spark --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import mlflow
import json
from datetime import datetime
from datasets import load_dataset
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit, current_timestamp, input_file_name, col
from pyspark.sql.types import StructType, StructField, StringType, TimestampType

spark = SparkSession.builder.getOrCreate()

# MLflow experiment — all 5 notebooks log to the same experiment
mlflow.set_experiment("/Shared/pyspark-coding-assistant-finetune")

# Config
CATALOG   = "main"
SCHEMA    = "pyspark_finetune"
BRONZE_TABLE = f"{CATALOG}.{SCHEMA}.bronze_raw"
VOLUME_PATH  = f"/Volumes/{CATALOG}/{SCHEMA}/raw_data"

# COMMAND ----------

# MAGIC %md ## 1 · Create catalog / schema / volume (idempotent)

# COMMAND ----------

spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.raw_data")

# COMMAND ----------

# MAGIC %md ## 2 · HuggingFace ingest — python_code_instructions_18k_alpaca

# COMMAND ----------

with mlflow.start_run(run_name="01_ingest", nested=True):

    # --- Source 1: python_code_instructions alpaca ---
    print("Loading HuggingFace dataset 1...")
    hf_ds1 = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train")
    
    rows_src1 = []
    for row in hf_ds1:
        rows_src1.append({
            "source":       "hf_python_alpaca",
            "instruction":  row.get("instruction", ""),
            "input":        row.get("input", ""),
            "output":       row.get("output", ""),
            "raw_json":     json.dumps(row),
            "ingested_at":  datetime.utcnow().isoformat(),
        })

    schema = StructType([
        StructField("source",      StringType(), True),
        StructField("instruction", StringType(), True),
        StructField("input",       StringType(), True),
        StructField("output",      StringType(), True),
        StructField("raw_json",    StringType(), True),
        StructField("ingested_at", StringType(), True),
    ])

    df_src1 = spark.createDataFrame(rows_src1, schema=schema)
    print(f"  Rows from source 1: {df_src1.count():,}")

    # --- Source 2: code_instructions_122k (filtered to PySpark-adjacent) ---
    print("Loading HuggingFace dataset 2...")
    hf_ds2 = load_dataset("TokenBender/code_instructions_122k_alpaca_style", split="train")

    PYSPARK_KEYWORDS = [
        "pyspark", "spark", "dataframe", "rdd", "sparkcontext",
        "sparksession", "delta", "databricks", "hadoop", "hive",
        "partitionby", "groupby", "agg", "withcolumn", "udf",
        "broadcast", "cache", "persist", "checkpoint",
    ]

    rows_src2 = []
    for row in hf_ds2:
        combined = (
            (row.get("instruction") or "") + " " +
            (row.get("output") or "")
        ).lower()
        if any(kw in combined for kw in PYSPARK_KEYWORDS):
            rows_src2.append({
                "source":       "hf_code_122k_pyspark_filtered",
                "instruction":  row.get("instruction", ""),
                "input":        row.get("input", ""),
                "output":       row.get("output", ""),
                "raw_json":     json.dumps(row),
                "ingested_at":  datetime.utcnow().isoformat(),
            })

    df_src2 = spark.createDataFrame(rows_src2, schema=schema)
    print(f"  Rows from source 2 (PySpark-filtered): {df_src2.count():,}")

    # --- Source 3: hand-curated PySpark seed patterns ---
    SEED_PATTERNS = [
        {
            "instruction": "Write a PySpark transformation to deduplicate a DataFrame on a composite key.",
            "input": "df with columns: user_id, event_date, event_type",
            "output": (
                "from pyspark.sql import Window\n"
                "from pyspark.sql.functions import row_number\n\n"
                "window = Window.partitionBy('user_id', 'event_date', 'event_type').orderBy('ingested_at'.desc())\n"
                "df_deduped = (\n"
                "    df\n"
                "    .withColumn('rn', row_number().over(window))\n"
                "    .filter(col('rn') == 1)\n"
                "    .drop('rn')\n"
                ")"
            ),
        },
        {
            "instruction": "How do I write a PySpark DataFrame to a Delta table with schema evolution enabled?",
            "input": "",
            "output": (
                "df.write\\\n"
                "  .format('delta')\\\n"
                "  .mode('append')\\\n"
                "  .option('mergeSchema', 'true')\\\n"
                "  .saveAsTable('catalog.schema.table')"
            ),
        },
        {
            "instruction": "Write a PySpark UDF that extracts domain from an email address column.",
            "input": "DataFrame with column: email (StringType)",
            "output": (
                "from pyspark.sql.functions import udf\n"
                "from pyspark.sql.types import StringType\n\n"
                "@udf(returnType=StringType())\n"
                "def extract_domain(email):\n"
                "    if email and '@' in email:\n"
                "        return email.split('@')[1].lower()\n"
                "    return None\n\n"
                "df = df.withColumn('domain', extract_domain(col('email')))"
            ),
        },
        {
            "instruction": "How do I perform a broadcast join in PySpark to avoid a shuffle?",
            "input": "large_df and small_df (< 10MB)",
            "output": (
                "from pyspark.sql.functions import broadcast\n\n"
                "# Hint Spark to broadcast the small table\n"
                "df_joined = large_df.join(\n"
                "    broadcast(small_df),\n"
                "    on='id',\n"
                "    how='left'\n"
                ")"
            ),
        },
        {
            "instruction": "Write a PySpark medallion pipeline that moves data from bronze to silver with quality checks.",
            "input": "bronze Delta table: catalog.schema.bronze_events",
            "output": (
                "from pyspark.sql.functions import col, count, when\n\n"
                "df_bronze = spark.read.table('catalog.schema.bronze_events')\n\n"
                "# Quality checks\n"
                "df_silver = (\n"
                "    df_bronze\n"
                "    .filter(col('event_id').isNotNull())\n"
                "    .filter(col('event_date') >= '2020-01-01')\n"
                "    .dropDuplicates(['event_id'])\n"
                "    .withColumn('is_valid', when(col('amount') > 0, True).otherwise(False))\n"
                ")\n\n"
                "# Write to silver with merge schema\n"
                "df_silver.write\\\n"
                "  .format('delta')\\\n"
                "  .mode('overwrite')\\\n"
                "  .option('overwriteSchema', 'true')\\\n"
                "  .saveAsTable('catalog.schema.silver_events')"
            ),
        },
        {
            "instruction": "How do I use window functions in PySpark to calculate a 7-day rolling average?",
            "input": "DataFrame with columns: store_id, sale_date, revenue",
            "output": (
                "from pyspark.sql import Window\n"
                "from pyspark.sql.functions import avg, col\n"
                "from pyspark.sql.types import DateType\n\n"
                "window_7d = (\n"
                "    Window\n"
                "    .partitionBy('store_id')\n"
                "    .orderBy(col('sale_date').cast('long'))\n"
                "    .rangeBetween(-6 * 86400, 0)  # 7 days in seconds\n"
                ")\n\n"
                "df = df.withColumn('revenue_7d_avg', avg('revenue').over(window_7d))"
            ),
        },
        {
            "instruction": "How do I read a partitioned Delta table efficiently in PySpark by filtering on the partition column?",
            "input": "Delta table partitioned by: event_date",
            "output": (
                "# Partition pruning — Spark reads only matching partitions\n"
                "df = (\n"
                "    spark.read\n"
                "    .format('delta')\n"
                "    .load('abfss://container@storage.dfs.core.windows.net/table')\n"
                "    .filter(col('event_date') >= '2024-01-01')  # partition filter\n"
                "    .filter(col('event_date') <  '2024-02-01')\n"
                ")\n"
                "# Confirm partition pruning with:\n"
                "# df.explain(True)"
            ),
        },
        {
            "instruction": "Write PySpark code to implement an SCD Type 2 merge using Delta Lake MERGE INTO.",
            "input": "target: catalog.schema.dim_customers (Delta), source: new_customers DataFrame",
            "output": (
                "from delta.tables import DeltaTable\n"
                "from pyspark.sql.functions import current_timestamp, lit\n\n"
                "target = DeltaTable.forName(spark, 'catalog.schema.dim_customers')\n\n"
                "target.alias('t').merge(\n"
                "    new_customers.alias('s'),\n"
                "    't.customer_id = s.customer_id AND t.is_current = true'\n"
                ").whenMatchedUpdate(\n"
                "    condition='t.email <> s.email',\n"
                "    set={\n"
                "        'is_current':  lit(False),\n"
                "        'end_date':    current_timestamp(),\n"
                "    }\n"
                ").whenNotMatchedInsert(\n"
                "    values={\n"
                "        'customer_id': 's.customer_id',\n"
                "        'email':       's.email',\n"
                "        'is_current':  lit(True),\n"
                "        'start_date':  current_timestamp(),\n"
                "        'end_date':    lit(None),\n"
                "    }\n"
                ").execute()"
            ),
        },
        {
            "instruction": "How do I cache a PySpark DataFrame and when should I use persist vs cache?",
            "input": "",
            "output": (
                "from pyspark import StorageLevel\n\n"
                "# cache() = MEMORY_AND_DISK (default)\n"
                "df.cache()\n\n"
                "# persist() lets you choose storage level explicitly\n"
                "df.persist(StorageLevel.MEMORY_ONLY)        # fast, evicts if OOM\n"
                "df.persist(StorageLevel.MEMORY_AND_DISK)    # spills to disk\n"
                "df.persist(StorageLevel.DISK_ONLY)          # low memory footprint\n\n"
                "# Trigger materialization\n"
                "df.count()\n\n"
                "# Always unpersist when done\n"
                "df.unpersist()"
            ),
        },
        {
            "instruction": "Write a PySpark job that reads Kafka stream and writes to a Delta table using Structured Streaming.",
            "input": "Kafka topic: raw_events, Delta target: catalog.schema.streaming_events",
            "output": (
                "df_stream = (\n"
                "    spark.readStream\n"
                "    .format('kafka')\n"
                "    .option('kafka.bootstrap.servers', 'broker:9092')\n"
                "    .option('subscribe', 'raw_events')\n"
                "    .option('startingOffsets', 'latest')\n"
                "    .load()\n"
                ")\n\n"
                "from pyspark.sql.functions import col, from_json\n"
                "from pyspark.sql.types import StructType, StructField, StringType, LongType\n\n"
                "event_schema = StructType([\n"
                "    StructField('event_id',   StringType(), True),\n"
                "    StructField('user_id',    StringType(), True),\n"
                "    StructField('event_type', StringType(), True),\n"
                "    StructField('ts',         LongType(),   True),\n"
                "])\n\n"
                "df_parsed = (\n"
                "    df_stream\n"
                "    .select(from_json(col('value').cast('string'), event_schema).alias('data'))\n"
                "    .select('data.*')\n"
                ")\n\n"
                "query = (\n"
                "    df_parsed.writeStream\n"
                "    .format('delta')\n"
                "    .outputMode('append')\n"
                "    .option('checkpointLocation', '/Volumes/main/schema/checkpoints/streaming_events')\n"
                "    .toTable('catalog.schema.streaming_events')\n"
                ")\n"
                "query.awaitTermination()"
            ),
        },
    ]

    rows_seed = [
        {
            "source":       "curated_pyspark_seed",
            "instruction":  p["instruction"],
            "input":        p["input"],
            "output":       p["output"],
            "raw_json":     json.dumps(p),
            "ingested_at":  datetime.utcnow().isoformat(),
        }
        for p in SEED_PATTERNS
    ]
    df_seed = spark.createDataFrame(rows_seed, schema=schema)
    print(f"  Rows from curated seed: {df_seed.count():,}")

    # --- Union all sources and write to Bronze ---
    df_bronze = df_src1.union(df_src2).union(df_seed)

    total = df_bronze.count()
    print(f"\nTotal bronze rows: {total:,}")

    (
        df_bronze.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .option("delta.enableChangeDataFeed", "true")
        .saveAsTable(BRONZE_TABLE)
    )

    # Log to MLflow
    mlflow.log_params({
        "bronze_table":        BRONZE_TABLE,
        "source_hf_alpaca":    len(rows_src1),
        "source_hf_122k":      len(rows_src2),
        "source_curated_seed": len(rows_seed),
        "total_bronze_rows":   total,
    })
    mlflow.log_metric("bronze_row_count", total)
    print(f"\n✓ Bronze table written: {BRONZE_TABLE} ({total:,} rows)")

# COMMAND ----------

# MAGIC %md ## 3 · Validate Bronze

# COMMAND ----------

display(spark.table(BRONZE_TABLE).groupBy("source").count().orderBy("count", ascending=False))
display(spark.table(BRONZE_TABLE).limit(5))
