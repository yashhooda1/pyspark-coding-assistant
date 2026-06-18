# Databricks notebook source
# MAGIC %md
# MAGIC # 05 · Evaluate + Serve
# MAGIC **Pipeline:** PySpark Coding Assistant Fine-Tuning  
# MAGIC **Evaluation:** MLflow LLM evaluate — ROUGE, exact match, LLM-as-judge (correctness)  
# MAGIC **Serving:** Mosaic AI Model Serving endpoint (REST API)

# COMMAND ----------

# MAGIC %pip install mlflow rouge-score evaluate --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import mlflow
import mlflow.evaluate
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

spark = SparkSession.builder.getOrCreate()

mlflow.set_experiment("/Shared/pyspark-coding-assistant-finetune")

CATALOG        = "main"
SCHEMA         = "pyspark_finetune"
GOLD_TABLE     = f"{CATALOG}.{SCHEMA}.gold_training"
MODEL_NAME_UC  = f"{CATALOG}.{SCHEMA}.pyspark_coding_assistant"

# COMMAND ----------

# MAGIC %md ## 1 · Build evaluation dataset from Gold val split

# COMMAND ----------

df_val = (
    spark.table(GOLD_TABLE)
    .filter(col("split") == "val")
    .select("instruction", "output", "pyspark_score")
    .orderBy(col("pyspark_score").desc())   # evaluate on highest-quality examples first
    .limit(100)                              # 100 examples is enough for eval
)

eval_df = df_val.toPandas()
eval_df = eval_df.rename(columns={
    "instruction": "inputs",
    "output":      "ground_truth",
})

print(f"Evaluation set size: {len(eval_df)}")
print(f"\nSample input:\n{eval_df['inputs'].iloc[0]}")
print(f"\nExpected output:\n{eval_df['ground_truth'].iloc[0][:300]}")

# COMMAND ----------

# MAGIC %md ## 2 · Load the fine-tuned model from Unity Catalog

# COMMAND ----------

# Load latest registered version
client = mlflow.tracking.MlflowClient()
latest_version = client.get_registered_model(MODEL_NAME_UC).latest_versions[0]
model_uri = f"models:/{MODEL_NAME_UC}/{latest_version.version}"

print(f"Evaluating model: {MODEL_NAME_UC} v{latest_version.version}")
print(f"URI: {model_uri}")

# COMMAND ----------

# MAGIC %md ## 3 · Define judge model for MLflow LLM evaluate

# COMMAND ----------

# MLflow will use this endpoint as the judge for code correctness scoring
# On Databricks, you can use the built-in Foundation Model API
JUDGE_MODEL_ENDPOINT = "databricks-dbrx-instruct"  # or "databricks-meta-llama-3-1-70b-instruct"

# Custom judge prompt — tells the judge what "correct" means for PySpark code
JUDGE_PROMPT = """
You are an expert PySpark engineer evaluating a code generation model.

Question asked: {inputs}

Expected answer: {ground_truth}

Model output: {output}

Score the model output from 1–5:
5 = Correct, idiomatic PySpark, uses best practices (DataFrame API, proper imports)
4 = Correct PySpark but missing minor best practices or comments
3 = Mostly correct but has small bugs or uses deprecated APIs
2 = Wrong approach but shows PySpark knowledge
1 = Incorrect or uses non-PySpark (pandas, plain Python)

Respond with ONLY the integer score (1-5) and nothing else.
""".strip()

# COMMAND ----------

# MAGIC %md ## 4 · Run MLflow LLM evaluate

# COMMAND ----------

# Build a simple predict function wrapping the loaded model
# (In production you'd call the Mosaic AI serving endpoint instead)
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from peft import PeftModel

BASE_MODEL = latest_version.tags.get("base_model", "mistralai/Mistral-7B-Instruct-v0.3")
ADAPTER_DIR = "/local_disk0/pyspark-assistant-lora"

# Re-download adapter from MLflow if not on local disk
if not __import__("os").path.exists(ADAPTER_DIR):
    mlflow.artifacts.download_artifacts(
        artifact_uri=f"models:/{MODEL_NAME_UC}/{latest_version.version}",
        dst_path=ADAPTER_DIR
    )

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, quantization_config=bnb_config, device_map="auto"
)
ft_model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
ft_model.eval()

gen_pipeline = pipeline(
    "text-generation",
    model=ft_model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    temperature=0.1,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id,
)

SYSTEM = (
    "You are an expert PySpark and Databricks engineer. "
    "Write clean, idiomatic PySpark code with concise inline comments."
)

def predict(inputs_df: pd.DataFrame) -> pd.Series:
    """Called by mlflow.evaluate for each row."""
    responses = []
    for instruction in inputs_df["inputs"]:
        prompt = f"<s>[INST] {SYSTEM}\n\n{instruction} [/INST]"
        out = gen_pipeline(prompt)[0]["generated_text"]
        # Strip the prompt prefix from generated output
        response = out[len(prompt):].strip()
        responses.append(response)
    return pd.Series(responses)

# COMMAND ----------

with mlflow.start_run(run_name="05_evaluate") as run:

    mlflow.log_params({
        "model_name":    MODEL_NAME_UC,
        "model_version": latest_version.version,
        "eval_size":     len(eval_df),
        "judge_model":   JUDGE_MODEL_ENDPOINT,
    })

    results = mlflow.evaluate(
        model=predict,
        data=eval_df,
        targets="ground_truth",
        model_type="text",
        evaluators="default",
        extra_metrics=[
            mlflow.metrics.genai.make_genai_metric(
                name="pyspark_correctness",
                definition=(
                    "Score 1-5: how correct and idiomatic is the PySpark code output "
                    "compared to the expected answer?"
                ),
                grading_prompt=JUDGE_PROMPT,
                examples=[],
                model=f"endpoints:/{JUDGE_MODEL_ENDPOINT}",
                parameters={"temperature": 0.0},
                aggregations=["mean", "variance"],
                greater_is_better=True,
            )
        ],
    )

    # Log key metrics
    metrics = results.metrics
    print("\n=== Evaluation Results ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
        mlflow.log_metric(k, v)

    correctness_mean = metrics.get("pyspark_correctness/mean", 0)

    # Pass/fail gate — model must score >= 3.5 / 5 to promote to Champion
    PASS_THRESHOLD = 3.5
    passed = correctness_mean >= PASS_THRESHOLD
    mlflow.log_metric("eval_passed", int(passed))
    print(f"\n  PySpark correctness (mean): {correctness_mean:.2f} / 5.0")
    print(f"  Gate threshold:             {PASS_THRESHOLD}")
    print(f"  Result:                     {'✓ PASSED' if passed else '✗ FAILED'}")

    if passed:
        # Promote to Champion alias in Unity Catalog model registry
        client = mlflow.tracking.MlflowClient()
        client.set_registered_model_alias(
            MODEL_NAME_UC, "Champion", latest_version.version
        )
        print(f"\n✓ Model v{latest_version.version} promoted to Champion alias")

# COMMAND ----------

# MAGIC %md ## 5 · Display per-example eval table

# COMMAND ----------

display(results.tables["eval_results_table"])

# COMMAND ----------

# MAGIC %md ## 6 · Deploy to Mosaic AI Model Serving

# COMMAND ----------

# MAGIC %md
# MAGIC ### Deploy via Databricks UI (recommended for portfolio)
# MAGIC 1. Go to **Serving** → **Create serving endpoint**
# MAGIC 2. Entity: `main.pyspark_finetune.pyspark_coding_assistant`
# MAGIC 3. Version: Champion alias
# MAGIC 4. Compute: **GPU Small** (1× A10, pay-per-token)
# MAGIC 5. Enable **AI Gateway** for rate limiting + logging
# MAGIC
# MAGIC ### Or deploy programmatically via REST API:

# COMMAND ----------

import requests, json

DATABRICKS_HOST  = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
DATABRICKS_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

endpoint_config = {
    "name": "pyspark-coding-assistant",
    "config": {
        "served_entities": [
            {
                "entity_name":    MODEL_NAME_UC,
                "entity_version": str(latest_version.version),
                "workload_size":  "Small",
                "scale_to_zero_enabled": True,
            }
        ],
        "traffic_config": {
            "routes": [{"served_model_name": "pyspark_coding_assistant", "traffic_percentage": 100}]
        },
    },
}

response = requests.post(
    f"{DATABRICKS_HOST}/api/2.0/serving-endpoints",
    headers={"Authorization": f"Bearer {DATABRICKS_TOKEN}"},
    json=endpoint_config,
)
print(f"Endpoint creation status: {response.status_code}")
print(json.dumps(response.json(), indent=2))

# COMMAND ----------

# MAGIC %md ## 7 · Test the live serving endpoint

# COMMAND ----------

def query_endpoint(instruction: str) -> str:
    payload = {
        "inputs": {
            "prompt": [
                f"<s>[INST] You are an expert PySpark engineer. {instruction} [/INST]"
            ]
        },
        "params": {"max_new_tokens": 256, "temperature": 0.1},
    }
    resp = requests.post(
        f"{DATABRICKS_HOST}/serving-endpoints/pyspark-coding-assistant/invocations",
        headers={"Authorization": f"Bearer {DATABRICKS_TOKEN}"},
        json=payload,
    )
    return resp.json()["predictions"][0]


# Sample queries to showcase in portfolio README
test_questions = [
    "Write PySpark code to read a Delta table and filter rows where revenue > 10000.",
    "How do I do a left anti join in PySpark to find rows in df_a not in df_b?",
    "Write a PySpark UDF to parse JSON strings stored in a column.",
]

for q in test_questions:
    print(f"\nQ: {q}")
    print(f"A: {query_endpoint(q)[:400]}")
    print("─" * 60)
