# Databricks notebook source
# MAGIC %md
# MAGIC # 04 · Train — LoRA Fine-Tuning + MLflow Tracking
# MAGIC **Pipeline:** PySpark Coding Assistant Fine-Tuning  
# MAGIC **Model:** Mistral-7B-Instruct-v0.3 (or Llama-3-8B-Instruct)  
# MAGIC **Method:** LoRA (Low-Rank Adaptation) via HuggingFace PEFT  
# MAGIC **Tracking:** MLflow — params, metrics, loss curves, model artifact  
# MAGIC **Registry:** Unity Catalog model registry  
# MAGIC
# MAGIC **Cluster requirement:** Single-node GPU cluster  
# MAGIC - Runtime: Databricks ML Runtime 15.x GPU  
# MAGIC - Node type: `g5.2xlarge` (AWS) or `Standard_NC6s_v3` (Azure) — 1× A10G/V100

# COMMAND ----------

# MAGIC %pip install peft accelerate bitsandbytes trl transformers==4.40.0 mlflow --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os
import mlflow
import mlflow.pytorch
import torch
from datetime import datetime

from datasets import Dataset
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from trl import SFTTrainer

spark = SparkSession.builder.getOrCreate()

mlflow.set_experiment("/Shared/pyspark-coding-assistant-finetune")

# COMMAND ----------

# MAGIC %md ## Config

# COMMAND ----------

# ── Model ──────────────────────────────────────────────────────────────────────
# Option A: Mistral 7B (recommended, strong code performance)
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"

# Option B: Llama 3 8B — swap in if you prefer Meta's ecosystem
# BASE_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"

# ── Data ───────────────────────────────────────────────────────────────────────
CATALOG      = "main"
SCHEMA       = "pyspark_finetune"
GOLD_TABLE   = f"{CATALOG}.{SCHEMA}.gold_training"

# ── Output ─────────────────────────────────────────────────────────────────────
OUTPUT_DIR       = "/local_disk0/pyspark-assistant-lora"
MODEL_NAME_UC    = f"{CATALOG}.{SCHEMA}.pyspark_coding_assistant"

# ── LoRA hyperparameters ───────────────────────────────────────────────────────
LORA_R          = 16      # rank — higher = more params, better quality, slower
LORA_ALPHA      = 32      # scaling factor — usually 2× rank
LORA_DROPOUT    = 0.05
LORA_TARGET_MODULES = [   # Mistral/Llama attention projection layers
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# ── Training hyperparameters ───────────────────────────────────────────────────
NUM_EPOCHS       = 3
BATCH_SIZE       = 4      # per-device batch size — reduce to 2 if OOM
GRAD_ACCUM       = 4      # effective batch = BATCH_SIZE × GRAD_ACCUM = 16
LEARNING_RATE    = 2e-4
WARMUP_RATIO     = 0.03
MAX_SEQ_LENGTH   = 2048
LOGGING_STEPS    = 10
SAVE_STEPS       = 100

print(f"Base model:       {BASE_MODEL}")
print(f"LoRA rank:        {LORA_R}")
print(f"Effective batch:  {BATCH_SIZE * GRAD_ACCUM}")
print(f"CUDA available:   {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU:              {torch.cuda.get_device_name(0)}")
    print(f"VRAM:             {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# COMMAND ----------

# MAGIC %md ## 1 · Load Gold data from Delta into HuggingFace Dataset

# COMMAND ----------

df_train = spark.table(GOLD_TABLE).filter(col("split") == "train").select("text")
df_val   = spark.table(GOLD_TABLE).filter(col("split") == "val").select("text")

# Convert to HuggingFace Dataset (collected to driver — fits in RAM at ~10k rows)
train_dataset = Dataset.from_pandas(df_train.toPandas())
val_dataset   = Dataset.from_pandas(df_val.toPandas())

print(f"Train examples: {len(train_dataset):,}")
print(f"Val examples:   {len(val_dataset):,}")
print(f"\nSample:\n{train_dataset[0]['text'][:400]}")

# COMMAND ----------

# MAGIC %md ## 2 · Load base model with 4-bit quantization (QLoRA)

# COMMAND ----------

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)
model = prepare_model_for_kbit_training(model)

print(f"Model loaded: {BASE_MODEL}")
print(f"Parameters: {model.num_parameters():,}")

# COMMAND ----------

# MAGIC %md ## 3 · Apply LoRA adapters

# COMMAND ----------

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=LORA_TARGET_MODULES,
    bias="none",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Expected output: ~0.6% trainable parameters — this is the power of LoRA

# COMMAND ----------

# MAGIC %md ## 4 · Training arguments

# COMMAND ----------

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LEARNING_RATE,
    warmup_ratio=WARMUP_RATIO,
    lr_scheduler_type="cosine",
    optim="paged_adamw_32bit",
    fp16=False,
    bf16=True,
    logging_steps=LOGGING_STEPS,
    evaluation_strategy="steps",
    eval_steps=SAVE_STEPS,
    save_strategy="steps",
    save_steps=SAVE_STEPS,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="mlflow",    # MLflow auto-captures all training metrics
    run_name=f"pyspark-assistant-lora-r{LORA_R}-{datetime.now().strftime('%Y%m%d-%H%M')}",
    dataloader_num_workers=4,
    group_by_length=True,  # pack similar-length sequences → faster training
)

# COMMAND ----------

# MAGIC %md ## 5 · Custom MLflow callback — log GPU memory each epoch

# COMMAND ----------

class GPUMemoryCallback(TrainerCallback):
    """Log peak GPU memory usage to MLflow after each epoch."""
    def on_epoch_end(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            peak_gb = torch.cuda.max_memory_allocated() / 1e9
            mlflow.log_metric("gpu_peak_memory_gb", peak_gb, step=state.global_step)
            torch.cuda.reset_peak_memory_stats()
            print(f"  [Epoch {state.epoch:.0f}] Peak GPU memory: {peak_gb:.2f} GB")

# COMMAND ----------

# MAGIC %md ## 6 · Fine-tune with SFTTrainer

# COMMAND ----------

with mlflow.start_run(run_name="04_train") as run:

    # Log all hyperparams explicitly in addition to HF auto-logging
    mlflow.log_params({
        "base_model":       BASE_MODEL,
        "lora_r":           LORA_R,
        "lora_alpha":       LORA_ALPHA,
        "lora_dropout":     LORA_DROPOUT,
        "num_epochs":       NUM_EPOCHS,
        "batch_size":       BATCH_SIZE,
        "grad_accum":       GRAD_ACCUM,
        "effective_batch":  BATCH_SIZE * GRAD_ACCUM,
        "learning_rate":    LEARNING_RATE,
        "warmup_ratio":     WARMUP_RATIO,
        "max_seq_length":   MAX_SEQ_LENGTH,
        "gold_table":       GOLD_TABLE,
        "train_examples":   len(train_dataset),
        "val_examples":     len(val_dataset),
        "target_modules":   ",".join(LORA_TARGET_MODULES),
    })

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        args=training_args,
        callbacks=[GPUMemoryCallback()],
        packing=True,   # pack multiple short samples into one sequence → GPU efficiency
    )

    print("Starting fine-tuning...")
    trainer.train()

    # Final eval metrics
    final_metrics = trainer.evaluate()
    mlflow.log_metrics({
        "final_eval_loss":       final_metrics["eval_loss"],
        "final_eval_perplexity": 2 ** final_metrics["eval_loss"],
    })
    print(f"\n✓ Training complete")
    print(f"  Final eval loss:       {final_metrics['eval_loss']:.4f}")
    print(f"  Final eval perplexity: {2 ** final_metrics['eval_loss']:.2f}")

    # COMMAND ----------

    # MAGIC %md ## 7 · Save adapter + register to Unity Catalog model registry

    # COMMAND ----------

    # Save LoRA adapter weights locally
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"  Adapter saved: {OUTPUT_DIR}")

    # Log adapter as MLflow artifact
    mlflow.log_artifacts(OUTPUT_DIR, artifact_path="lora_adapter")

    # Register to Unity Catalog model registry
    model_uri = f"runs:/{run.info.run_id}/lora_adapter"
    registered = mlflow.register_model(
        model_uri=model_uri,
        name=MODEL_NAME_UC,
    )

    # Add descriptive tags to the registered model version
    client = mlflow.tracking.MlflowClient()
    client.set_registered_model_tag(MODEL_NAME_UC, "task",        "pyspark-code-generation")
    client.set_registered_model_tag(MODEL_NAME_UC, "base_model",  BASE_MODEL)
    client.set_registered_model_tag(MODEL_NAME_UC, "method",      "LoRA QLoRA")
    client.set_registered_model_tag(MODEL_NAME_UC, "train_rows",  str(len(train_dataset)))

    print(f"\n✓ Model registered: {MODEL_NAME_UC} v{registered.version}")
    print(f"  Run ID: {run.info.run_id}")

# COMMAND ----------

# MAGIC %md ## 8 · Quick inference test

# COMMAND ----------

from peft import PeftModel

# Load adapter on top of quantized base for inference test
inf_model = PeftModel.from_pretrained(model, OUTPUT_DIR)
inf_model.eval()

test_prompt = (
    "<s>[INST] You are an expert PySpark and Databricks engineer. "
    "When asked a question, write clean idiomatic PySpark code.\n\n"
    "How do I calculate a 30-day rolling sum of sales per store in PySpark? [/INST]"
)

inputs = tokenizer(test_prompt, return_tensors="pt").to("cuda")
with torch.no_grad():
    output_ids = inf_model.generate(
        **inputs,
        max_new_tokens=300,
        temperature=0.2,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )

response = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print("=== Model response ===")
print(response)
