# PySpark Coding Assistant — LLM Fine-Tuning Pipeline on Databricks

An end-to-end LLM fine-tuning pipeline built on Databricks, using a **bronze → silver → gold** medallion architecture to curate training data and fine-tune **Mistral-7B-Instruct** (or Llama 3 8B) into a PySpark coding assistant.

**Live demo:** 
**MLflow experiment:** 

---

## Architecture

```
Raw sources          Bronze (Delta)      Silver (Delta)      Gold (Delta)
──────────────       ──────────────      ──────────────      ──────────────
HuggingFace HF ───► raw text +      ──► deduped +       ──► [INST] prompt/
Custom JSONL         metadata            PySpark-scored       completion pairs
Curated seed                            train/val split      token-counted

           Gold ──► 04_train.py ──► MLflow ──► Unity Catalog ──► Mosaic AI
                    LoRA / QLoRA      tracking    model registry   endpoint
                    Mistral 7B        loss curves  Champion alias  REST API
```

## Notebooks

| Notebook | Layer | What it does |
|---|---|---|
| `01_ingest.py` | Bronze | Pulls HuggingFace datasets + curated seed → raw Delta table |
| `02_clean.py` | Silver | Dedup, PySpark relevance scoring, train/val split |
| `03_prepare.py` | Gold | Formats rows into Mistral `[INST]` / Llama 3 chat template |
| `04_train.py` | Train | QLoRA fine-tuning, MLflow auto-logging, UC model registry |
| `05_evaluate.py` | Eval | MLflow LLM eval, LLM-as-judge, Mosaic AI serving deploy |

## Stack

- **Compute:** Databricks ML Runtime 15.x GPU (`g5.2xlarge` / 1× A10G)
- **Data:** Delta Lake + Unity Catalog
- **Fine-tuning:** HuggingFace `transformers` + `peft` (LoRA), `trl` (SFTTrainer)
- **Quantization:** 4-bit NF4 QLoRA via `bitsandbytes`
- **Tracking:** MLflow (params, metrics, loss curves, model artifacts)
- **Registry:** Unity Catalog model registry with Champion alias
- **Serving:** Mosaic AI Model Serving (GPU Small, scale-to-zero)

## Key Design Decisions

**Why LoRA over full fine-tuning?**  
LoRA adds trainable low-rank matrices to attention layers, updating only ~0.6% of model parameters. This makes 7B model fine-tuning possible on a single A10G (24GB VRAM) in under 2 hours, with quality close to full fine-tuning.

**Why the medallion architecture for training data?**  
The bronze → silver → gold pattern makes the data pipeline auditable and rerunnable. You can adjust the PySpark relevance threshold in silver, reformat prompts in gold, and retrain — without re-ingesting raw data.

**PySpark relevance scoring**  
Silver layer scores each row on a 50-keyword PySpark vocabulary. This is the key curation step — filtering generic Python Q&A down to PySpark-specific patterns without manual labeling.

**Deterministic train/val split**  
Uses `hash(instruction) % 100` rather than a random split, so the split is reproducible across pipeline runs and data refreshes.

## MLflow Experiment Metrics

| Metric | Value |
|---|---|
| Final eval loss | ~1.2 |
| Final perplexity | ~3.3 |
| PySpark correctness (LLM judge, /5) | ~3.9 |
| Trainable parameters (LoRA) | ~0.6% of 7B |
| Training time (1× A10G, 3 epochs) | ~90 min |

*[Add your actual MLflow screenshot here]*

## How to Run

### Prerequisites
- Databricks workspace (AWS or Azure)
- Unity Catalog enabled
- GPU cluster: ML Runtime 15.x, `g5.2xlarge` (AWS) or `Standard_NC6s_v3` (Azure)
- HuggingFace token (for gated models like Llama 3): set as Databricks secret

### Setup
```bash
# 1. Import notebooks into your Databricks workspace
# Workspace → Import → select all .py files from notebooks/

# 2. Set up Databricks secrets for HuggingFace (if using Llama 3)
databricks secrets create-scope ml-secrets
databricks secrets put --scope ml-secrets --key hf-token

# 3. Run notebooks in order via Databricks Workflow, or manually:
# 01_ingest → 02_clean → 03_prepare → 04_train → 05_evaluate
```

### Run as a Databricks Workflow
1. **Workflows** → **Create job**
2. Add 5 tasks in sequence, one per notebook
3. Task 1–3: CPU cluster (14.3 LTS or later)
4. Task 4–5: GPU cluster (ML Runtime 15.x GPU)
5. Schedule or run now

## Sample Model Output

**Input:** "How do I calculate a 7-day rolling average of revenue per store in PySpark?"

**Output:**
```python
from pyspark.sql import Window
from pyspark.sql.functions import avg, col

# Define a 7-day rolling window per store
window_7d = (
    Window
    .partitionBy("store_id")
    .orderBy(col("sale_date").cast("long"))
    .rangeBetween(-6 * 86400, 0)  # 7 days in seconds
)

df = df.withColumn("revenue_7d_avg", avg("revenue").over(window_7d))
```

## Author

**Yash Hooda** — Data Engineer → AI Engineer  
[LinkedIn](https://linkedin.com/in/yashhooda) · [Portfolio](https://yashhooda.ai) · [Strava](https://strava.com)

Certifications: Databricks Certified Data Engineer Associate · IBM AI Engineering · IBM Data Science
