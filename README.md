# Cloud-Based Large Language Model Deployment

**Replication data and code for:**
> Ploskonka, M. (2025). *Cloud-Based Large Language Model Deployment: A Comparative Analysis of Serverless and Bring-Your-Own-Container Architectures.* Manuscript submitted to *Acta Informatica Pragensia*.

---

## Overview

This repository contains the evaluation scripts, benchmark data, and model outputs supporting the empirical study comparing **serverless** and **Bring-Your-Own-Container (BYOC)** LLM deployment architectures on major cloud platforms (AWS and Azure).

The study evaluates **32 large language models** ranging from 1B to 70B+ parameters using the [Belebele](https://github.com/facebookresearch/belebele) multilingual reading comprehension benchmark (122 language variants), analysing accuracy, cost, and latency trade-offs across deployment architectures.

**Key finding:** Model performance scales logarithmically with parameter count and cost, with diminishing returns beyond ~30B parameters — enabling approximately 60% cost reduction by targeting mid-sized models without meaningful accuracy loss.

> **Note:** This study evaluates models in a **direct inference** setting — no retrieval-augmented generation (RAG) is used. Each benchmark question is answered by the model from the prompt alone. RAG-based evaluation is a separate line of research; see Ploskonka & Kliegr (2025), *Beyond Scale: Performance Plateau in Retrieval-Augmented Generation for Discriminative Question Answering Tasks*, submitted to ACM Transactions on Intelligent Systems and Technology.

---

## Repository Contents

| File | Description |
|------|-------------|
| `code_example-checkpoint.ipynb` | Example evaluation notebook demonstrating the inference pipeline for a single model (Llama 3.2 3B Instruct via AWS SageMaker JumpStart). Adapt the endpoint name and model ID to reproduce results for other models. |
| `data for cost.csv` | Aggregated cost and performance metrics per model — token pricing, accuracy scores, and cost-efficiency ratios used in the regression analysis. |
| `modelsanswers.zip` | Raw model responses for all evaluated LLMs across the Belebele benchmark subset. One row per question–model pair. See schema below. |

---

## `modelsanswers.zip` — Column Schema

Each file inside the archive corresponds to one evaluated model and contains the following columns:

| Column | Description | Example |
|--------|-------------|---------|
| `orgquestion` | Original question text from the Belebele benchmark | `According to the passage, which of the following did not affect relations between the USSR and the USA?` |
| `question` | Full prompt as sent to the model, including the four answer options | `Select correct answer to this question: … Answer options are: 1)Cultural differences 2)Opinions about Germany …` |
| `llmmodel` | Model identifier including deployment timestamp suffix | `meta-llama-3-8b-instruct-100517` |
| `full_prompt` | Complete API request payload including model parameters (`max_new_tokens`, `top_p`, `temperature`, `stop`) | `{'inputs': '<\|begin_of_text\|>…', 'parameters': {'max_new_tokens': 1100, 'top_p': 0.9, 'temperature': 0.4, …}}` |
| `llm_answer` | Raw API response as returned by the endpoint | `{'generated_text': '\n\n2'}` |
| `CORRECT ANSWER` | Ground-truth answer number (1–4) from the Belebele dataset | `2` |
| `LLM_ANWER` | Parsed LLM answer number extracted from the response | `2` |

**Inference parameters used are in the prompt. Example from Llama:**

```
max_new_tokens : 1100
top_p          : 0.9
temperature    : 0.4
stop           : <|eot_id|>   (or equivalent model-specific stop token)
```

Models are instructed to answer by providing only the option number (1–4). Accuracy is computed by comparing `LLM_ANWER` against `CORRECT ANSWER`.

---

## Evaluation Notebook — How It Works

The notebook demonstrates the full evaluation pipeline for **one model** deployed as a BYOC endpoint on **AWS SageMaker JumpStart**. The same pattern was applied to all 32 models in the study.

### Stack

- **AWS SageMaker JumpStart** — provisioned LLM endpoints (BYOC architecture)
- **LangChain** (`langchain`, `langchain-community`) — SageMaker endpoint integration
- **boto3** / `sagemaker` SDK — session and endpoint management
- **pandas** — dataset handling and result collection

### Pipeline steps

1. **Connect to SageMaker session** — region resolved automatically from the active session

2. **Point to a provisioned endpoint:**
   ```python
   model = 'meta-llama-3-2-3b-instruct-211224'
   llm_endpoint_name = 'jumpstart-dft-llama-3-2-3b-instruct-20250725-211224'
   ```

3. **Load the benchmark dataset** — Belebele-format multiple-choice questions (options 1–4)

4. **Batch inference loop** — sends each question to the endpoint, collects the full prompt and raw response:
   ```python
   for row in df.itertuples(index=False):
       response = predictor.predict(full_prompt)
       ...
   df['full_prompt'] = res
   df['llm_answer'] = res2
   ```

### Adapting for a different model

Change **two lines** to run the evaluation for another model:

```python
# model identifier
model = 'YOUR-MODEL-NAME'

# deployed endpoint name
llm_endpoint_name = 'YOUR-ENDPOINT-NAME'
```

> The prompt template uses Llama 3 chat tokens (`<|begin_of_text|>`, `<|start_header_id|>`, `<|eot_id|>`). For models with different chat formats (Mistral, Claude, Titan, etc.), update the prompt delimiters accordingly.

---

## Requirements

```bash
pip install langchain==0.1.14 boto3==1.34.58 sqlalchemy==2.0.29 sagemaker pandas
```

### AWS credentials

```bash
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_DEFAULT_REGION=us-east-1
```

The model endpoint must be **already provisioned** in SageMaker before running the notebook.

---

## Citation

If you use this data or code, please cite:

```
Ploskonka, M. (2025). Cloud-Based Large Language Model Deployment:
A Comparative Analysis of Serverless and Bring-Your-Own-Container
Architectures. Manuscript submitted to Acta Informatica Pragensia.
https://github.com/matpl2/Cloud-Based-Large-Language-Model-Deployment
```

---

## License

Data and code are released for academic replication purposes. Please contact the author for any other intended use.

---

## Author

**Mateusz Ploskonka**  
PhD Candidate, Applied Informatics — Prague University of Economics and Business (VŠE)  
Supervisor: Prof. Tomáš Kliegr
