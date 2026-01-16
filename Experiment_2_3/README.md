# DEXTER Experiment Setup Guide

This guide provides step-by-step instructions for setting up and running experiments using the DEXTER repository on colab.

## Prerequisites

- Runtime version to 2025.07 on colab.
- Hugging Face account with an access token

## Setup Instructions

### 1. Clone the DEXTER Repository

First, clone the DEXTER repository to your local environment:

```bash
!git clone https://github.com/VenkteshV/DEXTER.git
```

### 2. Configure Hugging Face Token

You'll need a valid Hugging Face token to access models and datasets.

1. Generate a token from your Hugging Face settings page: https://huggingface.co/settings/tokens
2. Set the environment variable in your Python environment:

```python
import os
os.environ["huggingface_token"] = 'YOUR_HF_TOKEN_HERE'
```

**Important:** Replace `'YOUR_HF_TOKEN_HERE'` with your actual Hugging Face token.

### 3. Set OpenAI Key (Dummy Value)

The DEXTER library requires an `OPENAI_KEY` environment variable to be set, even if you're not using OpenAI services. Set a dummy value to bypass this requirement:

```python
os.environ["OPENAI_KEY"] = "dummy_key_for_dexter_import"
```

### 4. Install Dependencies

Install the required version of transformers to avoid compatibility issues:

```bash
!pip install transformers==4.30.0
```

### 5. Make sure you can access the dataset

Sharing the dataset drive link

```
https://drive.google.com/drive/folders/1ZxZxriYxhaNbOX9akgiw8OOukoRtMLsj?usp=sharing
```

### 6. Mount Google Drive in colab

After executing the below, command, you should be able to see the shared dataset in colab.

```
from google.colab import drive
drive.mount('/content/drive')

```

## Running Experiments

### Experiment 2: WikiMultiHop RAG

This experiment runs RAG (Retrieval-Augmented Generation) on the WikiMultiHop dataset.

**Steps:**
1. Navigate to `DEXTER/evaluation/wikimultihop/llms/`
2. Upload the `run_rag.py` file to this directory
3. Set correct config.ini path in rag_oracle.py file.
4. Set correct path for retrieval_results.json
3. Execute the experiment:

```bash
!python3 /content/DEXTER/evaluation/wikimultihop/llms/run_rag.py
```

### Experiment 3: RAG Oracle

This experiment runs the RAG oracle evaluation.

**Steps:**
1. Navigate to 'DEXTER/evaluation/wikimultihop/llms/'
2. Upload the `rag_oracle.py` file to this directory
3. Set correct config.ini path in rag_oracle.py file.
3. Execute the experiment:

```bash
!python /content/DEXTER/evaluation/wikimultihop/llms/rag_oracle.py
```

## Troubleshooting

- **Import Errors:** Ensure that the dummy `OPENAI_KEY` is set before importing DEXTER modules
- **Transformer Version Issues:** Make sure you've installed `transformers==4.30.0` specifically
- **Hugging Face Authentication:** Verify that your token is valid and has the necessary permissions

## Notes
- Ensure all file paths are correct before running experiments
- Both `run_rag.py` and `rag_oracle.py` files need to be uploaded to their respective directories before execution

## Repository

For more information about DEXTER, visit: https://github.com/VenkteshV/DEXTER