# Hard Negative Mining

This branch of the repository contains scripts for mining hard negative examples and running them against RAG. It is advisory to use Google Colab when running the `run_rag` script.

## Prerequisites

- Git
- Conda (Miniconda or Anaconda)

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/CarThrottlefan/LLM_w_RAG_testing
cd LLM_w_RAG_testing
```

### 2. Create and Activate Conda Environment
```bash
conda create -n bcqa python=3.11 -y
conda activate bcqa
```

### 3. Install Dexter-CQA Package and other packages
```bash
pip install -e dexter-cqa
```
```bash
pip install pandas
```
```bash
pip install torch
```
```bash
pip install transformers
```
```bash
pip install tqdm
```
```bash
pip install numpy
```

### 4. Run Hard Negative Mining
```bash
python ./hard_negatives
```