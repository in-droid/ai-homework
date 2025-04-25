# Setup

This project requires **Python 3.11.5**. Make sure you're using the correct Python version (e.g., via `pyenv`, `conda`, or `virtualenv`).

## 1. Install Dependencies

Install the required Python packages by running:

```bash
pip install -r requirements.txt
pip install .

```
## 2. Environment variables
Copy the contents from the [example_env file](../.example_env) into your `env` file and set up the tokens.
```
cp .example_env .env
```

## 3. Usage
To use the best performing model you can use the following python commands
```python
from src.gemini_llm import GeminiLLM
from src.utils import wiki_intro

# model name can also be any other Gemini model
model = GeminiLLM(model_name="gemini-2.0-flash-lite", temperature=0)
category = model.categorize(company_name="Altria Group")
category_with_des = model.categorize_get_description(
    company_name="Altria Group",
    get_description=wiki_intro
)
```
`model.categorize_get_description` accepts any other function that may fetch additional information about the company.


## 4. Project structure

### Notebooks `notebooks/`
 - **`data_exploration.ipynb`** 
 Initial data exploration and visualizations.

- **`classify_embeddings.ipynb`**
Embedding + classifier pipeline. Getting descriptions for each company, generating sentence embeddings, training and evaluating an XGBoost classifier. Visualization of results.

- **`llm.ipynb`**
Evaluation of the prompt-engineering approach with Google's Gemini API.

### `src/` Package

- **utils.py**
Helper functions for data splitting / loading and getting descriptions.

- **model.py**
Core model interface. So both classification methods are compatible.

- **gemini_llm.py**
Gemini LLM classification pipeline.

- **eval.py**
Evaluation metrics with bootstrapping.

- **classify_emb.py**
Inference only class for the embedding + classification pipeline.

### Report
Report on the performance and methods used available [here](../report.md)
