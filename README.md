# Generating and evaluating relevant documentation (GERD)

GERD is developed as an experimental library to investigate how large language models (LLMs) can be used to generate and analyze (sets of) documents.

This project was initially forked from [Llama-2-Open-Source-LLM-CPU-Inference
](https://github.com/kennethleungty/Llama-2-Open-Source-LLM-CPU-Inference) by [Kenneth Leung](https://github.com/kennethleungty).
___

## Quickstart

```shell
# 1) Create a virtual environment with a tool of your choice
# e.g. conda, venv or micromamba
micromamba create -n gerd python=3.12
# 2) Activate env; prompt starts with (gerd) now
micromamba activate gerd
# 3) Install poetry
pip install poetry
# 4) Clone project
git clone git@github.com:caretech-owl/gerd.git
cd gerd
# 5) Install dependencies and GERD
# 5.1: You need to chose either CPU or GPU as extras
# to install the correct version of FAISS
# if you unsure, use 'cpu'
poetry install --extras cpu
# 5.2: Alternatively
# poetry install --extras gpu
# 5.3: To install development dependencies as well
# include 'dev' group
# poetry install --with dev --extras cpu
# 5.4: You can add further extras
# --extras recommended  # install a set of useful extras (includes cpu)
# --extras gguf # for llama.cpp and GGUF model support;
# --extras awq  # for GPU model quantization
# --extras gui  # for Gradio GUIs
# 5.5: Recommended for CPU inference
# poetry install --extras recommended

# 6) Run GERD 'hello' chat example
# This will download a LLM from https://huggingface.co 
# and may take a while
poe hello
```

Remember to acticate the virtual environment (step 2) every time you start using GERD.
For more information on development look at [DEV.md](DEV.md).
If you want to try this out in your browser, head over to binder 👉 [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/caretech-owl/gerd/HEAD?labpath=%2Fnotebooks%2Fhello_gerd.ipynb). 
Note that running LLMs on the CPU (and especially on limited virtual machines like binder) takes some time.
If you are in a hurry you might be better off by cloning the repo and running the [examples](./examples/) or [notebooks](./notebooks/) locally.

## Question and Answer example

Follow quickstart and make sure to install `gui` and `gguf` dependencies. Next start the qa server via `poe qa`.
When the server is done loading, open `http://127.0.0.1:7860` in your browser.

```shell
poe qa
# Some Llama.cpp outut
# ...
# * Running on local URL:  http://127.0.0.1:7860
```

Click the 'Click to Upload' button and search for a [GRASCCO](https://pubmed.ncbi.nlm.nih.gov/36073490/) document named `Caja.txt` which is located in the `tests/data/grascoo` folder and upload it into the vector store. Next, you can query information from the document. For instance `Wie heißt der Patient?` (What is the patient called?).

![](images/qa.png)

## Prompt Chaining

Prompt chaining is a prompt engineering approach to increase the 'reflection' of a large language model onto its given answer. Check [examples/chaining.py](examples/chaining.py) for an illustration. Also, have a look at how chaining is configured and used with GERD. You can find the config at [config/gen_chaining.yml](config/gen_chaining.yml)

```shell
python examples/chaining.py
# ...
====== Resolved prompt =====

system: You are a helpful assistant. Please answer the following question in a truthful and brief manner.
user: What type of mammal lays the biggest eggs?

# ...
Result: Based on the given information, the largest egg-laying mammal is the blue whale, which can lay up to 100 million eggs per year. However, the other assertions provided do not align with this information.
```

As you see, the answer does not make much sense with the default model which is rather small.
Give it a try with [meta-llama/Llama-3.2-3B](https://huggingface.co/meta-llama/Llama-3.2-3B).
To use this model, you need to login with the huggingface [cli](https://huggingface.co/docs/huggingface_hub/en/guides/cli) and accept the Meta Community License Agreement.

___
## Used Tools
- **LangChain**: Framework for developing applications powered by language models
- **C Transformers**: Python bindings for the Transformer models implemented in C/C++ using GGML library
- **FAISS**: Open-source library for efficient similarity search and clustering of dense vectors.
- **Sentence-Transformers (all-MiniLM-L6-v2)**: Open-source pre-trained transformer model for embedding text to a 384-dimensional dense vector space for tasks like
- **Poetry**: Tool for dependency management and Python packaging

___
## Files and Content
- `/assets`: Images relevant to the project
- `/config`: Configuration files for LLM applications
- `/examples`: Examples that demonstrate the different usage scenarios
- `/gerd`: Code related to `GERD`
- `/images`: Images for the documentation
- `/models`: Binary file of GGML quantized LLM model (i.e., Llama-2-7B-Chat)
- `/prompts`: Plain text prompt files
- `/templates`: Prompt files as jinja2 templates 
- `/tests`: Unit tests for `GERD`
- `/vectorstore`: FAISS vector store for documents
- `pyproject.toml`: TOML file to specify which versions of the dependencies used (Poetry)
___

## References
- https://github.com/kennethleungty/Llama-2-Open-Source-LLM-CPU-Inference
- https://pubmed.ncbi.nlm.nih.gov/36073490
- https://huggingface.co
