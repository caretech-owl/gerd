# Concepts

## Usage

GERD is primarly a tool for prototyping workflows for working with Large Language Models.
It is meant to act as 'glue' between different tools and services and should ease the access to these tools.

In general, there should be only be two components involved in a GERD workflow: A *configuration* and a *service*. The configuration can be assembled from different sources and should be able to be used in different services.
The foundation of such a configration is a YAML file.
GERD provides a set of those which can be found in the `config` directory.
Configurations in this directly can be accessed by name as shown in the following example.
A simple configutation file may look like this:

``` yaml
--8<-- "config/hello.yml"
```

And can be used with a `ChatService` via `load_gen_config` shown in the following example:

``` python {linenums="1" hl_lines="16 17"}
--8<-- "examples/hello.py"
```

## Configuration

Parameters in a configuration can be overwritten by environment variables.
This is useful for secrets or other sensitive information or when using the same configuration in different environments.
Environment variables may be provided in a `.env` file in the root directory of the project.
`GenerationConfig` are prefixed with `gerd_gen_`, so to override the temperature of a model, the environment variable `gerd_gen_model__temperature` needs to be set.
Similarly, `QAConfig` configurations are prefixed with `gerd_qa_`.
The path for the used vector store can be set with `gerd_qa_embedding__db_path`.
Note that nested configurations are separated by `__`.
All environment variables are case insensitive, so you can use `GERD_GEN_MODEL__TEMPERATURE` as well to keep the common style of environment variables.

## Services

Gerd can work with models using [transformers](https://huggingface.co/transformers/) or llama.cpp bindings via [llama-cpp-python](). As `llama-cpp-python` is an optional dependency, you need to specify `llama-cpp` when you install GERD.

```bash
# when installing from source
pip install .[llama-cpp]
# when using uv
uv sync --group llama-cpp
```

GERD choses whether to use `transformers` or `llama-cpp` based on the configuration. Usually, you need to specify a certain GGUF file when using quantized models which is the primary use for GERD and `llama-cpp`.
Consequently, if `model.file` is set, GERD will assume that you want to use a quantized model and load llama-cpp bindings.
If only a huggingface model handle (e.g. `org/model_name`) is provided, GERD will use the transformers library to load the model.

Furthermore, GERD supports the usage of llama.cpp servers as well as the OpenAI API.
If you set an [endpoint](/gerd/reference/gerd/models/model/#gerd.models.model.ModelEndpoint), the model handle and file parameters will be ignored and the model will be ignored.
Make sure to set the right endpoint type because the provided capabilities differ.
