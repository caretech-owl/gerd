# Related tools and service

There is a lot of momentum in the field of Large Language Models (LLMs) and many tools and services are being developed to work with them. GERD is not intended to be a daily driver for working with LLMs, but rather a tool to prototype workflows and to ease the access and configuration of currently available models.

There are many tools, some with large and striving communities, that can be used to work with LLMs. Here are some of them:

## [Hugging Face Transformers](https://huggingface.co/transformers/)
The Transformers library is a popular open-source library that provides a wide range of pre-trained models for Natural Language Processing (NLP) tasks. It is built on top of PyTorch and TensorFlow and provides a simple API for working with LLMs.   

## [llama.cpp](https://github.com/ggml-org/llama.cpp)
The main goal of llama.cpp is to enable LLM inference with minimal setup and state-of-the-art performance on a wide range of hardware - locally and in the cloud.

## [GPT4All](https://www.nomic.ai/gpt4all)
GPT4All is a desktop application that allows you to download many different LLM and use them locally.
The application is available for Windows, MacOS, and Linux.
Furthermore, you can use use the generation services from OpenAI, DeepSeek or Mistral as well.

## [Private GPT](https://privategpt.dev/)
This application is very similar to GPT4All.
It used to provide more options to tweak workflows that uses Retrieval Augmented Generation (RAG).
However, as the development of both applications progresses rapidly, it is best to check them out yourself.

## [Text generation web UI](https://github.com/oobabooga/text-generation-webui)
A feature-rich web UI based on gradio with a primary focus on text generation.
It can be used to test models for text generation, for instruction based tasks, even with different roles/custom characters.
As it is a web UI, it can also be used to host a llm service in a local network or -- thanks to gradio -- even on the internet.
The ui also features means to train LoRA on unstructured data or for instruction-based tasks.

## [Ollama](https://ollama.com/)
Ollama provides a terminal server and client to host and use llm models within minutes.
For most use cases it wraps configurations (e.g. for llama.cpp) and provides a simple interface to use them.
It a good tool to get started, however, it is not as flexible as llama.cpp itself and also sometimes provides configurations that could be misleading.
For instance, if you use `deepseek-r1` with ollama, it will download and host a quantized and distilled model, which is not the same as the original model.
Usually, having a look at the configuration in the online database should clarifty this and as most machine aren't capable of hosting a full `r1` model, this is not a bad thing, but it is important to know that the model is not the same as the original one.
