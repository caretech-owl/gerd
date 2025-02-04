# Concepts

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