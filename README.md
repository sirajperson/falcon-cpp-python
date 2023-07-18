#  Python Bindings for `ggllm.cpp`


Simple Python bindings for [`ggllm.cpp`](https://github.com/cmp-nct/ggllm.cpp) library.
This package provides:

- Low-level access to C API via `ctypes` interface.
- High-level Python API for text completion
  - OpenAI-like API
  - LangChain compatibility

This project is currently in alpha development and is not yet completely functional. Any contributions are warmly welcomed.


## High-level API

The high-level API provides a simple managed interface through the `Llama` class.

Below is a short example demonstrating how to use the high-level API to generate text:

```python
>>> from falcon_cpp import Falcon
>>> llm = Falcon(model_path="./models/7B/ggml-model.bin")
>>> output = llm("Q: Name the planets in the solar system? A: ", max_tokens=32, stop=["Q:", "\n"], echo=True)
>>> print(output)
{
  "id": "cmpl-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
  "object": "text_completion",
  "created": 1679561337,
  "model": "./models/7B/ggml-model.bin",
  "choices": [
    {
      "text": "Q: Name the planets in the solar system? A: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune and Pluto.",
      "index": 0,
      "logprobs": None,
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 14,
    "completion_tokens": 28,
    "total_tokens": 42
  }
}
```

## Web Server

`falcon-cpp-python` offers a web server which aims to act as a drop-in replacement for the OpenAI API.
This allows you to use ggllm.cpp to inference falcon models with any OpenAI compatible client (language libraries, services, etc).

To install the server package and get started:

```bash
python3 -m llama_cpp.server --model models/7B/ggml-model.bin
```

Navigate to [http://localhost:8000/docs](http://localhost:8000/docs) to see the OpenAPI documentation.

## Low-level API

The low-level API is a direct [`ctypes`](https://docs.python.org/3/library/ctypes.html) binding to the C API provided by `ggllm.cpp`.
The entire lowe-level API can be found in [falcon_cpp/falcon_cpp.py](https://github.com/sirajperson/falcon-cpp-python/blob/master/falcon_cpp/falcon_cpp.py) and directly mirrors the C API in [libfalcon.h](https://github.com/cmp-nct/ggllm.cpp/blob/master/libfalcon.h).

Below is a short example demonstrating how to use the low-level API to tokenize a prompt:

```python
>>> import falcon_cpp
>>> import ctypes
>>> params = falcon_cpp.falcon_context_default_params()
# use bytes for char * params
>>> ctx = falcon_cpp.falcon_init_backend("./models/7b/ggml-model.bin", params)
>>> max_tokens = params.n_ctx
# use ctypes arrays for array params
>>> tokens = (falcon_cpp.falcon_token * int(max_tokens))()
>>> n_tokens = falcon_cpp.falcon_tokenize(ctx, b"Q: Name the planets in the solar system? A: ", tokens, max_tokens, add_bos=llama_cpp.c_bool(True))
>>> falcon_cpp.falcon_free(ctx)
```

Check out the [examples folder](examples/low_level_api) for more examples of using the low-level API.

# Documentation
Coming soon...

# Development

Again, this package is under active development and I welcome any contributions.

To get started, clone the repository and install the package in development mode:

```bash
git clone --recurse-submodules git@github.com:abetlen/llama-cpp-python.git
cd llama-cpp-python

# Install with pip
pip install -e .

# if you want to use the fastapi / openapi server
pip install -e .[server]

# If you're a poetry user, installing will also include a virtual environment
poetry install --all-extras
. .venv/bin/activate

# Will need to be re-run any time vendor/llama.cpp is updated
python3 setup.py develop
```

# This Project is a fork of llama-cpp-python

This project was originally llama-cpp-python and owes an immense thanks to @abetlen.
This projects goal is to
- Provide a simple process to install `ggllm.cpp` and access the full C API in `libfalcon.h` from Python
- Provide a high-level Python API that can be used as a drop-in replacement for the OpenAI API so existing apps can be easily ported to use `ggllm.cpp`

Any contributions and changes to this package will be made with these goals in mind.

# License

This project is licensed under the terms of the MIT license.
