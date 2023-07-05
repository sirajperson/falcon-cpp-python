import falcon_cpp

MODEL = "./vendor/ggllm/models/ggml-vocab.bin"


def test_falcon():
    falcon = falcon_cpp.Falcon(model_path=MODEL, vocab_only=True)

    assert falcon
    assert falcon.ctx is not None

    text = b"Hello World"

    assert falcon.detokenize(falcon.tokenize(text)) == text


# @pytest.mark.skip(reason="need to update sample mocking")
def test_falcon_patch(monkeypatch):
    falcon = falcon_cpp.Falcon(model_path=MODEL, vocab_only=True)
    n_vocab = falcon_cpp.falcon_n_vocab(falcon.ctx)

    ## Set up mock function
    def mock_eval(*args, **kwargs):
        return 0

    def mock_get_logits(*args, **kwargs):
        return (falcon_cpp.c_float * n_vocab)(
            *[falcon_cpp.c_float(0) for _ in range(n_vocab)]
        )

    monkeypatch.setattr("falcon_cpp.falcon_cpp.falcon_eval", mock_eval)
    monkeypatch.setattr("falcon_cpp.falcon_cpp.falcon_get_logits", mock_get_logits)

    output_text = " jumps over the lazy dog."
    output_tokens = falcon.tokenize(output_text.encode("utf-8"))
    token_eos = falcon.token_eos()
    n = 0

    def mock_sample(*args, **kwargs):
        nonlocal n
        if n < len(output_tokens):
            n += 1
            return output_tokens[n - 1]
        else:
            return token_eos

    monkeypatch.setattr("falcon_cpp.falcon_cpp.falcon_cpp_sample_token", mock_sample)

    text = "The quick brown fox"

    ## Test basic completion until eos
    n = 0  # reset
    completion = falcon.create_completion(text, max_tokens=20)
    assert completion["choices"][0]["text"] == output_text
    assert completion["choices"][0]["finish_reason"] == "stop"

    ## Test streaming completion until eos
    n = 0  # reset
    chunks = falcon.create_completion(text, max_tokens=20, stream=True)
    assert "".join(chunk["choices"][0]["text"] for chunk in chunks) == output_text
    assert completion["choices"][0]["finish_reason"] == "stop"

    ## Test basic completion until stop sequence
    n = 0  # reset
    completion = falcon.create_completion(text, max_tokens=20, stop=["lazy"])
    assert completion["choices"][0]["text"] == " jumps over the "
    assert completion["choices"][0]["finish_reason"] == "stop"

    ## Test streaming completion until stop sequence
    n = 0  # reset
    chunks = falcon.create_completion(text, max_tokens=20, stream=True, stop=["lazy"])
    assert (
        "".join(chunk["choices"][0]["text"] for chunk in chunks) == " jumps over the "
    )
    assert completion["choices"][0]["finish_reason"] == "stop"

    ## Test basic completion until length
    n = 0  # reset
    completion = falcon.create_completion(text, max_tokens=2)
    assert completion["choices"][0]["text"] == " j"
    assert completion["choices"][0]["finish_reason"] == "length"

    ## Test streaming completion until length
    n = 0  # reset
    chunks = falcon.create_completion(text, max_tokens=2, stream=True)
    assert "".join(chunk["choices"][0]["text"] for chunk in chunks) == " j"
    assert completion["choices"][0]["finish_reason"] == "length"


def test_falcon_pickle():
    import pickle
    import tempfile

    fp = tempfile.TemporaryFile()
    falcon = falcon_cpp.Falcon(model_path=MODEL, vocab_only=True)
    pickle.dump(falcon, fp)
    fp.seek(0)
    falcon = pickle.load(fp)

    assert falcon
    assert falcon.ctx is not None

    text = b"Hello World"

    assert falcon.detokenize(falcon.tokenize(text)) == text


def test_utf8(monkeypatch):
    falcon = falcon_cpp.Falcon(model_path=MODEL, vocab_only=True)
    n_vocab = falcon_cpp.falcon_n_vocab(falcon.ctx)

    ## Set up mock function
    def mock_eval(*args, **kwargs):
        return 0

    def mock_get_logits(*args, **kwargs):
        return (falcon_cpp.c_float * n_vocab)(
            *[falcon_cpp.c_float(0) for _ in range(n_vocab)]
        )

    monkeypatch.setattr("falcon_cpp.falcon_cpp.falcon_eval", mock_eval)
    monkeypatch.setattr("falcon_cpp.falcon_cpp.falcon_get_logits", mock_get_logits)

    output_text = "😀"
    output_tokens = falcon.tokenize(output_text.encode("utf-8"))
    token_eos = falcon.token_eos()
    n = 0

    def mock_sample(*args, **kwargs):
        nonlocal n
        if n < len(output_tokens):
            n += 1
            return output_tokens[n - 1]
        else:
            return token_eos

    monkeypatch.setattr("falcon_cpp.falcon_cpp.falcon_sample_token", mock_sample)

    ## Test basic completion with utf8 multibyte
    n = 0  # reset
    completion = falcon.create_completion("", max_tokens=4)
    assert completion["choices"][0]["text"] == output_text

    ## Test basic completion with incomplete utf8 multibyte
    n = 0  # reset
    completion = falcon.create_completion("", max_tokens=1)
    assert completion["choices"][0]["text"] == ""


def test_falcon_server():
    from fastapi.testclient import TestClient
    from falcon_cpp.server.app import create_app, Settings

    settings = Settings(
        model=MODEL,
        vocab_only=True,
    )
    app = create_app(settings)
    client = TestClient(app)
    response = client.get("/v1/models")
    assert response.json() == {
        "object": "list",
        "data": [
            {
                "id": MODEL,
                "object": "model",
                "owned_by": "me",
                "permissions": [],
            }
        ],
    }
