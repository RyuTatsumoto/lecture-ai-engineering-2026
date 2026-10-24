from types import SimpleNamespace

from app import ask as ask_module


def test_ask_prints_input_and_output_token_counts(monkeypatch, capsys):
    response = SimpleNamespace(
        usage_metadata=SimpleNamespace(
            prompt_token_count=12,
            candidates_token_count=7,
        ),
        text="answer",
    )

    calls = []

    class FakeClient:
        class models:
            @staticmethod
            def generate_content(**kwargs):
                calls.append(kwargs)
                return response

    monkeypatch.setattr(ask_module.genai, "Client", lambda **kwargs: FakeClient())
    monkeypatch.setattr(ask_module, "get_api_key", lambda: "test-key")
    monkeypatch.setenv("AIE_TEMPERATURE", "0.2")

    assert ask_module.ask("question") == "answer"
    assert calls[0]["config"].temperature == 0.2
    assert capsys.readouterr().out == "[トークン] 入力: 12, 出力: 7\n"


def test_ask_leaves_temperature_unset_by_default(monkeypatch):
    response = SimpleNamespace(usage_metadata=None, text="answer")
    calls = []

    class FakeClient:
        class models:
            @staticmethod
            def generate_content(**kwargs):
                calls.append(kwargs)
                return response

    monkeypatch.setattr(ask_module.genai, "Client", lambda **kwargs: FakeClient())
    monkeypatch.setattr(ask_module, "get_api_key", lambda: "test-key")
    monkeypatch.delenv("AIE_TEMPERATURE", raising=False)

    assert ask_module.ask("question") == "answer"
    assert calls[0]["config"] is None