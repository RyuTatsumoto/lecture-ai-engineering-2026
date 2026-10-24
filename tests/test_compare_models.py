from types import SimpleNamespace

from app import compare_models


def test_run_model_calls_ask_with_model_and_parses_stats(monkeypatch, capsys):
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(
            stdout=(
                "[model] gemini-3.5-flash-lite\n"
                "[応答時間] 1.25秒 | [トークン] 入力: 12, 出力: 7\n"
                "answer\n"
            ),
            stderr="",
            returncode=0,
        )

    monkeypatch.setattr(compare_models.subprocess, "run", fake_run)

    result = compare_models.run_model("gemini-3.5-flash-lite", "質問")

    command, kwargs = calls[0]
    assert command[-2:] == ["app.ask", "質問"]
    assert kwargs["env"]["AIE_MODEL"] == "gemini-3.5-flash-lite"
    assert result.elapsed_seconds == 1.25
    assert result.input_tokens == 12
    assert result.output_tokens == 7
    assert capsys.readouterr().out.endswith("answer\n")


def test_main_uses_same_question_for_both_models(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(
        "sys.argv", ["app.compare_models", "共通の質問"]
    )

    def fake_run_model(model, question):
        calls.append((model, question))
        return compare_models.Result(model, 2.0, 10, 20)

    monkeypatch.setattr(compare_models, "run_model", fake_run_model)

    compare_models.main()

    assert calls == [
        ("gemini-3.5-flash-lite", "共通の質問"),
        ("gemini-3.8-flash", "共通の質問"),
    ]
    output = capsys.readouterr().out
    assert "10 | 20 | 10.00 トークン/秒" in output
