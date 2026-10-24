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
    ticks = iter([10.0, 11.25])
    monkeypatch.setattr(ask_module.time, "perf_counter", lambda: next(ticks))

    assert ask_module.ask("question") == "answer"
    assert calls[0]["config"].temperature == 0.2
    assert (
        capsys.readouterr().out
        == "[応答時間] 1.25秒 | [トークン] 入力: 12, 出力: 7\n"
    )


def test_ask_leaves_temperature_unset_by_default(monkeypatch, capsys):
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
    ticks = iter([10.0, 11.25])
    monkeypatch.setattr(ask_module.time, "perf_counter", lambda: next(ticks))

    assert ask_module.ask("question") == "answer"
    assert calls[0]["config"] is None
    assert capsys.readouterr().out == "[応答時間] 1.25秒\n"


def test_run_chat_reuses_chat_session_for_follow_up_questions(
    monkeypatch, capsys
):
    questions = []

    class FakeChat:
        def send_message(self, question):
            questions.append(question)
            return SimpleNamespace(
                usage_metadata=None,
                text=f"answer: {question}",
            )

    fake_chat = FakeChat()
    chat_configs = []

    class FakeChats:
        @staticmethod
        def create(**kwargs):
            chat_configs.append(kwargs)
            return fake_chat

    class FakeClient:
        chats = FakeChats()

    user_inputs = iter(["second", "終了"])
    monkeypatch.setattr(ask_module.genai, "Client", lambda **_: FakeClient())
    monkeypatch.setattr(ask_module, "get_api_key", lambda: "test-key")
    monkeypatch.setattr("builtins.input", lambda _: next(user_inputs))
    ticks = iter([10.0, 11.5, 12.0, 13.25])
    monkeypatch.setattr(ask_module.time, "perf_counter", lambda: next(ticks))

    ask_module.run_chat("first")

    assert questions == ["first", "second"]
    assert chat_configs[0]["model"] == ask_module.MODEL
    output = capsys.readouterr().out
    assert "answer: first" in output
    assert "answer: second" in output
    assert output.count("[応答時間]") == 2


def test_main_starts_chat_mode(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr("sys.argv", ["app.ask", "--chat", "first question"])
    monkeypatch.setattr(
        ask_module, "run_chat", lambda question: calls.append(question)
    )

    ask_module.main()

    assert calls == ["first question"]
    assert capsys.readouterr().out == f"[model] {ask_module.MODEL}\n"