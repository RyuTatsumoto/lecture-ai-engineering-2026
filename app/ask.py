"""第1回: 素のLLMに質問するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 質問を省略すると、講座に固有の質問をする
    uv run python -m app.ask --chat     # 会話履歴を保って続けて質問する

ツールを持たない「素の」呼び出し。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）
"""

import argparse
import os
import time

from google import genai
from google.genai import errors
from google.genai import chats
from google.genai import types

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"


def _generation_config() -> types.GenerateContentConfig | None:
    temperature = os.getenv("AIE_TEMPERATURE")
    if temperature is None:
        return None
    return types.GenerateContentConfig(temperature=float(temperature))


def _print_response_stats(
    response: types.GenerateContentResponse, elapsed_seconds: float
) -> None:
    usage = response.usage_metadata
    stats = f"[応答時間] {elapsed_seconds:.2f}秒"
    if usage is not None:
        stats += (
            f" | [トークン] 入力: {usage.prompt_token_count}, "
            f"出力: {usage.candidates_token_count}"
        )
    print(stats)


def ask(question: str) -> str:
    client = genai.Client(api_key=get_api_key())
    started_at = time.perf_counter()
    response = client.models.generate_content(
        model=MODEL, contents=question, config=_generation_config()
    )
    elapsed_seconds = time.perf_counter() - started_at
    _print_response_stats(response, elapsed_seconds)

    return response.text


def _answer_in_chat(chat: chats.Chat, question: str) -> None:
    started_at = time.perf_counter()
    response = chat.send_message(question)
    elapsed_seconds = time.perf_counter() - started_at
    print(response.text)
    _print_response_stats(response, elapsed_seconds)


def run_chat(initial_question: str | None = None) -> None:
    """質問を繰り返し、同じチャットセッションの履歴をAPIに渡す。"""
    client = genai.Client(api_key=get_api_key())
    chat = client.chats.create(model=MODEL, config=_generation_config())

    if initial_question:
        print(f"[質問] {initial_question}\n")
        _answer_in_chat(chat, initial_question)

    print("会話モードです。終了するには「終了」と入力してください。")
    while True:
        try:
            question = input("あなた> ").strip()
        except EOFError:
            print()
            break

        if question.lower() in {"終了", "exit", "quit"}:
            break
        if not question:
            continue

        _answer_in_chat(chat, question)


def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMに質問する")
    parser.add_argument(
        "--chat",
        action="store_true",
        help="会話履歴を保って続けて質問する",
    )
    parser.add_argument("question", nargs="?", help="最初の質問")
    args = parser.parse_args()

    print(f"[model] {MODEL}")
    try:
        if args.chat:
            run_chat(args.question)
        else:
            question = args.question or DEFAULT_QUESTION
            print(f"[質問] {question}\n")
            print(ask(question))
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
