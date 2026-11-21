"""Compare response time and token usage for two Gemini models.

Usage:
    uv run python -m app.compare_models "LoRAとは何ですか？"
"""

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass


MODELS = ("gemini-3.5-flash-lite", "gemini-3.8-flash")
STATS_PATTERN = re.compile(
    r"^\[応答時間\]\s+([0-9]+(?:\.[0-9]+)?)秒\s+\|\s+"
    r"\[トークン\]\s+入力:\s*(\d+),\s*出力:\s*(\d+)\s*$",
    re.MULTILINE,
)


@dataclass
class Result:
    model: str
    elapsed_seconds: float
    input_tokens: int
    output_tokens: int


def run_model(model: str, question: str) -> Result:
    env = os.environ.copy()
    env["AIE_MODEL"] = model
    process = subprocess.run(
        [sys.executable, "-m", "app.ask", question],
        capture_output=True,
        check=False,
        env=env,
        text=True,
    )

    if process.stdout:
        print(process.stdout, end="")
    if process.stderr:
        print(process.stderr, end="", file=sys.stderr)
    if process.returncode != 0:
        raise SystemExit(process.returncode)

    stats = STATS_PATTERN.search(process.stdout)
    if stats is None:
        raise RuntimeError(f"{model} の応答時間・トークン数を読み取れませんでした")

    return Result(
        model=model,
        elapsed_seconds=float(stats.group(1)),
        input_tokens=int(stats.group(2)),
        output_tokens=int(stats.group(3)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="同じ質問を2つのGeminiモデルに送り、応答時間とトークン数を比較する"
    )
    parser.add_argument("question", help="両方のモデルに送る質問")
    args = parser.parse_args()

    results = []
    for model in MODELS:
        print(f"\n=== {model} ===")
        results.append(run_model(model, args.question))

    print("\n=== 比較 ===")
    print("モデル | 応答時間 | 入力トークン | 出力トークン | 出力速度（概算）")
    for result in results:
        speed = (
            f"{result.output_tokens / result.elapsed_seconds:.2f} トークン/秒"
            if result.elapsed_seconds > 0
            else "算出不可"
        )
        print(
            f"{result.model} | {result.elapsed_seconds:.2f}秒 | "
            f"{result.input_tokens} | {result.output_tokens} | {speed}"
        )


if __name__ == "__main__":
    main()
