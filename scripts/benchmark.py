"""応答時間とトークン数を、複数のモデル・質問で計測して比較する（第1回 発展）。

使い方:
    uv run python scripts/benchmark.py
    uv run python scripts/benchmark.py --runs 3 --models gemini-3.5-flash-lite gemini-3.5-flash

無料枠のレート制限（429）に当たったら、その回は失敗として記録して続行する。
"""

import argparse
import statistics
import time

from google import genai
from google.genai import errors

from app.config import get_api_key

MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
]

QUESTIONS = {
    "短答": "LoRAとは何ですか？ 1文で答えてください。",
    "長文": "LoRAとは何ですか？ 仕組み・利点・欠点・使いどころを詳しく説明してください。",
    "講座固有": "AIE2026の講座の宿題にある「3レベル」とは何ですか？",
}


def measure(client, model: str, question: str) -> dict:
    start = time.perf_counter()
    try:
        response = client.models.generate_content(model=model, contents=question)
    except errors.APIError as e:
        return {"ok": False, "sec": time.perf_counter() - start, "error": e.code}
    usage = response.usage_metadata
    return {
        "ok": True,
        "sec": time.perf_counter() - start,
        "in": usage.prompt_token_count,
        "out": usage.candidates_token_count,
        # 思考トークン（thinking）は出力トークンとは別に課金・計上される
        "think": getattr(usage, "thoughts_token_count", None) or 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="モデル・質問ごとの応答時間とトークン数を比較する")
    parser.add_argument("--runs", type=int, default=2, help="1条件あたりの試行回数")
    parser.add_argument("--models", nargs="+", default=MODELS)
    args = parser.parse_args()

    client = genai.Client(api_key=get_api_key())
    started = time.perf_counter()
    calls = 0

    print("| モデル | 質問 | 成功/試行 | 平均秒 | 入力tok | 出力tok | 思考tok | 出力tok/秒 | 失敗 |")
    print("|---|---|---|---|---|---|---|---|---|")
    for model in args.models:
        for label, question in QUESTIONS.items():
            results = []
            for _ in range(args.runs):
                results.append(measure(client, model, question))
                calls += 1
            ok = [r for r in results if r["ok"]]
            fails = sorted({str(r["error"]) for r in results if not r["ok"]})
            if ok:
                sec = statistics.mean(r["sec"] for r in ok)
                out = statistics.mean(r["out"] for r in ok)
                print(
                    f"| {model} | {label} | {len(ok)}/{len(results)} | {sec:.2f} "
                    f"| {statistics.mean(r['in'] for r in ok):.0f} | {out:.0f} "
                    f"| {statistics.mean(r['think'] for r in ok):.0f} | {out / sec:.0f} "
                    f"| {','.join(fails) or '-'} |"
                )
            else:
                print(f"| {model} | {label} | 0/{len(results)} | - | - | - | - | - | {','.join(fails)} |")

    elapsed = time.perf_counter() - started
    print(f"\n合計 {calls} リクエスト / {elapsed:.0f} 秒 = {calls / elapsed * 60:.1f} リクエスト/分")


if __name__ == "__main__":
    main()
