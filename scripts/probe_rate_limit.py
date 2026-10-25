"""無料枠のレート制限を探る: 短い質問を連続送信し、最初の429で止める（第1回 発展）。

使い方:
    uv run python -m scripts.probe_rate_limit
    uv run python -m scripts.probe_rate_limit --model gemini-3.5-flash --max-requests 60
"""

import argparse
import time

from google import genai
from google.genai import errors

from app.config import DEFAULT_MODEL, get_api_key

QUESTION = "1+1は？"


def main() -> None:
    parser = argparse.ArgumentParser(description="最初の429が出るまで連続でリクエストを送る")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-requests", type=int, default=40)
    args = parser.parse_args()

    client = genai.Client(api_key=get_api_key())
    start = time.perf_counter()
    for i in range(1, args.max_requests + 1):
        try:
            client.models.generate_content(model=args.model, contents=QUESTION)
        except errors.APIError as e:
            print(f"{i}回目で {e.code} / 開始から {time.perf_counter() - start:.1f}秒")
            print(str(e.message)[:300])
            return
    print(f"{args.max_requests}回すべて成功 / {time.perf_counter() - start:.1f}秒")


if __name__ == "__main__":
    main()
