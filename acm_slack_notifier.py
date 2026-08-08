import json
import os
import re
import sys
import time
from typing import Dict, List, Optional

from google import genai
import requests

# ---------------------------------------------------------------------------
# 設定値
# ---------------------------------------------------------------------------
PROCEEDINGS_URL = "https://dl.acm.org/doi/proceedings/10.1145/3772318"
HISTORY_FILE = "processed_papers.json"
MAX_DAILY_PAPERS = 5

# 代表的な論文DOIサフィックス（CHI 2026 論文リスト）
PAPER_DOI_LIST = [
    "3791766", "3791278", "3791899", "3790977", "3790735",
    "3790721", "3791725", "3791100", "3791200", "3791300",
    "3791400", "3791500", "3791600", "3791700", "3791800"
]


# ---------------------------------------------------------------------------
# 履歴管理機能
# ---------------------------------------------------------------------------
def load_history() -> List[str]:
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[WARN] 履歴ファイルの読み込み失敗: {e}")
            return []
    return []


def save_history(history: List[str]) -> None:
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# 学術API (Semantic Scholar) による高速・軽量データ取得 (ブラウザ不要)
# ---------------------------------------------------------------------------
def fetch_paper_details_api(doi_suffix: str) -> Optional[Dict[str, str]]:
    """API 経由でタイトル、著者名、Abstract を高速取得する"""
    doi = f"10.1145/3772318.{doi_suffix}"
    url = f"https://dl.acm.org/doi/{doi}"
    api_url = f"https://api.semanticscholar.org/graph/v1/paper/{doi}?fields=title,abstract,url,venue,authors"
    headers = {"User-Agent": "AcademicPaperNotifier/1.0"}

    try:
        r = requests.get(api_url, headers=headers, timeout=10)
        if r.status_code == 200:
            data = r.json()
            title = data.get("title") or f"Paper {doi}"
            abstract = data.get("abstract") or ""
            venue = data.get("venue") or "ACM CHI Conference"
            raw_authors = data.get("authors", [])
            authors = [a.get("name") for a in raw_authors if a.get("name")]
            authors_str = ", ".join(authors) if authors else "Authors Unknown"

            if abstract:
                return {
                    "doi": doi,
                    "title": title,
                    "abstract": abstract,
                    "session": venue,
                    "authors": authors_str,
                    "url": url
                }
    except Exception as e:
        print(f"[WARN] API 取得エラー (DOI: {doi}): {e}")
    return None


# ---------------------------------------------------------------------------
# Gemini API (gemini-2.5-flash / 自動リトライ付き)
# ---------------------------------------------------------------------------
def summarize_with_gemini(title: str, text: str) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("環境変数 GEMINI_API_KEY が設定されていません。")

    client = genai.Client(api_key=api_key)

    prompt = f"""
以下の学術論文のテキスト（抄録/概要）を解析し、日本語で分かりやすく構造化要約を作成してください。

【論文タイトル】: {title}

【要約フォーマット】:
■ 一言概要 (1〜2行)
■ 研究の背景・課題
■ 提案手法・アプローチ
■ 主な結果・成果

【論文テキスト】:
{text[:12000]}
"""

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model="gemini-3.5-flash",
                contents=prompt,
            )
            return response.text
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                print(f"[WARN] Gemini API レート制限(429)を検知。30秒待機してリトライします... ({attempt + 1}/3)")
                time.sleep(30)
            else:
                raise e

    raise RuntimeError("Gemini API のリトライ上限に達しました。")


# ---------------------------------------------------------------------------
# Slack 通知
# ---------------------------------------------------------------------------
def send_to_slack(session: str, url: str, title_en: str, authors_en: str, summary: str) -> None:
    webhook_url = os.getenv("SLACK_WEBHOOK_URL")
    
    header_text = f"*📌 Session:* {session}\n*📄 Title (EN):* {title_en}\n*👥 Authors:* {authors_en}\n*🔗 URL:* {url}"

    if not webhook_url:
        print("[WARN] SLACK_WEBHOOK_URL 未設定のため画面に要約を出力します:\n")
        print(header_text)
        print("\n" + summary)
        return

    payload = {
        "blocks": [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": header_text
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": summary
                }
            },
            {"type": "divider"}
        ]
    }

    res = requests.post(webhook_url, json=payload)
    if res.status_code == 200:
        print("[INFO] Slackへの送信が完了しました (タイトル/著者付き・爆速版)。")
    else:
        print(f"[ERROR] Slack送信エラー: {res.status_code} - {res.text}")


# ---------------------------------------------------------------------------
# メインルーチン
# ---------------------------------------------------------------------------
def main():
    history = load_history()
    processed_count = 0

    print("[INFO] 超軽量・超高速モードで論文データを取得中...")

    for suf in PAPER_DOI_LIST:
        if processed_count >= MAX_DAILY_PAPERS:
            print(f"[INFO] 本日の上限 ({MAX_DAILY_PAPERS}件) に達したため終了します。")
            break

        paper_url = f"https://dl.acm.org/doi/10.1145/3772318.{suf}"

        if paper_url in history:
            continue

        print(f"\n==========================================")
        print(f"[処理開始 ({processed_count + 1}/{MAX_DAILY_PAPERS})]: DOI 10.1145/3772318.{suf}")

        details = fetch_paper_details_api(suf)
        if not details or not details.get("abstract"):
            print(f"[SKIP] テキスト(Abstract)が取得できなかったためスキップします。")
            continue

        try:
            summary = summarize_with_gemini(details["title"], details["abstract"])
            send_to_slack(
                session=details["session"],
                url=details["url"],
                title_en=details["title"],
                authors_en=details["authors"],
                summary=summary
            )

            history.append(paper_url)
            save_history(history)
            processed_count += 1

            # API レート制限回避のため 6秒待機
            time.sleep(6)

        except Exception as e:
            print(f"[ERROR] 処理中にエラーが発生しました: {e}")

    print(f"\n[INFO] 処理完了。本日新たに処理した論文数: {processed_count} 件")


if __name__ == "__main__":
    main()
