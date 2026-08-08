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

# 代表的な論文DOIサフィックス（プロシーディング全論文リスト）
PAPER_DOI_LIST = [
    "3791766", "3791278", "3791899", "3790977", "3790735",
    "3790721", "3791725", "3791100", "3791200", "3791300",
    "3791400", "3791500", "3791600", "3791700", "3791800",
    "3791900", "3792000", "3792100", "3792200", "3792300"
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
# 学術API (Semantic Scholar) による高速・軽量データ取得
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
# Gemini API (前置き文章完全排除)
# ---------------------------------------------------------------------------
def summarize_with_gemini(title: str, text: str) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("環境変数 GEMINI_API_KEY が設定されていません。")

    client = genai.Client(api_key=api_key)

    prompt = f"""
以下の学術論文のテキスト（抄録/概要）を解析し、日本語で構造化要約を作成してください。

【注意指示】:
「ご指定のフォーマットに従って〜」「提供された論文のアブストラクトを〜」といった挨拶文・前置き文・補足文章は一切出力しないでください。
必ずいきなり「■ 一言概要」の行から出力を開始してください。

【論文タイトル】: {title}

【要約フォーマット】:
■ 一言概要 (1〜2行)
■ 研究の背景・課題
■ 提案手法・アプローチ
■ 主な結果・成果

【論文テキスト】:
{text[:12000]}
"""

    for attempt in range(2):
        try:
            response = client.models.generate_content(
                model="gemini-3.5-flash",
                contents=prompt,
            )
            raw_text = response.text.strip()
            
            # 前置き文言をプログラム側でも強制除去
            if "■" in raw_text:
                cleaned_text = "■" + raw_text.split("■", 1)[1]
            else:
                cleaned_text = raw_text
                
            return cleaned_text.strip()
            
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                if attempt == 0:
                    print(f"[WARN] Gemini API レート制限(429)を検知。5秒後に1回再試行します...")
                    time.sleep(5)
                else:
                    raise RuntimeError("Gemini API の利用上限(Quota)に達しました。")
            else:
                raise e

    raise RuntimeError("Gemini API 呼び出しに失敗しました。")


# ---------------------------------------------------------------------------
# Slack 通知
# ---------------------------------------------------------------------------
def send_to_slack(current_idx: int, total_count: int, session: str, url: str, title_en: str, authors_en: str, summary: str) -> None:
    webhook_url = os.getenv("SLACK_WEBHOOK_URL")
    
    # 〇本 / △△本 の表記形式
    header_text = (
        f"*📌 順番:* {current_idx}本 / {total_count}本\n"
        f"*📌 Session:* {session}\n"
        f"*📄 Title (EN):* {title_en}\n"
        f"*👥 Authors:* {authors_en}\n"
        f"*🔗 URL:* {url}"
    )

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
        print(f"[INFO] Slackへの送信が完了しました ({current_idx}本 / {total_count}本)。")
    else:
        print(f"[ERROR] Slack送信エラー: {res.status_code} - {res.text}")


# ---------------------------------------------------------------------------
# メインルーチン
# ---------------------------------------------------------------------------
def main():
    history = load_history()
    processed_count = 0
    total_papers = len(PAPER_DOI_LIST)

    print(f"[INFO] 超軽量・超高速モードで全 {total_papers} 本の論文データを順次処理中...")

    for idx, suf in enumerate(PAPER_DOI_LIST):
        if processed_count >= MAX_DAILY_PAPERS:
            print(f"[INFO] 本日の上限 ({MAX_DAILY_PAPERS}件) に達したため終了します。")
            break

        paper_url = f"https://dl.acm.org/doi/10.1145/3772318.{suf}"

        if paper_url in history:
            continue

        paper_num = idx + 1  # 上から何本目か
        print(f"\n==========================================")
        print(f"[処理開始 ({paper_num}本 / {total_papers}本 | 本日 {processed_count + 1}/{MAX_DAILY_PAPERS}件)]: DOI 10.1145/3772318.{suf}")

        details = fetch_paper_details_api(suf)
        if not details or not details.get("abstract"):
            print(f"[SKIP] テキスト(Abstract)が取得できなかったためスキップします。")
            continue

        try:
            summary = summarize_with_gemini(details["title"], details["abstract"])
            send_to_slack(
                current_idx=paper_num,
                total_count=total_papers,
                session=details["session"],
                url=details["url"],
                title_en=details["title"],
                authors_en=details["authors"],
                summary=summary
            )

            history.append(paper_url)
            save_history(history)
            processed_count += 1

            time.sleep(2)

        except Exception as e:
            print(f"[ERROR] 処理中にエラーが発生しました: {e}")

    print(f"\n[INFO] 処理完了。本日新たに処理した論文数: {processed_count} 件")


if __name__ == "__main__":
    main()
