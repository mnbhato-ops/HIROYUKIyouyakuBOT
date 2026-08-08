import json
import os
import re
import sys
import time
import urllib.parse
from io import BytesIO
from typing import Dict, List, Optional

from bs4 import BeautifulSoup
from google import genai
import pypdf
import requests

# ---------------------------------------------------------------------------
# 設定値
# ---------------------------------------------------------------------------
PROCEEDINGS_DOI = "10.1145/3772318"
PROCEEDINGS_URL = f"https://dl.acm.org/doi/proceedings/{PROCEEDINGS_DOI}"
HISTORY_FILE = "processed_papers.json"
MAX_DAILY_PAPERS = 5

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Referer": PROCEEDINGS_URL,
}


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
# ACM 公式 JSON API から論文一覧を確実に取得
# ---------------------------------------------------------------------------
def fetch_papers_list() -> List[Dict[str, str]]:
    """ACM Digital Library の TOC (Table of Contents) データから論文を取得"""
    print(f"[INFO] ACM API から Proceedings 情報を取得中 (DOI: {PROCEEDINGS_DOI})...")
    
    # 1. ACM の Crossref エンドポイント経由で全メンバーDOIを取得
    url = f"https://api.crossref.org/works/{PROCEEDINGS_DOI}"
    papers = []
    
    try:
        res = requests.get(url, headers={"User-Agent": "ACM-Paper-Summarizer/1.0"}, timeout=30)
        if res.status_code == 200:
            data = res.json().get("message", {})
            relation = data.get("relation", {})
            
            # has-part や includes に含まれる論文DOIのリストを取得
            has_part = relation.get("has-part", [])
            print(f"[INFO] 関連エントリー数: {len(has_part)}")
            
            for part in has_part:
                doi = part.get("id", "")
                if doi and "10.1145" in doi:
                    paper_url = f"https://dl.acm.org/doi/{doi}"
                    papers.append({
                        "title": f"Paper ({doi})",
                        "url": paper_url,
                        "doi": doi,
                        "session": "Proceedings Paper",
                        "is_open_access": True
                    })
    except Exception as e:
        print(f"[WARN] Crossref 取得エラー: {e}")

    # 2. 上記で取得できない場合、ACM DL の HTML メタデータから直接全DOIパターンを抽出
    if not papers:
        print("[INFO] ACM DL ページからのメタデータダイレクト抽出を開始...")
        try:
            res = requests.get(PROCEEDINGS_URL, headers=HEADERS, timeout=30)
            # ページ内のすべての DOI パターンを正規表現で一発抽出
            dois = set(re.findall(r'10\.1145/3772318\.\d+|10\.1145/\d+\.\d+', res.text))
            
            # プロシーディング自身のDOIを除外
            dois.discard(PROCEEDINGS_DOI)
            
            for doi in dois:
                papers.append({
                    "title": f"ACM Paper ({doi})",
                    "url": f"https://dl.acm.org/doi/{doi}",
                    "doi": doi,
                    "session": "Session Paper",
                    "is_open_access": True
                })
            print(f"[INFO] 正規表現抽出により {len(papers)} 件の論文DOIを発見しました。")
        except Exception as e:
            print(f"[ERROR] 抽出失敗: {e}")

    print(f"[INFO] 合計 {len(papers)} 件の対象論文を検出しました。")
    return papers


def extract_pdf_text(paper_url: str) -> Optional[str]:
    """PDFをダウンロードしてテキスト化（タイトル補正付き）"""
    pdf_url = paper_url.replace("/doi/", "/doi/pdf/").replace("/doi/abs/", "/doi/pdf/")
    print(f"[INFO] PDFを取得中: {pdf_url}")

    try:
        res = requests.get(pdf_url, headers=HEADERS, timeout=30)
        if res.status_code != 200:
            print(f"[WARN] PDF取得失敗 (Status: {res.status_code})。有料または閲覧制限のためスキップします。")
            return None

        pdf_file = BytesIO(res.content)
        reader = pypdf.PdfReader(pdf_file)
        
        extracted_text = ""
        max_pages = min(len(reader.pages), 8)
        for page_idx in range(max_pages):
            text = reader.pages[page_idx].extract_text()
            if text:
                extracted_text += text + "\n"

        if len(extracted_text.strip()) > 200:
            return extracted_text
        return None

    except Exception as e:
        print(f"[ERROR] PDF抽出失敗: {e}")
        return None


# ---------------------------------------------------------------------------
# Gemini API (gemini-3.5-flash)
# ---------------------------------------------------------------------------
def summarize_with_gemini(title: str, text: str) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("環境変数 GEMINI_API_KEY が設定されていません。")

    client = genai.Client(api_key=api_key)

    prompt = f"""
以下の学術論文のテキスト本文を解析し、日本語で分かりやすく構造化要約を作成してください。
最初に本文から正確な「論文タイトル」を読み取って記載してください。

【要約フォーマット】:
■ 論文タイトル (日本語訳および英語原題)
■ 一言概要 (1〜2行)
■ 研究の背景・課題
■ 提案手法・アプローチ
■ 主な結果・成果

【論文テキスト】:
{text[:12000]}
"""

    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=prompt,
    )
    return response.text


# ---------------------------------------------------------------------------
# Slack 通知
# ---------------------------------------------------------------------------
def send_to_slack(session: str, url: str, summary: str) -> None:
    webhook_url = os.getenv("SLACK_WEBHOOK_URL")
    if not webhook_url:
        print("[WARN] SLACK_WEBHOOK_URL 未設定。")
        print(summary)
        return

    payload = {
        "blocks": [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*📌 ACM Proceeding Paper*\n*🔗 Original URL:* {url}"
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
        print("[INFO] Slackへの送信が完了しました。")
    else:
        print(f"[ERROR] Slack送信エラー: {res.status_code} - {res.text}")


# ---------------------------------------------------------------------------
# メインルーチン
# ---------------------------------------------------------------------------
def main():
    history = load_history()
    papers = fetch_papers_list()

    processed_count = 0

    for paper in papers:
        if processed_count >= MAX_DAILY_PAPERS:
            print(f"[INFO] 本日の上限 ({MAX_DAILY_PAPERS}件) に達したため終了します。")
            break

        paper_id = paper["url"]

        if paper_id in history:
            continue

        print(f"\n==========================================")
        print(f"[処理開始 ({processed_count + 1}/{MAX_DAILY_PAPERS})]: {paper['url']}")
        
        pdf_text = extract_pdf_text(paper["url"])
        if not pdf_text:
            print(f"[SKIP] PDFを取得できなかったため（有料記事またはダウンロード制限）スキップします。")
            continue

        try:
            summary = summarize_with_gemini(paper["title"], pdf_text)
            send_to_slack(paper["session"], paper["url"], summary)

            history.append(paper_id)
            save_history(history)
            processed_count += 1

            time.sleep(5)

        except Exception as e:
            print(f"[ERROR] 処理中にエラーが発生しました: {e}")

    print(f"\n[INFO] 処理完了。本日新たに処理した論文数: {processed_count} 件")


if __name__ == "__main__":
    main()
