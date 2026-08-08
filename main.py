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
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
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
# Crossref API / Direct Parse による論文一覧取得
# ---------------------------------------------------------------------------
def fetch_papers_list() -> List[Dict[str, str]]:
    """Crossref API を使用して Proceeding に含まれる論文一覧を取得する"""
    print(f"[INFO] Crossref API から論文一覧を取得中 (DOI: {PROCEEDINGS_DOI})...")
    
    api_url = f"https://api.crossref.org/works?filter=container-title:3772318&rows=100"

    items = []
    try:
        res = requests.get(api_url, headers=HEADERS, timeout=30)
        if res.status_code == 200:
            data = res.json()
            items = data.get("message", {}).get("items", [])
    except Exception as e:
        print(f"[ERROR] API取得失敗: {e}")

    # APIで一覧が取れなかった場合のフォールバック
    if not items:
        return fetch_papers_via_acm_direct()

    papers = []
    for item in items:
        doi = item.get("DOI", "")
        title = item.get("title", [""])[0]
        url = item.get("URL", f"https://dl.acm.org/doi/{doi}")

        is_oa = False
        licenses = item.get("license", [])
        for lic in licenses:
            if "creative commons" in lic.get("URL", "").lower() or "open" in lic.get("URL", "").lower():
                is_oa = True
                break

        if title and doi:
            papers.append({
                "title": title,
                "url": url,
                "doi": doi,
                "session": "Main Proceedings",
                "is_open_access": is_oa or True
            })

    print(f"[INFO] 全 {len(papers)} 件の論文エントリーを検出しました。")
    return papers


def fetch_papers_via_acm_direct() -> List[Dict[str, str]]:
    """ACM DL ページのパースフォールバック"""
    print(f"[INFO] ACM DL ページをパース中...")
    page_url = PROCEEDINGS_URL
    
    session = requests.Session()
    session.headers.update(HEADERS)
    
    try:
        res = session.get(page_url, timeout=30)
        soup = BeautifulSoup(res.text, "html.parser")
        
        papers = []
        links = soup.find_all("a", href=re.compile(r"/doi/(10\.1145/\d+)"))
        seen = set()
        
        for a in links:
            href = a.get("href", "")
            title = a.get_text(strip=True)
            if not title or len(title) < 5 or "pdf" in href.lower():
                continue
            
            full_url = urllib.parse.urljoin("https://dl.acm.org", href)
            if full_url in seen:
                continue
            seen.add(full_url)
            
            papers.append({
                "title": title,
                "url": full_url,
                "session": "Session Paper",
                "is_open_access": True
            })
            
        print(f"[INFO] 直接パースにより {len(papers)} 件検出しました。")
        return papers
    except Exception as e:
        print(f"[ERROR] パース失敗: {e}")
        return []


def extract_pdf_text(paper_url: str) -> Optional[str]:
    """PDF ダウンロード"""
    pdf_url = paper_url.replace("/doi/", "/doi/pdf/").replace("/doi/abs/", "/doi/pdf/")
    print(f"[INFO] PDFを取得中: {pdf_url}")

    try:
        res = requests.get(pdf_url, headers=HEADERS, timeout=30)
        if res.status_code != 200:
            print(f"[WARN] PDFの直接取得に失敗 (Status: {res.status_code})。スキップします。")
            return None

        pdf_file = BytesIO(res.content)
        reader = pypdf.PdfReader(pdf_file)
        
        extracted_text = ""
        max_pages = min(len(reader.pages), 8)
        for page_idx in range(max_pages):
            text = reader.pages[page_idx].extract_text()
            if text:
                extracted_text += text + "\n"

        return extracted_text if len(extracted_text.strip()) > 200 else None

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
以下の学術論文のテキストを解析し、日本語で分かりやすく構造化要約を作成してください。

【論文タイトル】: {title}

【要約フォーマット】:
■ 論文タイトル (日本語訳)
■ 一言概要 (1〜2行)
■ 研究の背景・課題
■ 提案手法・アプローチ
■ 主な結果・成果

【抽出テキスト】:
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
                    "text": f"*📌 Session:* {session}\n*🔗 Original URL:* {url}"
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
        print(f"[処理開始 ({processed_count + 1}/{MAX_DAILY_PAPERS})]: {paper['title']}")
        
        pdf_text = extract_pdf_text(paper["url"])
        if not pdf_text:
            print(f"[SKIP] PDFテキストが抽出できなかったため（有料または取得制限）スキップします。")
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
