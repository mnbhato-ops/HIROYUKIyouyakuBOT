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
from playwright.sync_api import sync_playwright
import pypdf
import requests

# ---------------------------------------------------------------------------
# 設定値
# ---------------------------------------------------------------------------
PROCEEDINGS_URL = "https://dl.acm.org/doi/proceedings/10.1145/3772318"
HISTORY_FILE = "processed_papers.json"
MAX_DAILY_PAPERS = 5


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
# Playwright による ACM ページ取得 (アコーディオン全開対応)
# ---------------------------------------------------------------------------
def fetch_papers_list() -> List[Dict[str, str]]:
    """Playwright でアコーディオンを展開し、HTML を取得する"""
    print(f"[INFO] 論文一覧を取得中 (Playwright): {PROCEEDINGS_URL}")
    
    html_content = ""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        )
        page = context.new_page()

        try:
            # ページ読み込み
            page.goto(PROCEEDINGS_URL, wait_until="domcontentloaded", timeout=60000)
            time.sleep(5)

            # クッキー同意ポップアップなどがあれば閉じる
            try:
                page.click("#onetrust-accept-btn-handler", timeout=3000)
            except Exception:
                pass

            # 「Expand All」またはアコーディオンの展開ボタンをすべてクリックする
            print("[INFO] セッションアコーディオンを展開中...")
            
            # Expand All ボタンを探索してクリック
            expand_btn = page.query_selector('a:has-text("Expand all"), button:has-text("Expand all"), .expand-all')
            if expand_btn:
                expand_btn.click()
                time.sleep(3)
            else:
                # 無ければ個別のセッションヘッダーをすべてクリックして開く
                headers = page.query_selector_all('.accordion-tabbed__control, .section__title, [data-toggle="collapse"]')
                for h in headers:
                    try:
                        h.click()
                        time.sleep(0.5)
                    except Exception:
                        pass

            # スクロールしてコンテンツの遅延読み込み（Lazy Load）を完了させる
            for _ in range(5):
                page.mouse.wheel(0, 1500)
                time.sleep(1)

            html_content = page.content()

        except Exception as e:
            print(f"[WARN] ページ処理中に例外が発生しました: {e}")
            html_content = page.content()
        finally:
            browser.close()

    if not html_content:
        print("[ERROR] ページコンテンツが取得できませんでした。")
        return []

    soup = BeautifulSoup(html_content, "html.parser")
    papers = []

    # 展開されたHTMLから全リンクおよびDOI要素を解析
    # ACMのDOI形式 (/doi/10.1145/ または /doi/abs/10.1145/) に該当するすべてのaタグを対象にする
    all_links = soup.find_all("a", href=re.compile(r"/doi/(abs/|full/)?10\.1145/"))
    seen_urls = set()

    for a in all_links:
        href = a.get("href", "")
        # PDFダウンロードリンクや重複は除く
        if "/pdf/" in href or "cited-by" in href:
            continue

        title = a.get_text(strip=True)
        # タイトルが短すぎるもの（「PDF」ボタンやアイコンリンク等）は除外
        if not title or len(title) < 8 or title.lower() in ["pdf", "epub", "abstract"]:
            continue

        full_url = urllib.parse.urljoin("https://dl.acm.org", href).split("?")[0]
        # DOIの標準URL化 (/abs/ などを正規化)
        full_url = re.sub(r"/doi/(abs|full)/", "/doi/", full_url)

        if full_url in seen_urls:
            continue
        seen_urls.add(full_url)

        # 親要素を遡ってオープンアクセス・PDF可否判定
        parent = a.find_parent(["div", "li", "tr"])
        is_oa = False
        session_name = "General Session"

        if parent:
            # オープンアクセス表示またはPDFリンクが存在するか
            if parent.find("i", class_=re.compile(r"fa-unlock|open-access|oa-icon")) or \
               parent.find("img", alt=re.compile(r"Open Access", re.I)) or \
               parent.find("a", href=re.compile(r"/doi/pdf/")):
                is_oa = True

            # セッションタイトルの取得
            session_elem = parent.find_previous(["h2", "h3", "h4", "div"], class_=re.compile(r"section-title|topic-heading|accordion-tabbed__title|heading"))
            if session_elem:
                session_name = session_elem.get_text(strip=True)

        papers.append({
            "title": title,
            "url": full_url,
            "session": session_name,
            "is_open_access": is_oa,
        })

    print(f"[INFO] 全 {len(papers)} 件の論文エントリーを検出しました。")
    return papers


def extract_pdf_text(paper_url: str) -> Optional[str]:
    """PDF ダウンロード"""
    pdf_url = paper_url.replace("/doi/", "/doi/pdf/")
    print(f"[INFO] PDFを取得中: {pdf_url}")
    
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    try:
        res = requests.get(pdf_url, headers=headers, timeout=30)
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

        if not paper["is_open_access"]:
            print(f"[SKIP] 有料/アクセス制限ありの論文のためスキップ: {paper['title']}")
            continue

        print(f"\n==========================================")
        print(f"[処理開始 ({processed_count + 1}/{MAX_DAILY_PAPERS})]: {paper['title']}")
        
        pdf_text = extract_pdf_text(paper["url"])
        if not pdf_text:
            print(f"[SKIP] PDFテキストが抽出できなかったためスキップします。")
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
