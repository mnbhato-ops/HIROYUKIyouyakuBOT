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
# Playwright による ACM ページ取得 (描画待機・構造解析強化版)
# ---------------------------------------------------------------------------
def fetch_papers_list() -> List[Dict[str, str]]:
    """Playwright を使用して HTML を正確に取得する"""
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
            
            # 論文リンク要素が表示されるまで最大15秒待つ
            page.wait_for_selector('a[href*="/doi/10.1145/"]', timeout=15000)
            
            # スクロールして動的コンテンツ（Lazy Load）を展開
            for _ in range(5):
                page.mouse.wheel(0, 2000)
                time.sleep(1)

            html_content = page.content()
        except Exception as e:
            print(f"[WARN] 待機中にタイムアウトまたはエラー発生 (取得済みコンテンツで解析を続行します): {e}")
            html_content = page.content()
        finally:
            browser.close()

    if not html_content:
        print("[ERROR] ページコンテンツが取得できませんでした。")
        return []

    soup = BeautifulSoup(html_content, "html.parser")
    papers = []

    # ACM Proceedings の論文カード要素を探す
    # 複数パターン（.issue-item, .accordion-tabbed__content, .issue-item__title など）に対応
    paper_nodes = soup.find_all("div", class_=re.compile(r"issue-item|article-snippet|item-meta"))
    
    # ノードが取れない場合のバックアップ: 直接論文リンク（a[href*="/doi/10.1145/"]）を抽出
    if not paper_nodes:
        print("[INFO] クラス指定ノードが見つからないため、汎用リンク解析を実行します。")
        doi_links = soup.find_all("a", href=re.compile(r"/doi/(abs/)?10\.1145/"))
        seen_urls = set()
        
        for a in doi_links:
            href = a.get("href", "")
            title = a.get_text(strip=True)
            
            # 代表的なDOIリンクのみに絞り込み
            if not title or len(title) < 5 or "pdf" in href.lower():
                continue
            
            full_url = urllib.parse.urljoin("https://dl.acm.org", href).replace("/abs/", "/")
            if full_url in seen_urls:
                continue
            seen_urls.add(full_url)

            # 親要素周辺からオープンアクセス判定
            parent = a.find_parent("div") or a.find_parent("li")
            is_oa = False
            if parent:
                if parent.find("i", class_=re.compile(r"fa-unlock|open-access|oa-icon")) or parent.find("img", alt=re.compile(r"Open Access", re.I)):
                    is_oa = True
                elif parent.find("a", href=re.compile(r"/doi/pdf/")):
                    is_oa = True

            papers.append({
                "title": title,
                "url": full_url,
                "session": "Proceedings Session",
                "is_open_access": is_oa,
            })
    else:
        current_session = "General Session"
        for node in paper_nodes:
            session_elem = node.find_previous(["h2", "h3", "h4", "div"], class_=re.compile(r"section-title|topic-heading|accordion-tabbed__title"))
            if session_elem:
                current_session = session_elem.get_text(strip=True)

            title_elem = node.find(["h5", "h6", "span", "div"], class_=re.compile(r"issue-item__title|publication-title"))
            a_tag = None
            if title_elem:
                a_tag = title_elem.find("a")
            else:
                a_tag = node.find("a", href=re.compile(r"/doi/"))

            if not a_tag or not a_tag.get("href"):
                continue

            paper_title = a_tag.get_text(strip=True)
            if not paper_title or len(paper_title) < 5:
                continue

            href = a_tag["href"]
            paper_url = urllib.parse.urljoin("https://dl.acm.org", href)

            is_open_access = False
            if node.find("i", class_=re.compile(r"fa-unlock|open-access|oa-icon")) or node.find("img", alt=re.compile(r"Open Access", re.I)):
                is_open_access = True
            elif node.find("a", href=re.compile(r"/doi/pdf/")):
                is_open_access = True

            papers.append({
                "title": paper_title,
                "url": paper_url,
                "session": current_session,
                "is_open_access": is_open_access,
            })

    print(f"[INFO] 全 {len(papers)} 件の論文エントリーを検出しました。")
    return papers


def extract_pdf_text(paper_url: str) -> Optional[str]:
    """PDF ダウンロード"""
    pdf_url = paper_url.replace("/doi/", "/doi/pdf/").replace("/doi/abs/", "/doi/pdf/")
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
