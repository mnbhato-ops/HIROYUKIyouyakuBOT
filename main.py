import json
import os
import re
import sys
import time
import urllib.parse
from typing import Dict, List, Optional

from bs4 import BeautifulSoup
from DrissionPage import ChromiumPage, ChromiumOptions
from google import genai
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
# DrissionPage による ACM ページ取得 (Cloudflare 回避 & アコーディオン全開対応)
# ---------------------------------------------------------------------------
def fetch_papers_list(page: ChromiumPage) -> List[Dict[str, str]]:
    """DrissionPage で Cloudflare を回避し、アコーディオンを展開して論文一覧を取得する"""
    print(f"[INFO] 論文一覧を取得中 (ACM DL): {PROCEEDINGS_URL}")
    
    page.get(PROCEEDINGS_URL)
    
    # Cloudflare 通過待ち (最大 20 秒)
    for i in range(20):
        if "Just a moment" in page.title or "しばらく" in page.title:
            time.sleep(1)
        else:
            break

    print(f"[INFO] ページ読み込み完了: {page.title}")

    # クッキー同意ポップアップがあれば閉じる
    try:
        onetrust_btn = page.ele("#onetrust-accept-btn-handler", timeout=2)
        if onetrust_btn:
            onetrust_btn.click()
    except Exception:
        pass

    # スクロールして遅延読み込みを完了させる
    for _ in range(3):
        page.scroll.down(1000)
        time.sleep(1)

    # 「Expand all」またはアコーディオンを展開
    print("[INFO] セッションアコーディオンを展開中...")
    try:
        expand_btn = page.ele('text:Expand all', timeout=3)
        if expand_btn:
            expand_btn.click()
            time.sleep(3)
        else:
            headers = page.eles('.accordion-tabbed__control')
            for h in headers:
                try:
                    h.click()
                    time.sleep(0.3)
                except Exception:
                    pass
    except Exception as e:
        print(f"[WARN] アコーディオン展開中にスキップ: {e}")

    time.sleep(2)
    html_content = page.html
    soup = BeautifulSoup(html_content, "html.parser")
    
    # 論文DOIリンクの正規表現抽出
    all_links = soup.find_all("a", href=re.compile(r"/doi/(abs/|full/)?10\.1145/3772318\.\d+"))
    seen_urls = set()
    papers = []

    for a in all_links:
        href = a.get("href", "")
        # PDFダウンロードリンクや補足資料、重複は除く
        if "/pdf/" in href or "cited-by" in href or "purchase-access" in href or "supplemental" in href:
            continue

        title = a.get_text(strip=True)
        # タイトルが短すぎるもの・Podcast・補足ファイルは除外
        if not title or len(title) < 8 or title.lower() in ["pdf", "epub", "abstract", "get access"] or title.endswith((".mp4", ".pdf", ".vtt", ".zip")):
            continue
            
        if "session summary podcast" in title.lower() or "podcast:" in title.lower():
            continue

        full_url = urllib.parse.urljoin("https://dl.acm.org", href).split("?")[0]
        full_url = re.sub(r"/doi/(abs|full)/", "/doi/", full_url)

        if full_url in seen_urls:
            continue
        seen_urls.add(full_url)

        # セッションタイトルの取得
        session_name = "General Session"
        section_elem = a.find_previous(class_=re.compile(r"accordion-tabbed__title|section__title|topic-heading|heading"))
        if section_elem:
            session_name = section_elem.get_text(strip=True)

        papers.append({
            "title": title,
            "url": full_url,
            "session": session_name,
        })

    print(f"[INFO] 全 {len(papers)} 件の有効な論文エントリーを検出しました。")
    return papers


def fetch_paper_details(page: ChromiumPage, paper_url: str) -> Dict[str, Optional[str]]:
    """論文個別ページから Abstract (抄録) を取得する"""
    print(f"[INFO] 論文詳細ページを取得中: {paper_url}")
    
    page.get(paper_url)
    for i in range(15):
        if "Just a moment" in page.title or "しばらく" in page.title:
            time.sleep(1)
        else:
            break

    html = page.html
    soup = BeautifulSoup(html, "html.parser")

    # Abstract 抽出
    abstract_elem = soup.find(class_=re.compile(r"abstractSection|abstractInFull|abstract-content")) or \
                    soup.find("section", id="abstract") or \
                    soup.find("div", class_=re.compile(r"abstract"))
    abstract_text = abstract_elem.get_text(strip=True) if abstract_elem else ""

    return {
        "abstract": abstract_text,
    }


# ---------------------------------------------------------------------------
# Gemini API (gemini-2.5-flash)
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
■ 論文タイトル (日本語訳)
■ 一言概要 (1〜2行)
■ 研究の背景・課題
■ 提案手法・アプローチ
■ 主な結果・成果

【論文テキスト】:
{text[:12000]}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    return response.text


# ---------------------------------------------------------------------------
# Slack 通知
# ---------------------------------------------------------------------------
def send_to_slack(session: str, url: str, summary: str) -> None:
    webhook_url = os.getenv("SLACK_WEBHOOK_URL")
    if not webhook_url:
        print("[WARN] SLACK_WEBHOOK_URL 未設定のため画面に要約を出力します:\n")
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
    
    co = ChromiumOptions()
    co.auto_port()
    page = ChromiumPage(co)

    try:
        papers = fetch_papers_list(page)
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

            details = fetch_paper_details(page, paper["url"])
            
            paper_text = details["abstract"]
            if not paper_text or len(paper_text.strip()) < 50:
                print(f"[SKIP] テキスト(Abstract)が抽出できなかったためスキップします。")
                continue

            try:
                summary = summarize_with_gemini(paper["title"], paper_text)
                send_to_slack(paper["session"], paper["url"], summary)

                history.append(paper_id)
                save_history(history)
                processed_count += 1

                time.sleep(3)

            except Exception as e:
                print(f"[ERROR] 処理中にエラーが発生しました: {e}")

        print(f"\n[INFO] 処理完了。本日新たに処理した論文数: {processed_count} 件")

    finally:
        page.quit()


if __name__ == "__main__":
    main()
