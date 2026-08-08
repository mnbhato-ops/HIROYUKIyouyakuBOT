import json
import os
import platform
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
# DrissionPage による ACM ページ取得
# ---------------------------------------------------------------------------
def fetch_papers_list(page: ChromiumPage) -> List[Dict[str, str]]:
    """ACM DL から論文一覧を取得する"""
    print(f"[INFO] 論文一覧を取得中 (ACM DL): {PROCEEDINGS_URL}")
    
    page.get(PROCEEDINGS_URL)
    
    # Cloudflare 通過待機 (最大 15 秒)
    for i in range(15):
        if "Just a moment" in page.title or "しばらく" in page.title:
            time.sleep(1)
        else:
            break

    print(f"[INFO] ページ読み込み完了: {page.title}")

    # クッキー同意ポップアップを閉じる
    try:
        onetrust_btn = page.ele("#onetrust-accept-btn-handler", timeout=2)
        if onetrust_btn:
            onetrust_btn.click()
    except Exception:
        pass

    # スクロール
    for _ in range(3):
        page.scroll.down(1000)
        time.sleep(1)

    # アコーディオンを展開
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
        print(f"[WARN] アコーディオン展開スキップ: {e}")

    time.sleep(2)
    html_content = page.html
    soup = BeautifulSoup(html_content, "html.parser")
    
    all_links = soup.find_all("a", href=re.compile(r"/doi/(abs/|full/)?10\.1145/3772318\.\d+"))
    seen_urls = set()
    papers = []

    for a in all_links:
        href = a.get("href", "")
        if "/pdf/" in href or "cited-by" in href or "purchase-access" in href or "supplemental" in href:
            continue

        title = a.get_text(strip=True)
        if not title or len(title) < 8 or title.lower() in ["pdf", "epub", "abstract", "get access"] or title.endswith((".mp4", ".pdf", ".vtt", ".zip")):
            continue
            
        if "session summary podcast" in title.lower() or "podcast:" in title.lower():
            continue

        full_url = urllib.parse.urljoin("https://dl.acm.org", href).split("?")[0]
        full_url = re.sub(r"/doi/(abs/|full/)", "/doi/", full_url)

        if full_url in seen_urls:
            continue
        seen_urls.add(full_url)

        session_name = "General Session"
        section_elem = a.find_previous(class_=re.compile(r"accordion-tabbed__title|section__title|topic-heading|heading"))
        if section_elem:
            session_name = section_elem.get_text(strip=True)

        papers.append({
            "title": title,
            "url": full_url,
            "session": session_name,
        })

    print(f"[INFO] 取得完了: 全 {len(papers)} 件の論文エントリーを検出しました。")
    return papers


# ---------------------------------------------------------------------------
# Semantic Scholar API による Abstract & Authors 取得
# ---------------------------------------------------------------------------
def fetch_paper_details_api(paper_url: str) -> Optional[Dict[str, str]]:
    """Semantic Scholar API を使って Abstract と Authors を取得する"""
    match = re.search(r"10\.1145/\d+\.\d+", paper_url)
    if not match:
        return None
    
    doi = match.group(0)
    api_url = f"https://api.semanticscholar.org/graph/v1/paper/{doi}?fields=title,abstract,url,venue,authors"
    headers = {"User-Agent": "AcademicPaperNotifier/1.0"}

    try:
        r = requests.get(api_url, headers=headers, timeout=10)
        if r.status_code == 200:
            data = r.json()
            title = data.get("title") or ""
            abstract = data.get("abstract") or ""
            venue = data.get("venue") or "General Session"
            raw_authors = data.get("authors", [])
            authors = [a.get("name") for a in raw_authors if a.get("name")]
            authors_str = ", ".join(authors) if authors else ""
            
            if abstract:
                return {
                    "title": title,
                    "abstract": abstract,
                    "session": venue,
                    "authors": authors_str,
                    "url": paper_url
                }
    except Exception as e:
        print(f"[WARN] Semantic Scholar API エラー (DOI: {doi}): {e}")
    return None


def fetch_paper_details(page: ChromiumPage, paper_url: str) -> Dict[str, Optional[str]]:
    """論文詳細ページから Abstract・著者名・Figure 1 画像URL を取得"""
    print(f"[INFO] 論文詳細情報を取得中: {paper_url}")
    
    # 1. API でタイトル・Abstract・著者名を取得
    api_data = fetch_paper_details_api(paper_url)
    
    # 2. ブラウザで ACM DL 論文個別ページから Figure 1 (メイン図表) 画像 URL と著者名を補完取得
    figure_url = None
    page_authors = ""
    abstract_text = api_data.get("abstract") if api_data else ""
    paper_title = api_data.get("title") if api_data else None

    try:
        page.get(paper_url)
        for i in range(10):
            if "Just a moment" in page.title or "しばらく" in page.title:
                time.sleep(1)
            else:
                break

        html = page.html
        soup = BeautifulSoup(html, "html.parser")

        # Abstract が API で取れなかった場合ブラウザから抽出
        if not abstract_text:
            abstract_elem = soup.find(class_=re.compile(r"abstractSection|abstractInFull|abstract-content")) or \
                            soup.find("section", id="abstract") or \
                            soup.find("div", class_=re.compile(r"abstract"))
            abstract_text = abstract_elem.get_text(strip=True) if abstract_elem else ""

        # 著者名が API で取れなかった場合ブラウザから抽出
        if not (api_data and api_data.get("authors")):
            authors_elems = soup.find_all(class_=re.compile(r"author-name|author|given-name"))
            if authors_elems:
                names = [a.get_text(strip=True) for a in authors_elems if len(a.get_text(strip=True)) > 2]
                page_authors = ", ".join(dict.fromkeys(names))

        # Figure 1 / メインティーザー画像の抽出
        fig_tags = soup.find_all(["figure", "div"], class_=re.compile(r"figure|teaser|graphical-abstract|article-figure", re.I))
        for ft in fig_tags:
            img = ft.find("img")
            if img and img.get("src"):
                src = img.get("src")
                figure_url = "https://dl.acm.org" + src if src.startswith("/") else src
                break

        if not figure_url:
            for img in soup.find_all("img"):
                src = img.get("src", "")
                if "/cms/10.1145/" in src or "/cms/attachment/" in src or "fig1" in src.lower() or "downloadFigures" in src:
                    figure_url = "https://dl.acm.org" + src if src.startswith("/") else src
                    break

    except Exception as e:
        print(f"[WARN] ブラウザでの詳細取得中にエラー (スキップして続行): {e}")

    authors_final = (api_data.get("authors") if api_data and api_data.get("authors") else page_authors) or "Authors Unknown"

    return {
        "title": paper_title,
        "abstract": abstract_text,
        "authors": authors_final,
        "figure_url": figure_url
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
def send_to_slack(session: str, url: str, title_en: str, authors_en: str, figure_url: Optional[str], summary: str) -> None:
    webhook_url = os.getenv("SLACK_WEBHOOK_URL")
    
    header_text = f"*📌 Session:* {session}\n*📄 Title (EN):* {title_en}\n*👥 Authors:* {authors_en}\n*🔗 URL:* {url}"

    if not webhook_url:
        print("[WARN] SLACK_WEBHOOK_URL 未設定のため画面に要約を出力します:\n")
        print(header_text)
        if figure_url:
            print(f"🖼️ Figure Image URL: {figure_url}")
        print("\n" + summary)
        return

    blocks = [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": header_text
            }
        }
    ]

    # 画像が存在すれば Slack Block Kit の image ブロックを挿入
    if figure_url:
        blocks.append({
            "type": "image",
            "image_url": figure_url,
            "alt_text": f"Figure 1 of {title_en[:30]}"
        })

    blocks.append({
        "type": "section",
        "text": {
            "type": "mrkdwn",
            "text": summary
        }
    })
    blocks.append({"type": "divider"})

    payload = {"blocks": blocks}

    res = requests.post(webhook_url, json=payload)
    if res.status_code == 200:
        print("[INFO] Slackへの送信が完了しました (タイトル/著者/画像付き)。")
    else:
        print(f"[ERROR] Slack送信エラー: {res.status_code} - {res.text}")


# ---------------------------------------------------------------------------
# メインルーチン
# ---------------------------------------------------------------------------
def main():
    history = load_history()
    
    co = ChromiumOptions()
    co.auto_port()
    
    if platform.system() == "Linux":
        chrome_binaries = ["/usr/bin/google-chrome", "/usr/bin/chromium-browser", "/usr/bin/chromium"]
        for b in chrome_binaries:
            if os.path.exists(b):
                co.set_browser_path(b)
                print(f"[INFO] Linux Chrome binary detected: {b}")
                break
        co.set_argument('--no-sandbox')
        co.set_argument('--disable-gpu')
        co.set_argument('--disable-dev-shm-usage')

    page = ChromiumPage(co)

    try:
        papers = fetch_papers_list(page)
        
        # もし Webスクレイピングが Cloudflare で0件となった場合、サンプルDOIリストから自動でロード
        if not papers:
            print("[INFO] Webスクレイピングがブロックされたため、API経由で論文リストを自動ロードします...")
            sample_doi_suffixes = [
                "3791766", "3791278", "3791899", "3790977", "3790735",
                "3790721", "3791725", "3791100", "3791200", "3791300"
            ]
            for suf in sample_doi_suffixes:
                u = f"https://dl.acm.org/doi/10.1145/3772318.{suf}"
                papers.append({
                    "title": f"Paper 10.1145/3772318.{suf}",
                    "url": u,
                    "session": "ACM CHI Conference"
                })

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
            
            paper_title_en = details.get("title") or paper["title"]
            paper_authors_en = details.get("authors") or "Authors Unknown"
            figure_url = details.get("figure_url")
            paper_text = details.get("abstract")
            
            if not paper_text or len(paper_text.strip()) < 50:
                print(f"[SKIP] テキスト(Abstract)が抽出できなかったためスキップします。")
                continue

            try:
                summary = summarize_with_gemini(paper_title_en, paper_text)
                send_to_slack(
                    session=paper["session"],
                    url=paper["url"],
                    title_en=paper_title_en,
                    authors_en=paper_authors_en,
                    figure_url=figure_url,
                    summary=summary
                )

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
