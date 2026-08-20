import json
import os
import re
import sys
import time
from typing import Dict, List, Optional

from google import genai
import requests

# ---------------------------------------------------------------------------
# 設定値 (UIST 2025 用)
# ---------------------------------------------------------------------------
PROCEEDINGS_URL = "https://dl.acm.org/doi/proceedings/10.1145/3746059"
HISTORY_FILE = "processed_uist.json"  # UIST専用履歴ファイル
MAX_DAILY_PAPERS = 2  # 1日2件

DOI_START = 3747600
DOI_END = 3747800
TOTAL_PROCEEDINGS_PAPERS = DOI_END - DOI_START + 1  # 全201本


def load_history() -> List[str]:
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
        except Exception as e:
            print(f"[WARN] 履歴ファイルの読み込み失敗: {e}")
            return []
    return []


def save_history(history: List[str]) -> None:
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def fetch_paper_details_api(doi_suffix: int) -> Optional[Dict[str, str]]:
    doi = f"10.1145/3746059.{doi_suffix}"
    url = f"https://dl.acm.org/doi/{doi}"
    api_url = f"https://api.semanticscholar.org/graph/v1/paper/{doi}?fields=title,abstract,url,venue,authors"
    headers = {"User-Agent": "AcademicPaperNotifier/1.0"}

    try:
        r = requests.get(api_url, headers=headers, timeout=5)
        if r.status_code == 200:
            data = r.json()
            title = data.get("title") or ""
            abstract = data.get("abstract") or ""
            raw_authors = data.get("authors", [])
            authors = [a.get("name") for a in raw_authors if a.get("name")]
            authors_str = ", ".join(authors) if authors else "Authors Unknown"

            if abstract and len(abstract) > 50 and title:
                return {
                    "doi": doi,
                    "title": title.strip(),
                    "abstract": abstract.strip(),
                    "authors": authors_str,
                    "url": url
                }
    except Exception as e:
        pass
    return None


def summarize_with_gemini(title: str, text: str) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("環境変数 GEMINI_API_KEY が設定されていません。")

    client = genai.Client(api_key=api_key)

    prompt = f"""
以下の学術論文のテキスト（抄録/概要）を解析し、日本語で構造化要約を作成してください。

【厳格な見出し指示】:
・見出しには『◆ 一言概要』『◆ 研究の背景・課題』『◆ 提案手法・アプローチ』『◆ 主な結果・成果』のみを使用してください。
・見出しの後ろに『(1〜2行)』などの補足やカッコ書きは絶対に付けないでください。
・前置き文や挨拶文は一切出力せず、必ずいきなり「◆ 一言概要」から始めてください。

【論文タイトル】: {title}

【要約フォーマット】:
◆ 一言概要
[概要を記入]

◆ 研究の背景・課題
[背景・課題を記入]

◆ 提案手法・アプローチ
[手法・アプローチを記入]

◆ 主な結果・成果
[結果・成果を記入]

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
            
            if "◆" in raw_text:
                cleaned_text = "◆" + raw_text.split("◆", 1)[1]
            elif "■" in raw_text:
                cleaned_text = "◆" + raw_text.split("■", 1)[1]
            else:
                cleaned_text = raw_text

            cleaned_text = re.sub(r"[◆■]\s*一言概要\s*[\(（][^\)）]*[\)）]", "◆ 一言概要", cleaned_text)
            cleaned_text = re.sub(r"[◆■]\s*研究の背景・課題\s*[\(（][^\)）]*[\)）]", "◆ 研究の背景・課題", cleaned_text)
            cleaned_text = re.sub(r"[◆■]\s*提案手法・アプローチ\s*[\(（][^\)）]*[\)）]", "◆ 提案手法・アプローチ", cleaned_text)
            cleaned_text = re.sub(r"[◆■]\s*主な結果・成果\s*[\(（][^\)）]*[\)）]", "◆ 主な結果・成果", cleaned_text)
            
            cleaned_text = cleaned_text.replace("■ 一言概要", "◆ 一言概要")
            cleaned_text = cleaned_text.replace("■ 研究の背景・課題", "◆ 研究の背景・課題")
            cleaned_text = cleaned_text.replace("■ 提案手法・アプローチ", "◆ 提案手法・アプローチ")
            cleaned_text = cleaned_text.replace("■ 主な結果・成果", "◆ 主な結果・成果")

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


def send_to_slack(current_paper_num: int, total_count: int, url: str, title: str, authors: str, summary: str) -> None:
    webhook_url = os.getenv("SLACK_WEBHOOK_URL")
    
    header_text = (
        f"*🚀 順番:* {current_paper_num}本 / {total_count}本  💎 UIST2025\n"
        f"*📖 Title:* {title}\n"
        f"*✍️ Authors:* {authors}\n"
        f"*🌐 URL:* {url}"
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
        print(f"[INFO] Slackへの送信が完了しました ({current_paper_num}本 / {total_count}本)。")
    else:
        print(f"[ERROR] Slack送信エラー: {res.status_code} - {res.text}")


def main():
    history = load_history()
    processed_count = 0

    print(f"[INFO] UIST '25 プロシーディング全体 ({TOTAL_PROCEEDINGS_PAPERS}本) から有効な論文データを探索中...")

    for doi_suffix in range(DOI_START, DOI_END + 1):
        if processed_count >= MAX_DAILY_PAPERS:
            print(f"[INFO] 本日の上限 ({MAX_DAILY_PAPERS}件) に達したため終了します。")
            break

        paper_url = f"https://dl.acm.org/doi/10.1145/3746059.{doi_suffix}"

        if paper_url in history:
            continue

        details = fetch_paper_details_api(doi_suffix)
        if not details or not details.get("abstract"):
            continue

        if details["title"] in history or details["doi"] in history:
            print(f"[SKIP] タイトル/DOIが過去の送信履歴と重複しているためスキップ: {details['title'][:30]}")
            continue

        valid_paper_counter = len([h for h in history if h.startswith("https://")]) + 1

        print(f"\n==========================================")
        print(f"[処理開始 ({valid_paper_counter}本 / {TOTAL_PROCEEDINGS_PAPERS}本 | 本日 {processed_count + 1}/{MAX_DAILY_PAPERS}件)]: DOI 10.1145/3746059.{doi_suffix}")

        try:
            summary = summarize_with_gemini(details["title"], details["abstract"])
            send_to_slack(
                current_paper_num=valid_paper_counter,
                total_count=TOTAL_PROCEEDINGS_PAPERS,
                url=details["url"],
                title=details["title"],
                authors=details["authors"],
                summary=summary
            )

            history.append(paper_url)
            history.append(details["doi"])
            history.append(details["title"])
            save_history(history)
            
            processed_count += 1
            time.sleep(2)

        except Exception as e:
            print(f"[ERROR] 処理中にエラーが発生しました: {e}")

    print(f"\n[INFO] UIST '25 処理完了。本日新たに処理した論文数: {processed_count} 件")


if __name__ == "__main__":
    main()
