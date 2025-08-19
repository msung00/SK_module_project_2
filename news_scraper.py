import requests
from bs4 import BeautifulSoup
import json
import csv
from datetime import datetime
import time

def scrape_article(url):
    # 뉴스 기사 크롤링
    try:
        res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=5)
        if res.status_code != 200:
            return None

        soup = BeautifulSoup(res.text, "html.parser")
        title = soup.select_one("#news_title02")  # 제목
        body = soup.select_one("#news_content")   # 본문
        date = soup.select_one("#news_util01")    # 날짜

        return {
            "url": url,
            "title": title.get_text(strip=True) if title else None,
            "date": date.get_text(strip=True) if date else datetime.now().strftime("%Y-%m-%d"),
            "content": body.get_text("\n", strip=True) if body else None,
        }
    except Exception:
        return None

def scrape_by_idx_range(start_idx, end_idx, json_file="articles.json", csv_file="articles.csv"):
    # idx 범위 뉴스 크롤링 → JSON + CSV 저장
    missing_idx = []
    count = 0

    with open(json_file, "w", encoding="utf-8") as f_json, \
            open(csv_file, "w", encoding="utf-8-sig", newline="") as f_csv:

        writer = csv.DictWriter(f_csv, fieldnames=["url", "title", "date", "content"])
        writer.writeheader()

        for idx in range(start_idx, end_idx + 1):
            url = f"https://www.boannews.com/media/view.asp?idx={idx}"
            article = scrape_article(url)
            if article:
                f_json.write(json.dumps(article, ensure_ascii=False) + "\n")
                writer.writerow(article)
                count += 1
                print(f"[DONE] {url}")
            else:
                missing_idx.append(idx)
                print(f"[MISSING] {url}")

            time.sleep(0.2)

    print(f"\n[INFO] Total collected articles: {count}")
    if missing_idx:
        print(f"[INFO] Missing idx: {missing_idx}")

if __name__ == "__main__":
    scrape_by_idx_range(138000, 138770, json_file="articles.json", csv_file="articles.csv")