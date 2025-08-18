import requests
from bs4 import BeautifulSoup
import json
from datetime import datetime

def scrape_article(url):
    res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    res.raise_for_status()

    soup = BeautifulSoup(res.text, "html.parser")

    # 보안뉴스 기준 (사이트마다 선택자 다름)
    title = soup.select_one("#news_title02")  # 제목
    body = soup.select_one("#news_content")  # 본문
    date = soup.select_one("#news_util01")  # 발행일

    data = {
        "url": url,
        "title": title.get_text(strip=True) if title else None,
        "date": date.get_text(strip=True) if date else datetime.now().strftime("%Y-%m-%d"),
        "content": body.get_text("\n", strip=True) if body else None,
    }
    return data

def scrape_multiple(urls, output_file="articles.jsonl"):
    with open(output_file, "w", encoding="utf-8") as f:
        for url in urls:
            try:
                article = scrape_article(url)
                f.write(json.dumps(article, ensure_ascii=False) + "\n")
                print(f"[DONE] {url}")
            except Exception as e:
                print(f"[ERROR] {url} - {e}")

# 실행 예시
if __name__ == "__main__":
    urls = [
        "https://www.boannews.com/media/view.asp?idx=138727",
        "https://www.boannews.com/media/view.asp?idx=135182",
        # 추가 URL 여기에 넣기
    ]
    scrape_multiple(urls, "articles.jsonl")
