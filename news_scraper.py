import feedparser
import requests
from bs4 import BeautifulSoup
import json
from datetime import datetime


def scrape_article(url):
    # 개별 기사 페이지 크롤링
    res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    res.raise_for_status()
    soup = BeautifulSoup(res.text, "html.parser")

    # 제목, 본문, 날짜 추출
    title = soup.select_one("#news_title02") or soup.select_one("h4.tit")
    body = soup.select_one("#news_content") or soup.select_one("div.view_txt")
    date = soup.select_one("#news_util01") or soup.select_one("span.date")

    return {
        "url": url,
        "title": title.get_text(strip=True) if title else None,
        "date": date.get_text(strip=True) if date else datetime.now().strftime("%Y-%m-%d"),
        "content": body.get_text("\n", strip=True) if body else None,
    }


def scrape_rss(rss_url, output_file="rss_articles.json"):
    # RSS 피드 파싱
    feed = feedparser.parse(rss_url)
    articles = []

    # 각 기사 링크 방문 후 크롤링
    for entry in feed.entries:
        url = entry.link
        try:
            article = scrape_article(url)
            articles.append(article)
            print(f"[DONE] {url}")
        except Exception as e:
            print(f"[ERROR] {url} - {e}")

    # JSON 파일로 저장
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)


# 실행
if __name__ == "__main__":
    rss_url = "http://www.boannews.com/media/news_rss.xml?mkind=1"  # 보안뉴스 RSS 주소
    scrape_rss(rss_url)  # RSS 기반 기사 크롤링 실행
