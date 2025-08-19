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
    title = soup.select_one("#news_title02") # 제목
    body = soup.select_one("#news_content") # 본문
    date = soup.select_one("#news_util01") # 날짜

    return {
        "url": url,
        "title": title.get_text(strip=True) if title else None,
        "date": date.get_text(strip=True) if date else datetime.now().strftime("%Y-%m-%d"),
        "content": body.get_text("\n", strip=True) if body else None,
    }


def scrape_rss(rss_url, output_file="articles.json"):
    # 단일 RSS 피드 파싱 및 저장
    feed = feedparser.parse(rss_url)
    articles = []
    for entry in feed.entries:
        url = entry.link
        try:
            article = scrape_article(url)
            articles.append(article)
            print(f"[DONE] {url}")
        except Exception as e:
            print(f"[ERROR] {url} - {e}")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)


def scrape_multiple_rss(rss_feeds, output_file="articles.json"):
    """
    여러 RSS 피드를 순회하여 기사 수집 후 단일 JSON으로 저장.
    rss_feeds: List[Tuple[str, str]]  # (피드명, URL)
    """
    seen_urls = set()
    all_articles = []

    for feed_name, rss_url in rss_feeds:
        try:
            print(f"[FEED] {feed_name} -> {rss_url}")
            feed = feedparser.parse(rss_url)
        except Exception as e:
            print(f"[ERROR][FEED PARSE] {feed_name} - {e}")
            continue

        for entry in getattr(feed, "entries", []):
            url = getattr(entry, "link", None)
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)

            try:
                article = scrape_article(url)
                # 수집한 카테고리 정보 포함
                article["feed"] = feed_name
                all_articles.append(article)
                print(f"[DONE] {feed_name} :: {url}")
            except Exception as e:
                print(f"[ERROR] {feed_name} :: {url} - {e}")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_articles, f, ensure_ascii=False, indent=2)


# 실행
if __name__ == "__main__":
    # 보안뉴스 전체/메인/세부 카테고리 RSS 전부 수집
    rss_list = [
        # 메인 카테고리 (mkind)
        ("SECURITY", "http://www.boannews.com/media/news_rss.xml?mkind=1"),
        ("IT", "http://www.boannews.com/media/news_rss.xml?mkind=2"),
        ("SAFETY", "http://www.boannews.com/media/news_rss.xml?mkind=4"),
        ("SecurityWorld", "http://www.boannews.com/media/news_rss.xml?mkind=5"),

        # 뉴스 카테고리 (kind)
        ("사건ㆍ사고", "http://www.boannews.com/media/news_rss.xml?kind=1"),
        ("공공ㆍ정책", "http://www.boannews.com/media/news_rss.xml?kind=2"),
        ("비즈니스", "http://www.boannews.com/media/news_rss.xml?kind=3"),
        ("국제", "http://www.boannews.com/media/news_rss.xml?kind=4"),
        ("테크", "http://www.boannews.com/media/news_rss.xml?kind=5"),
        ("오피니언", "http://www.boannews.com/media/news_rss.xml?kind=6"),

        # 세부 카테고리 (skind)
        ("긴급경보", "http://www.boannews.com/media/news_rss.xml?skind=5"),
        ("기획특집", "http://www.boannews.com/media/news_rss.xml?skind=7"),
        ("인터뷰", "http://www.boannews.com/media/news_rss.xml?skind=3"),
        ("보안컬럼", "http://www.boannews.com/media/news_rss.xml?skind=2"),
        ("보안정책", "http://www.boannews.com/media/news_rss.xml?skind=6"),
    ]

    # 단일 파일에 합쳐 저장 (중복 URL 제거)
    scrape_multiple_rss(rss_list, output_file="articles.json")
