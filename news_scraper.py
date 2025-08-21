import requests
import feedparser
import time
from bs4 import BeautifulSoup
from datetime import datetime

def scrape_article(url: str):
    """보안뉴스 기사 상세 스크래핑 (타이틀/본문/일자)"""
    try:
        res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=7)
        if res.status_code != 200:
            return None
        soup = BeautifulSoup(res.text, "html.parser")

        # 다양한 템플릿 대응
        title = soup.select_one("#news_title02") or soup.select_one("h4.tit")
        body = soup.select_one("#news_content") or soup.select_one("div.view_txt")
        date = soup.select_one("#news_util01") or soup.select_one("span.date")

        return {
            "url": url,
            "title": title.get_text(strip=True) if title else "제목 없음",
            "date": date.get_text(strip=True) if date else datetime.now().strftime("%Y-%m-%d"),
            "content": body.get_text("\n", strip=True) if body else "내용 없음",
            "source": "보안뉴스"
        }
    except Exception:
        return None

def fetch_latest_news_by_rss():
    """보안뉴스 RSS 여러 피드에서 최신 기사 수집"""
    rss_list = [
        ("SECURITY", "http://www.boannews.com/media/news_rss.xml?mkind=1"),
        ("IT", "http://www.boannews.com/media/news_rss.xml?mkind=2"),
        ("SAFETY", "http://www.boannews.com/media/news_rss.xml?mkind=4"),
        ("사건ㆍ사고", "http://www.boannews.com/media/news_rss.xml?kind=1"),
        ("공공ㆍ정책", "http://www.boannews.com/media/news_rss.xml?kind=2"),
        ("비즈니스", "http://www.boannews.com/media/news_rss.xml?kind=3"),
        ("국제", "http://www.boannews.com/media/news_rss.xml?kind=4"),
        ("테크", "http://www.boannews.com/media/news_rss.xml?kind=5"),
    ]
    seen_urls, seen_titles, all_articles = set(), set(), []
    for feed_name, rss_url in rss_list:
        try:
            feed = feedparser.parse(rss_url)
        except Exception:
            continue
        for entry in getattr(feed, "entries", []):
            url = getattr(entry, "link", None)
            title = getattr(entry, "title", "").strip()
            if not url or url in seen_urls or title in seen_titles:
                continue
            seen_urls.add(url)
            seen_titles.add(title)
            data = scrape_article(url)
            if data and data['title'] != '제목 없음':
                data["source"] = feed_name
                all_articles.append(data)
            time.sleep(0.1)
    return all_articles
