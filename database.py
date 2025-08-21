import sqlite3
import json
from datetime import datetime

def init_db():
    """데이터베이스 초기화 및 테이블 생성"""
    conn = sqlite3.connect('bookmarks.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS saved_news (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            url TEXT UNIQUE,
            summary TEXT,
            keywords TEXT,
            risk_level TEXT,
            risk_score REAL,
            saved_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS saved_playbooks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            summary TEXT,
            playbook_content TEXT,
            keywords TEXT,
            saved_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def save_news_to_favorites(news_item):
    """뉴스 기사를 즐겨찾기에 저장"""
    conn = sqlite3.connect('bookmarks.db')
    c = conn.cursor()
    try:
        c.execute('''
            INSERT INTO saved_news (title, url, summary, keywords, risk_level, risk_score)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            news_item['title'],
            news_item['url'],
            news_item['summary'],
            json.dumps(news_item['keywords'], ensure_ascii=False),
            news_item['risk_level'],
            news_item['risk_score']
        ))
        conn.commit()
        return True, f"'{news_item['title']}' 기사를 즐겨찾기에 추가했습니다."
    except sqlite3.IntegrityError:
        return False, f"'{news_item['title']}' 기사는 이미 즐겨찾기에 있습니다."
    finally:
        conn.close()

def get_saved_news():
    """저장된 뉴스 기사 목록 조회"""
    conn = sqlite3.connect('bookmarks.db')
    c = conn.cursor()
    c.execute("SELECT * FROM saved_news ORDER BY saved_at DESC")
    rows = c.fetchall()
    conn.close()
    return rows

def delete_news_from_favorites(news_id):
    """즐겨찾기에서 뉴스 기사 삭제"""
    conn = sqlite3.connect('bookmarks.db')
    c = conn.cursor()
    c.execute("DELETE FROM saved_news WHERE id = ?", (news_id,))
    conn.commit()
    conn.close()

def save_playbook_to_favorites(playbook_title, playbook_content, report_summary, llm_selected_keywords):
    """대응 플레이북을 즐겨찾기에 저장"""
    conn = sqlite3.connect('bookmarks.db')
    c = conn.cursor()
    try:
        clean_playbook_content = playbook_content.replace('<br>', '\n')
        c.execute('''
            INSERT INTO saved_playbooks (title, summary, playbook_content, keywords)
            VALUES (?, ?, ?, ?)
        ''', (
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}] {playbook_title}",
            report_summary,
            clean_playbook_content,
            json.dumps([k['keyword'] for k in llm_selected_keywords], ensure_ascii=False)
        ))
        conn.commit()
        return True, "대응 플레이북을 즐겨찾기에 추가했습니다."
    except sqlite3.IntegrityError:
        return False, "플레이북 저장 중 오류가 발생했습니다."
    finally:
        conn.close()

def get_saved_playbooks():
    """저장된 플레이북 목록 조회"""
    conn = sqlite3.connect('bookmarks.db')
    c = conn.cursor()
    c.execute("SELECT * FROM saved_playbooks ORDER BY saved_at DESC")
    rows = c.fetchall()
    conn.close()
    return rows

def delete_playbook_from_favorites(playbook_id):
    """즐겨찾기에서 플레이북 삭제"""
    conn = sqlite3.connect('bookmarks.db')
    c = conn.cursor()
    c.execute("DELETE FROM saved_playbooks WHERE id = ?", (playbook_id,))
    conn.commit()
    conn.close()
