import asyncio
import argparse
import os
import json
import re
import sqlite3
import csv
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import torch.nn.functional as F

load_dotenv()

#Constants

SECTIONS_FILE = "sections.json"
DB_FILE = "wsj.db"
MODEL_NAME = "ProsusAI/finbert"
LABELS = ["positive", "negative", "neutral"]
ARTICLE_PATTERN = re.compile(
    r"https://www\.wsj\.com/(articles|business|finance|tech|us-news|politics|economy|world)/.+-[a-f0-9]{6,}$"
)

#Database

def init_db():
    conn = sqlite3.connect(DB_FILE)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS articles (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            url              TEXT    NOT NULL UNIQUE,
            headline         TEXT,
            byline           TEXT,
            date             TEXT,
            section          TEXT,
            body             TEXT,
            paragraphs       INTEGER,
            scraped_at       TEXT    NOT NULL,
            sentiment_label  TEXT,
            sentiment_score  REAL,
            sentiment_detail TEXT,
            analyzed_at      TEXT
        )
    """)
    conn.commit()
    conn.close()

def get_completed_urls():
    conn = sqlite3.connect(DB_FILE)
    rows = conn.execute("SELECT url FROM articles").fetchall()
    conn.close()
    return {row[0] for row in rows}

def insert_article(article, section):
    conn = sqlite3.connect(DB_FILE)
    conn.execute("""
        INSERT OR IGNORE INTO articles
            (url, headline, byline, date, section, body, paragraphs, scraped_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        article["url"],
        article["headline"],
        article["byline"],
        article["date"],
        section,
        article["body"],
        article["paragraphs"],
        datetime.now().isoformat(),
    ))
    conn.commit()
    conn.close()

def count_articles(section=None, date=None):
    conn = sqlite3.connect(DB_FILE)
    query = "SELECT COUNT(*) FROM articles WHERE 1=1"
    params = []
    if section:
        query += " AND section = ?"
        params.append(section)
    if date:
        query += " AND date LIKE ?"
        params.append(f"%{date}%")
    count = conn.execute(query, params).fetchone()[0]
    conn.close()
    return count

def get_unanalyzed_articles(section=None):
    conn = sqlite3.connect(DB_FILE)
    query = "SELECT id, headline, body FROM articles WHERE sentiment_label IS NULL"
    params = []
    if section:
        query += " AND section = ?"
        params.append(section)
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return rows

def save_sentiment(article_id, result):
    conn = sqlite3.connect(DB_FILE)
    conn.execute("""
        UPDATE articles
        SET sentiment_label = ?,
            sentiment_score = ?,
            sentiment_detail = ?,
            analyzed_at = ?
        WHERE id = ?
    """, (
        result["label"],
        result["score"],
        json.dumps(result),
        datetime.now().isoformat(),
        article_id,
    ))
    conn.commit()
    conn.close()

def load_wsj_cookies():
    storage_path = os.path.expanduser("~/.crawl4ai/profiles/wsj/storage_state.json")
    if not os.path.exists(storage_path):
        return []
    with open(storage_path) as f:
        state = json.load(f)
    return state.get("cookies", [])

#Trends 

def get_sentiment_trend(section=None, days=365):
    conn = sqlite3.connect(DB_FILE)
    query = "SELECT date, sentiment_score FROM articles WHERE sentiment_label IS NOT NULL"
    params = []
    if section:
        query += " AND section = ?"
        params.append(section)
    rows = conn.execute(query, params).fetchall()
    conn.close()

    from collections import defaultdict
    daily = defaultdict(list)

    for date_str, score in rows:
        if not date_str or score is None:
            continue
        try:
            clean = date_str.replace("Updated", "").strip()
            clean = clean.replace("ET", "").strip()
            parts = clean.rsplit(" ", 2)
            if len(parts) == 3 and ":" in parts[1]:
                clean = parts[0]
            dt = datetime.strptime(clean.strip(), "%B %d, %Y")
            daily[dt.strftime("%Y-%m-%d")].append(score)
        except ValueError:
            continue

    cutoff = datetime.now() - timedelta(days=days)
    trend  = []
    for day, scores in sorted(daily.items()):
        try:
            dt = datetime.strptime(day, "%Y-%m-%d")
        except ValueError:
            continue
        if dt < cutoff:
            continue
        avg   = round(sum(scores) / len(scores), 3)
        label = "positive" if avg > 0.05 else "negative" if avg < -0.05 else "neutral"
        trend.append({
            "date":      day,
            "avg_score": avg,
            "label":     label,
            "articles":  len(scores),
        })
    return trend

def print_trend(trend, section=None, export=False):
    if not trend:
        print("No trend data found. Scrape and analyze more articles first.")
        return

    icons = {"positive": "📈", "negative": "📉", "neutral": "➡️"}
    section_label = section or "all sections"

    print(f"\n━━━ {len(trend)}-Day Sentiment Trend — {section_label} ━━━━━━━━━━━━━━━")
    print(f"{'Date':<14} {'Score':>8}  {'Label':<10} {'Articles':>8}  Chart")
    print(f"{'─'*14} {'─'*8}  {'─'*10} {'─'*8}  {'─'*20}")

    for row in trend:
        icon = icons.get(row["label"], "")
        bar_pos = int((row["avg_score"] + 1) / 2 * 20)
        bar = "─" * bar_pos + "│" + "─" * (20 - bar_pos)
        print(f"{row['date']:<14} {row['avg_score']:>+8.3f}  {icon} {row['label']:<8} {row['articles']:>8}  {bar}")

    scores = [r["avg_score"] for r in trend]
    print(f"\nOverall avg : {sum(scores)/len(scores):+.3f}")
    print(f"Most bearish: {min(trend, key=lambda r: r['avg_score'])['date']} ({min(scores):+.3f})")
    print(f"Most bullish: {max(trend, key=lambda r: r['avg_score'])['date']} ({max(scores):+.3f})")
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    if export:
        filename = f"wsj_trend_{section or 'all'}.csv"
        with open(filename, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["date", "avg_score", "label", "articles"])
            writer.writeheader()
            writer.writerows(trend)
        print(f"Exported to {filename}")

#Query 

def query_articles(section=None, date_arg=None, keyword=None, sentiment=None, sort_sentiment=False):
    conn = sqlite3.connect(DB_FILE)
    query = """
        SELECT url, headline, byline, date, section, body, paragraphs,
               sentiment_label, sentiment_score
        FROM articles WHERE 1=1
    """
    params = []

    if section:
        query += " AND section = ?"
        params.append(section)
    if date_arg:
        target_dates = resolve_date(date_arg)
        date_clauses = " OR ".join("date LIKE ?" for _ in target_dates)
        query += f" AND ({date_clauses})"
        params.extend(f"%{d}%" for d in target_dates)
    if keyword:
        query += " AND (headline LIKE ? OR body LIKE ?)"
        params.extend([f"%{keyword}%", f"%{keyword}%"])
    if sentiment:
        query += " AND sentiment_label = ?"
        params.append(sentiment)

    query += " ORDER BY sentiment_score ASC" if sort_sentiment else " ORDER BY date DESC"
    rows = conn.execute(query, params).fetchall()
    conn.close()

    return [
        {
            "url": row[0],
            "headline": row[1],
            "byline": row[2],
            "date": row[3],
            "section": row[4],
            "body": row[5],
            "paragraphs": row[6],
            "sentiment_label": row[7],
            "sentiment_score": row[8],
        }
        for row in rows
    ]

def print_query_results(articles, export=False, export_file=None, sentiment_report=False):
    if not articles:
        print("No articles found matching your query.")
        return

    icons = {"positive": "📈", "negative": "📉", "neutral": "➡️"}

    if sentiment_report:
        groups = {"positive": [], "negative": [], "neutral": []}
        for a in articles:
            if a["sentiment_label"] in groups:
                groups[a["sentiment_label"]].append(a)
        pos = groups["positive"]
        neg = groups["negative"]
        neu  = groups["neutral"]
        scores = [a["sentiment_score"] for a in articles if a["sentiment_score"] is not None]
        avg  = round(sum(scores) / len(scores), 3) if scores else 0
        print(f"\n━━━ Sentiment Report ━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"Total articles : {len(articles)}")
        print(f"Average score  : {avg:+.3f}")
        print(f"📈 Positive    : {len(pos)}")
        print(f"📉 Negative    : {len(neg)}")
        print(f"➡️  Neutral     : {len(neu)}")
        if pos:
            best = max(pos, key=lambda a: a["sentiment_score"])
            print(f"\nMost Positive  : {best['headline'][:60]} ({best['sentiment_score']:+.3f})")
        if neg:
            worst = min(neg, key=lambda a: a["sentiment_score"])
            print(f"Most Negative  : {worst['headline'][:60]} ({worst['sentiment_score']:+.3f})")
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        return

    print(f"\n━━━ {len(articles)} articles found ━━━━━━━━━━━━━━━━━━━━━━━━")
    for i, article in enumerate(articles, 1):
        icon = icons.get(article.get("sentiment_label"), "")
        score = f" ({article['sentiment_score']:+.3f})" if article.get("sentiment_score") is not None else ""
        print(f"\n[{i}] {icon} {article['headline']}{score}")
        print(f"{article['byline']} — {article['date']}")
        print(f"{article['section']} | {article['paragraphs']} paragraphs")
        print(f"{article['url']}")

    if export:
        with open(export_file, "w") as f:
            json.dump(articles, f, indent=2)
        print(f"\n━━━ Exported to {export_file} ━━━━━━━━━━━━━━━━━━━━━━━━")

# ── Config ───────────────────────────────────────────────────────────────────

def load_sections():
    with open(SECTIONS_FILE) as f:
        return json.load(f)

def resolve_section(section_name):
    sections = load_sections()
    if section_name not in sections:
        available = ", ".join(sections.keys())
        raise ValueError(f"Unknown section '{section_name}'. Available: {available}")
    return sections[section_name]

def clean_url(url):
    return url.split("?")[0]

#Date

def resolve_date(date_arg):
    today = datetime.now()
    if date_arg == "today":
        return [today.strftime("%B %-d, %Y")]
    if date_arg == "week":
        return [(today - timedelta(days=i)).strftime("%B %-d, %Y") for i in range(7)]
    try:
        dt = datetime.strptime(date_arg, "%Y-%m-%d")
        return [dt.strftime("%B %-d, %Y")]
    except ValueError:
        raise ValueError(f"Invalid date '{date_arg}'. Use 'today', 'week', or 'YYYY-MM-DD'.")

def matches_date(article_date, target_dates):
    if not target_dates:
        return True
    return any(target in article_date for target in target_dates)

def build_archive_url(date_str):
    if date_str == "today":
        dt = datetime.now()
    else:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
    return f"https://www.wsj.com/news/archive/{dt.strftime('%Y/%m/%d')}"

#Browser

def make_browser_config():
    """Managed browser for article crawling — most stable approach."""
    profile_path = os.path.expanduser("~/.crawl4ai/profiles/wsj")
    return BrowserConfig(
        headless=True,
        use_managed_browser=True,
        browser_type="chromium",
        user_data_dir=profile_path,
    )

def make_harvest_browser_config(cookies=None):
    """Browserless for archive/section harvesting — bypasses DataDome headlessly."""
    token = os.environ.get("BROWSERLESS_TOKEN")
    if token:
        return BrowserConfig(
            headless=True,
            browser_mode="cdp",
            cdp_url=f"wss://production-sfo.browserless.io/chromium?token={token}",
            cookies=cookies or [],
        )
    # Fallback to local browser if no token
    profile_path = os.path.expanduser("~/.crawl4ai/profiles/wsj")
    return BrowserConfig(
        headless=False,
        use_managed_browser=True,
        browser_type="chromium",
        user_data_dir=profile_path,
        cookies=cookies or [],
    )

def make_run_config():
    return CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        page_timeout=60000,
        delay_before_return_html=8.0,
    )

#Sentiment

def load_model():
    print("Loading FinBERT model...")
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    print("Model ready.")
    return tokenizer, model

def analyze_text(text, tokenizer, model):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )
    with torch.no_grad():
        outputs = model(**inputs)
    probs = F.softmax(outputs.logits, dim=-1)[0].tolist()
    label = LABELS[probs.index(max(probs))]
    score = round(probs[0] - probs[1], 4)
    return {
        "label": label,
        "score": score,
        "positive": round(probs[0], 4),
        "negative": round(probs[1], 4),
        "neutral": round(probs[2], 4),
    }

def analyze_article(headline, body, tokenizer, model):
    paragraphs = [p.strip() for p in body.split("\n\n") if len(p.strip()) > 30]
    headline_result = analyze_text(headline, tokenizer, model)
    paragraph_results = [
        {"text": p[:100], **analyze_text(p, tokenizer, model)}
        for p in paragraphs[:20]
    ]
    all_scores = [headline_result["score"]] * 2 + [r["score"] for r in paragraph_results]
    avg_score  = round(sum(all_scores) / len(all_scores), 4) if all_scores else 0.0
    label = "positive" if avg_score > 0.05 else "negative" if avg_score < -0.05 else "neutral"
    return {
        "label": label,
        "score": avg_score,
        "headline": headline_result,
        "paragraphs": paragraph_results,
    }

def run_analysis(section=None):
    articles = get_unanalyzed_articles(section)
    if not articles:
        print("No unanalyzed articles found.")
        return
    print(f"Analyzing {len(articles)} articles...")
    tokenizer, model = load_model()
    icons = {"positive": "📈", "negative": "📉", "neutral": "➡️"}
    for i, (article_id, headline, body) in enumerate(articles):
        if not body:
            continue
        result = analyze_article(headline or "", body, tokenizer, model)
        save_sentiment(article_id, result)
        print(f"  {icons.get(result['label'], '')} [{i+1}/{len(articles)}] {headline[:60]} → {result['label']} ({result['score']:+.3f})")
    print(f"\nDone. {len(articles)} articles analyzed.")

#Extraction

def extract_article(result):
    soup = BeautifulSoup(result.html, "html.parser")

    paragraphs = soup.find_all(attrs={"data-testid": "paragraph"})
    if not paragraphs:
        paragraphs = soup.find_all(attrs={"data-type": "paragraph"})
    body = "\n\n".join(p.get_text(strip=True) for p in paragraphs)

    headline_el = soup.find(attrs={"data-testid": "headline"}) or soup.find("h1")
    headline = headline_el.get_text(strip=True) if headline_el else ""

    byline_el = (
        soup.find(attrs={"data-testid": "author-link"}) or
        soup.find(attrs={"data-testid": "byline"})
    )
    byline = byline_el.get_text(strip=True) if byline_el else ""

    date_el = soup.find(attrs={"data-testid": "timestamp-text"})
    date = date_el.get_text(strip=True) if date_el else ""

    return {
        "url":        result.url,
        "headline":   headline,
        "byline":     byline,
        "date":       date,
        "body":       body,
        "paragraphs": len(paragraphs),
    }

#Harvesting

async def harvest_links(section_url):
    cookies = load_wsj_cookies()
    run_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        page_timeout=60000,
        delay_before_return_html=8.0,
    )
    async with AsyncWebCrawler(config=make_harvest_browser_config(cookies=cookies)) as crawler:
        result = await crawler.arun(url=section_url, config=run_config)

    urls = re.findall(r'https://www\.wsj\.com[^\s\)\"\']+', result.markdown or "")
    return list({clean_url(u) for u in urls if ARTICLE_PATTERN.match(clean_url(u))})

async def harvest_archive_links(date_arg):
    if date_arg == "week":
        archive_urls = [
            build_archive_url((datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d"))
            for i in range(7)
        ]
    else:
        archive_urls = [build_archive_url(date_arg)]

    cookies = load_wsj_cookies()
    run_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        page_timeout=60000,
        delay_before_return_html=8.0,
    )

    all_urls = []
    async with AsyncWebCrawler(config=make_harvest_browser_config(cookies=cookies)) as crawler:
        for archive_url in archive_urls:
            result = await crawler.arun(url=archive_url, config=run_config)
            soup = BeautifulSoup(result.html, "html.parser")
            links = [clean_url(a.get("href", "")) for a in soup.find_all("a", href=True)]
            all_urls.extend(l for l in links if ARTICLE_PATTERN.match(l))

    return list(set(all_urls))

#Scrape

async def run(section_name, date_arg):
    section_url = resolve_section(section_name)
    target_dates = resolve_date(date_arg) if date_arg else None
    section_slug = section_url.replace("https://www.wsj.com/", "").split("/")[0]

    print(f"━━━ WSJ Scraper ━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"Section : {section_name}")
    print(f"Date    : {date_arg or 'all'}")
    print(f"Database: {DB_FILE}")
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    init_db()

    print("\n[1/3] Harvesting article URLs...")
    if date_arg:
        all_urls = await harvest_archive_links(date_arg)
    else:
        all_urls = await harvest_links(section_url)

    urls = [u for u in all_urls if f"wsj.com/{section_slug}/" in u or "wsj.com/articles/" in u]
    print(f"      {len(urls)} articles found in '{section_name}'")

    if not urls:
        print("No articles to crawl. Exiting.")
        return

    completed = get_completed_urls()
    remaining = [u for u in urls if u not in completed]
    print(f"\n[2/3] Resuming: {len(completed)} in DB, {len(remaining)} remaining")

    if not remaining:
        print("All articles already in DB.")
        return

    print(f"\n[3/3] Crawling articles...")
    saved   = 0
    skipped = 0
    failed  = []

    async with AsyncWebCrawler(config=make_browser_config()) as crawler:
        for i, url in enumerate(remaining):
            try:
                result = await crawler.arun(url=url, config=make_run_config())

                if result.success and result.html:
                    article = extract_article(result)

                    if matches_date(article["date"], target_dates):
                        insert_article(article, section_name)
                        saved += 1
                        print(f"  ✅ [{i+1}/{len(remaining)}] {article['headline'][:60]}")
                    else:
                        skipped += 1
                        print(f"  ⏭  [{i+1}/{len(remaining)}] Skipped: {article['date']}")
                else:
                    failed.append(url)
                    print(f"  ❌ [{i+1}/{len(remaining)}] Failed: {url}")

            except Exception as e:
                failed.append(url)
                print(f"  ❌ [{i+1}/{len(remaining)}] Error: {str(e)[:80]}")

            await asyncio.sleep(3)

    total_in_db = count_articles()
    print(f"\n━━━ Done ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"Saved      : {saved} articles → {DB_FILE}")
    print(f"Skipped    : {skipped}")
    print(f"Failed     : {len(failed)}")
    print(f"Total in DB: {total_in_db} articles")
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

#Main

def main():
    parser = argparse.ArgumentParser(description="WSJ Scraper")
    parser.add_argument("--section", help="Section: finance, tech, markets, economy, politics, business, us-news, world")
    parser.add_argument("--date", default=None, help="Date: 'today', 'week', or 'YYYY-MM-DD'")
    parser.add_argument("--query", action="store_true", help="Query DB instead of scraping")
    parser.add_argument("--analyze", action="store_true", help="Run sentiment analysis on unanalyzed articles")
    parser.add_argument("--trends", action="store_true", help="Show sentiment trend over time")
    parser.add_argument("--days", type=int, default=30, help="Number of days for trend (default: 30)")
    parser.add_argument("--keyword", help="Filter by keyword in headline or body")
    parser.add_argument("--sentiment", choices=["positive", "negative", "neutral"], help="Filter by sentiment label")
    parser.add_argument("--sort-sentiment", action="store_true", help="Sort by sentiment score (most negative first)")
    parser.add_argument("--sentiment-report", action="store_true", help="Show sentiment summary report")
    parser.add_argument("--export", action="store_true", help="Export results to file")
    args = parser.parse_args()

    if args.query:
        init_db()
        export_file = f"wsj_export_{args.section or 'all'}_{args.date or 'all'}.json"
        articles = query_articles(
            section=args.section,
            date_arg=args.date,
            keyword=args.keyword,
            sentiment=args.sentiment,
            sort_sentiment=args.sort_sentiment,
        )
        print_query_results(
            articles,
            export=args.export,
            export_file=export_file,
            sentiment_report=args.sentiment_report,
        )

    elif args.analyze:
        init_db()
        run_analysis(section=args.section)

    elif args.trends:
        init_db()
        trend = get_sentiment_trend(section=args.section, days=args.days)
        print_trend(trend, section=args.section, export=args.export)

    else:
        if not args.section:
            parser.error("--section is required when scraping")
        asyncio.run(run(args.section, args.date or "today"))

if __name__ == "__main__":
    main()