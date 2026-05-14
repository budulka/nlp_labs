
import sys
import json
import time
import random
import subprocess
from datetime import datetime

def _pip(pkg):
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", pkg, "-q"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

DEPS = {
    "playwright": "playwright",
    "playwright-stealth": "playwright_stealth",
    "beautifulsoup4": "bs4",
    "pandas": "pandas",
    "tabulate": "tabulate",
    "transformers": "transformers",
    "torch": "torch",
    "lxml": "lxml",
}

for pkg, imp in DEPS.items():
    try:
        __import__(imp)
    except ImportError:
        _pip(pkg)

import pandas as pd
from bs4 import BeautifulSoup
from transformers import pipeline
from tabulate import tabulate
from playwright.sync_api import sync_playwright
from playwright_stealth import Stealth

PRODUCT_CATEGORIES = [
    ("Смартфони", "https://prom.ua/ua/Mobilnye-telefony"),
    ("Телевізори", "https://prom.ua/ua/Televizory"),
    ("Карти пам'яті", "https://prom.ua/ua/Karty-pamyati"),
]

PRODUCTS_PER_CATEGORY = 3
REVIEWS_PER_PRODUCT = 10

PAGE_LOAD_TIMEOUT = 30000

LABEL_MAP = {
    "LABEL_0": "negative",
    "LABEL_1": "neutral",
    "LABEL_2": "positive"
}

classifier = pipeline(
    task="sentiment-analysis",
    model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
    tokenizer="cardiffnlp/twitter-xlm-roberta-base-sentiment",
    device=-1,
)

def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")

def classify(text):
    try:
        r = classifier(text[:512])[0]
        label = LABEL_MAP.get(r["label"], r["label"]).lower()
        return label, round(float(r["score"]), 3)
    except:
        return "neutral", 0.0

def get_browser():
    playwright = sync_playwright().start()

    browser = playwright.chromium.launch(
        headless=True,
        args=["--no-sandbox", "--disable-dev-shm-usage"]
    )

    return playwright, browser

def new_page(browser):
    ctx = browser.new_context(
        locale="uk-UA",
        extra_http_headers={
            "Accept-Language": "uk-UA,uk;q=0.9"
        },
        user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
    )

    page = ctx.new_page()
    Stealth().apply_stealth_sync(page)

    return page

def scroll(page, times=2):
    for _ in range(times):
        page.evaluate("window.scrollBy(0, window.innerHeight)")
        time.sleep(1)

def get_product_urls(page, category_url, limit):
    try:
        page.goto(
            category_url,
            timeout=PAGE_LOAD_TIMEOUT,
            wait_until="networkidle"
        )

        page.wait_for_selector(
            '[data-qaid="qa_product_tile"]',
            timeout=15000
        )

        scroll(page)

    except:
        return []

    soup = BeautifulSoup(page.content(), "lxml")

    results = []

    for card in soup.select('[data-qaid="qa_product_tile"]')[:limit]:
        link_el = card.select_one('[data-qaid="product_link"]')

        if not link_el:
            continue

        name = link_el.get_text(strip=True)

        href = link_el.get("href", "")

        if href and not href.startswith("http"):
            href = "https://prom.ua" + href

        if href:
            results.append((name, href))

    return results

def get_product_reviews(page, product_name, product_url, category, limit):
    try:
        page.goto(
            product_url,
            timeout=PAGE_LOAD_TIMEOUT,
            wait_until="networkidle"
        )

        time.sleep(2)

        page.wait_for_selector(
            '[data-qaid="opinion_item"]',
            timeout=10000
        )

        scroll(page)

    except:
        return []

    soup = BeautifulSoup(page.content(), "lxml")

    rows = []

    for block in soup.select('[data-qaid="opinion_item"]')[:limit]:
        text = block.get_text(" ", strip=True)

        if len(text) < 5:
            continue

        sentiment, confidence = classify(text)

        rows.append({
            "group": category,
            "product": product_name,
            "text": text[:300],
            "sentiment": sentiment,
            "confidence": confidence,
            "url": product_url
        })

    return rows

def compute_stats(df):
    rows = []

    for name, grp in df.groupby("group"):
        total = len(grp)

        pos = (grp["sentiment"] == "positive").sum()
        neg = (grp["sentiment"] == "negative").sum()
        neu = (grp["sentiment"] == "neutral").sum()

        rows.append({
            "Категорія": name,
            "Відгуків": total,
            "Позитивні": pos,
            "Нейтральні": neu,
            "Негативні": neg,
            "% негативних": round(neg / total * 100, 1)
        })

    return pd.DataFrame(rows)

def print_table(df):
    print(
        tabulate(
            df.values.tolist(),
            headers=df.columns,
            tablefmt="simple"
        )
    )

def save(df, stats):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    df.to_csv(
        f"reviews_{ts}.csv",
        index=False,
        encoding="utf-8-sig"
    )

    stats.to_csv(
        f"stats_{ts}.csv",
        index=False,
        encoding="utf-8-sig"
    )

    report = {
        "generated": ts,
        "reviews": len(df),
        "stats": stats.to_dict(orient="records")
    }

    with open(
        f"report_{ts}.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

def main():
    print("\nАналізатор відгуків Prom.ua\n")

    playwright, browser = get_browser()

    page = new_page(browser)

    all_reviews = []

    try:
        for category_name, category_url in PRODUCT_CATEGORIES:
            log(f"Категорія: {category_name}")

            products = get_product_urls(
                page,
                category_url,
                PRODUCTS_PER_CATEGORY
            )

            for product_name, product_url in products:
                log(product_name[:60])

                reviews = get_product_reviews(
                    page,
                    product_name,
                    product_url,
                    category_name,
                    REVIEWS_PER_PRODUCT
                )

                all_reviews.extend(reviews)
                log(f"Відгуків: {len(reviews)}")
                time.sleep(random.uniform(1, 2))

    finally:
        browser.close()
        playwright.stop()

    if not all_reviews:
        print("Не вдалося отримати відгуки")
        return

    df = pd.DataFrame(all_reviews)

    stats = compute_stats(df)

    print("\nСТАТИСТИКА:\n")

    print_table(stats)

    save(df, stats)

    print("\nГотово\n")

if __name__ == "__main__":
    main()