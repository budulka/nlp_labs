#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, json, time, random, subprocess, re
from datetime import datetime

def _pip(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

DEPS = {
    "playwright":     "playwright",
    "requests":       "requests",
    "beautifulsoup4": "bs4",
    "pandas":         "pandas",
    "tabulate":       "tabulate",
    "pyttsx3":        "pyttsx3",
    "transformers":   "transformers",
    "torch":          "torch",
    "lxml":           "lxml",
}
for pkg, imp in DEPS.items():
    try:
        __import__(imp)
    except ImportError:
        print(f"  Встановлення {pkg}...")
        _pip(pkg)

import pandas as pd
from bs4 import BeautifulSoup
from tabulate import tabulate

PRODUCT_CATEGORIES = [
    ("Смартфони",     "https://prom.ua/ua/Mobilnye-telefony"),
    ("Телевізори",    "https://prom.ua/ua/Televizory"),
    ("Карти пам'яті", "https://prom.ua/ua/Karty-pamyati"),
]

SELLERS = [
    ("Техно Кошик", "https://prom.ua/ua/opinions/list/2487753"),
    ("Мегабайт",    "https://prom.ua/ua/opinions/list/3606600"),
    ("MobilaX",     "https://prom.ua/ua/opinions/list/2988553"),
    ("BestCases",   "https://prom.ua/ua/opinions/list/2664056"),
]

PRODUCTS_PER_CATEGORY = 1
REVIEWS_PER_PRODUCT   = 1
REVIEWS_PER_SELLER    = 30

CRITICAL_NEG_PCT = 40
WARN_NEG_PCT     = 25
CRITICAL_MIN     = 5

PAGE_LOAD_TIMEOUT = 30_000
SCROLL_PAUSE      = 1.5


def log(msg, level="INFO"):
    ts = datetime.now().strftime("%H:%M:%S")
    p = {"INFO": " ", "OK": "+", "WARN": "!", "ERR": "X"}.get(level, " ")
    print(f"[{ts}] [{p}] {msg}")


_tts = None

def speak(text: str):
    global _tts
    log(text, "WARN")
    try:
        import pyttsx3
        if _tts is None:
            _tts = pyttsx3.init()
            _tts.setProperty("rate", 155)
            _tts.setProperty("volume", 0.9)
            for v in _tts.getProperty("voices"):
                vid = (v.id or "").lower()
                if any(x in vid for x in ["uk", "ru", "ukr", "russian", "ukrainian"]):
                    _tts.setProperty("voice", v.id)
                    break
        _tts.say(text)
        _tts.runAndWait()
    except Exception as e:
        log(f"TTS помилка: {e}", "WARN")

def announce_critical(alerts: list):
    if not alerts:
        return
    speak("Увага! Виявлено критичні показники негативних відгуків.")
    for a in alerts[:5]:
        kind = "Продавець" if a["type"] == "seller" else "Група товарів"
        speak(f"{kind} {a['name']}: {a['neg_pct']} відсотків негативних відгуків.")
    if len(alerts) > 5:
        speak(f"Та ще {len(alerts) - 5} критичних випадків у звіті.")


log("Завантаження NLP моделі...")
from transformers import pipeline as hf_pipeline

_LABEL_MAP = {"LABEL_0": "negative", "LABEL_1": "neutral", "LABEL_2": "positive"}
_classifier = hf_pipeline(
    task="sentiment-analysis",
    model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
    tokenizer="cardiffnlp/twitter-xlm-roberta-base-sentiment",
    device=-1,
)
log("Модель завантажена", "OK")


def classify(text: str) -> tuple[str, float]:
    try:
        r = _classifier(text.lower()[:512])[0]
        label = r["label"].lower()   
        return label, round(float(r["score"]), 3)
    except:
        return "neutral", 0.0

_playwright = None
_browser    = None

def get_browser():
    global _playwright, _browser
    if _browser is None:
        from playwright.sync_api import sync_playwright
        _playwright = sync_playwright().start()
        _browser = _playwright.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage"],
        )
    return _browser

def close_browser():
    global _playwright, _browser
    if _browser:
        _browser.close()
        _browser = None
    if _playwright:
        _playwright.stop()
        _playwright = None

from playwright_stealth import Stealth

def new_page():
    ctx = get_browser().new_context(
        locale="uk-UA",
        extra_http_headers={"Accept-Language": "uk-UA,uk;q=0.9"},
        user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
    )
    page = ctx.new_page()
    Stealth().apply_stealth_sync(page)
    return page


def _scroll_and_wait(page, times: int = 3):
    for _ in range(times):
        page.evaluate("window.scrollBy(0, window.innerHeight)")
        time.sleep(SCROLL_PAUSE)


def get_product_urls(page, category_url: str, limit: int) -> list[tuple[str, str]]:
    log(f"  Каталог: {category_url}")
    try:
        page.goto(category_url, timeout=PAGE_LOAD_TIMEOUT, wait_until="networkidle")
        page.wait_for_selector('[data-qaid="qa_product_tile"]', timeout=15_000)
        _scroll_and_wait(page, 2)
    except Exception as e:
        log(f"  Не вдалося завантажити каталог: {e}", "ERR")
        return []

    soup = BeautifulSoup(page.content(), "lxml")
    results = []

    for card in soup.select('[data-qaid="qa_product_tile"]')[:limit]:
        link_el = card.select_one('[data-qaid="product_link"]')
        if not link_el:
            continue
        name_el = link_el.select_one('span')
        name = name_el.get_text(strip=True) if name_el else link_el.get_text(strip=True)
        href = link_el.get("href", "")
        if href and not href.startswith("http"):
            href = "https://prom.ua" + href
        if href and name:
            results.append((name, href))

    log(f"  Знайдено {len(results)} товарів", "OK" if results else "WARN")
    return results


def get_product_reviews(page, product_name: str, product_url: str,
                        category: str, limit: int) -> list[dict]:
    log(f"    Товар: {product_name[:50]}")
    try:
        page.goto(product_url, timeout=PAGE_LOAD_TIMEOUT, wait_until="networkidle")
        time.sleep(2)

        opinion_count = page.query_selector('[data-qaid="opinion_count"]')
        if opinion_count:
            if opinion_count.inner_text().strip() == "0":
                log("    Відгуків немає", "WARN")
                return []

        opinion_btn = page.query_selector('[data-qaid="product_opinion_block"]')
        if opinion_btn:
            opinion_btn.scroll_into_view_if_needed()
            time.sleep(1)

        page.wait_for_selector('[data-qaid="opinion_item"]', timeout=10_000)
        _scroll_and_wait(page, 2)
    except Exception:
        return []

    soup = BeautifulSoup(page.content(), "lxml")
    rows = []

    for block in soup.select('[data-qaid="opinion_item"]')[:limit]:
        pros = block.select_one('[data-qaid="pros_text"]')
        cons = block.select_one('[data-qaid="disadvantages_text"]')
        body = block.select_one('[data-qaid="opinion_body"]') or block.select_one('p')

        parts = []
        if pros: parts.append(pros.get_text(strip=True))
        if cons: parts.append(cons.get_text(strip=True))
        if body: parts.append(body.get_text(strip=True))
        text = " ".join(parts) if parts else block.get_text(strip=True)[:300]

        author_el = block.select_one('[data-qaid="opinion_author"]') or block.select_one('[data-qaid="prom_label_text"]')
        date_el   = block.select_one('time[datetime]') or block.select_one('[data-qaid="date_created"]')

        author = author_el.get_text(strip=True) if author_el else "Анонім"
        date   = (date_el.get("datetime") or date_el.get_text(strip=True))[:10] if date_el else ""

        if not text or len(text) < 5:
            continue

        sentiment, confidence = classify(text)
        rows.append({
            "type":       "product",
            "group":      category,
            "seller":     "",
            "product":    product_name[:80],
            "author":     author,
            "date":       date,
            "text":       text[:300],
            "sentiment":  sentiment,
            "confidence": confidence,
            "url":        product_url,
        })

    return rows


def get_seller_reviews(page, seller_name: str, seller_url: str, limit: int) -> list[dict]:
    log(f"  Продавець: {seller_name}")
    try:
        page.goto(seller_url, timeout=PAGE_LOAD_TIMEOUT, wait_until="networkidle")
        time.sleep(2)

        for _ in range(3):
            more_btn = page.query_selector('[data-qaid="show_more_opinions"]')
            if not more_btn:
                break
            more_btn.click()
            time.sleep(1.5)

        _scroll_and_wait(page, 2)
    except Exception as e:
        log(f"  Не вдалося завантажити відгуки продавця: {e}", "ERR")
        return []

    soup = BeautifulSoup(page.content(), "lxml")
    rows = []

    for block in soup.select('[data-qaid="opinion_text"]')[:limit]:
        text = block.get_text(strip=True)
        if not text or len(text) < 5:
            continue

        parent = block.parent
        for _ in range(5):
            if parent is None:
                break
            if parent.select_one('[data-qaid="author_name"]'):
                break
            parent = parent.parent

        author_el = parent.select_one('[data-qaid="author_name"]') if parent else None
        date_el   = parent.select_one('[data-qaid="date_created"]') if parent else None

        author = author_el.get_text(strip=True) if author_el else "Анонім"
        date   = date_el.get_text(strip=True)[:10] if date_el else ""

        sentiment, confidence = classify(text)
        rows.append({
            "type":       "seller",
            "group":      "",
            "seller":     seller_name,
            "product":    "",
            "author":     author,
            "date":       date,
            "text":       text[:300],
            "sentiment":  sentiment,
            "confidence": confidence,
            "url":        seller_url,
        })

    log(f"  Отримано {len(rows)} відгуків", "OK" if rows else "WARN")
    return rows


def compute_stats(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []
    for name, grp in df.groupby(group_col):
        total   = len(grp)
        pos     = (grp["sentiment"] == "positive").sum()
        neg     = (grp["sentiment"] == "negative").sum()
        neu     = (grp["sentiment"] == "neutral").sum()
        neg_pct = round(neg / total * 100, 1) if total else 0
        pos_pct = round(pos / total * 100, 1) if total else 0

        if total >= CRITICAL_MIN and neg_pct >= CRITICAL_NEG_PCT:
            status = "КРИТИЧНО"
        elif neg_pct >= WARN_NEG_PCT:
            status = "Увага"
        else:
            status = "Норма"

        rows.append({
            "Назва":      name,
            "Відгуків":   total,
            "Позитивні":  pos,
            "Нейтральні": neu,
            "Негативні":  neg,
            "% позит.":   pos_pct,
            "% негат.":   neg_pct,
            "Статус":     status,
        })
    return pd.DataFrame(rows).sort_values("% негат.", ascending=False)


def print_table(title: str, df: pd.DataFrame):
    print(f"\n{'='*70}\n  {title}\n{'='*70}")
    if df.empty:
        print("  (немає даних)")
        return
    rows = [[str(v)[:38] if isinstance(v, str) else v for v in r]
            for r in df.values.tolist()]
    print(tabulate(rows, headers=list(df.columns), tablefmt="simple"))


def save(df: pd.DataFrame, s_sellers: pd.DataFrame, s_groups: pd.DataFrame, alerts: list):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    df.to_csv(        f"prom_reviews_{ts}.csv",  index=False, encoding="utf-8-sig")
    s_sellers.to_csv( f"prom_sellers_{ts}.csv",  index=False, encoding="utf-8-sig")
    s_groups.to_csv(  f"prom_products_{ts}.csv", index=False, encoding="utf-8-sig")
    report = {
        "generated":              ts,
        "source":                 "prom.ua",
        "total_reviews":          len(df),
        "critical_alerts":        alerts,
        "stats_by_seller":        s_sellers.to_dict(orient="records"),
        "stats_by_product_group": s_groups.to_dict(orient="records"),
    }
    with open(f"prom_report_{ts}.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    log("Файли збережено:", "OK")
    for fn in [f"prom_reviews_{ts}.csv", f"prom_sellers_{ts}.csv",
               f"prom_products_{ts}.csv", f"prom_report_{ts}.json"]:
        print(f"   >> {fn}")


def main():
    print(f"\n{'='*70}")
    print("  Аналізатор відгуків Prom.ua")
    print(f"  Категорій товарів: {len(PRODUCT_CATEGORIES)}  |  Продавців: {len(SELLERS)}")
    print(f"  Критичний поріг: {CRITICAL_NEG_PCT}%")
    print(f"{'='*70}\n")

    all_reviews = []
    page = new_page()

    try:
        log("=== ЗБІР ВІДГУКІВ ПРО ТОВАРИ ===")
        for cat_name, cat_url in PRODUCT_CATEGORIES:
            log(f"Категорія: {cat_name}")
            products = get_product_urls(page, cat_url, PRODUCTS_PER_CATEGORY)
            for prod_name, prod_url in products:
                reviews = get_product_reviews(page, prod_name, prod_url,
                                              cat_name, REVIEWS_PER_PRODUCT)
                all_reviews.extend(reviews)
                log(f"    +{len(reviews)} відгуків", "OK" if reviews else "WARN")
                time.sleep(random.uniform(1.0, 2.0))

        log("=== ЗБІР ВІДГУКІВ ПРО ПРОДАВЦІВ ===")
        for seller_name, seller_url in SELLERS:
            reviews = get_seller_reviews(page, seller_name, seller_url, REVIEWS_PER_SELLER)
            all_reviews.extend(reviews)
            time.sleep(random.uniform(1.5, 2.5))

    finally:
        close_browser()

    if not all_reviews:
        log("Не вдалося отримати жодного відгуку.", "ERR")
        sys.exit(1)

    df = pd.DataFrame(all_reviews)
    log(f"Зібрано: {len(df)} відгуків | "
        f"{df['seller'].replace('', pd.NA).dropna().nunique()} продавців | "
        f"{df['group'].replace('', pd.NA).dropna().nunique()} груп товарів", "OK")

    df_sellers  = df[df["type"] == "seller"]
    df_products = df[df["type"] == "product"]

    s_sellers = compute_stats(df_sellers,  "seller") if not df_sellers.empty  else pd.DataFrame()
    s_groups  = compute_stats(df_products, "group")  if not df_products.empty else pd.DataFrame()

    print_table("СТАТИСТИКА ПО ПРОДАВЦЯХ",      s_sellers)
    print_table("СТАТИСТИКА ПО ГРУПАХ ТОВАРІВ", s_groups)

    alerts = []
    for stats, atype in [(s_sellers, "seller"), (s_groups, "product_group")]:
        if stats.empty:
            continue
        for _, row in stats.iterrows():
            if row["Статус"] == "КРИТИЧНО":
                alerts.append({"type": atype, "name": row["Назва"],
                               "neg_pct": int(row["% негат."])})

    if alerts:
        print(f"\n{'='*70}\n  КРИТИЧНІ ПОКАЗНИКИ ({len(alerts)} шт.)\n{'='*70}")
        for a in alerts:
            kind = "Продавець" if a["type"] == "seller" else "Група товарів"
            print(f"  {kind}: {a['name']} -- {a['neg_pct']}% негативних відгуків")
        announce_critical(alerts)
    else:
        print("\n  Критичних показників не виявлено.")

    save(df, s_sellers, s_groups, alerts)
    print("\nГотово!\n")


if __name__ == "__main__":
    main()