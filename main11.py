#!/usr/bin/env python3
"""
RSS Feed Processor

All articles from all feeds go to Mistral in batches of 100.
Mistral classifies each headline into signal or noise and deduplicates them.

Output:  curated_feed.xml
Stats:   fetch_stats.json
"""

import feedparser
import json
import os
import time
import re
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
import xml.etree.ElementTree as ET
from mistralai.client import Mistral
from email.utils import parsedate_to_datetime
from urllib.parse import urljoin, urlparse
import requests

try:
    from dateutil import parser as dateutil_parser
except Exception:
    dateutil_parser = None

# -- FEEDS ---------------------------------------------------------------------

FEED_URLS = [
    "https://tbsnews.net/top-news/rss.xml",
    "https://evilgodfahim.github.io/ds/business.xml",
    "https://en.prothomalo.com/feed/",
    "https://evilgodfahim.github.io/tbs/articles.xml",
    "https://evilgodfahim.github.io/dstar/feeds/feed.xml",
    "https://evilgodfahim.github.io/bd24ar/feeds/feed.xml",
    "https://evilgodfahim.github.io/dt/home.xml",
    "https://evilgodfahim.github.io/fen/fe_homepage.xml",
    "https://evilgodfahim.github.io/fen/fe_today.xml",
    "https://evilgodfahim.github.io/wl/pau.xml",
    "https://evilgodfahim.github.io/ds/todays_news.xml",
    "https://evilgodfahim.github.io/bben/feed.xml",
    "https://evilgodfahim.github.io/wl/reuters.xml"
]

EXISTING_API_FEEDS = set(FEED_URLS)
KL_API_FEEDS = set()

# -- CONFIG --------------------------------------------------------------------

MISTRAL_MODEL = "mistral-medium-latest"

PROCESSED_FILE = "processed_articles.json"
SELECTED_FILE = "selected_articles.json"
OUTPUT_XML = "curated_feed.xml"
EXCLUDED_XML = "ex.xml"
STATS_FILE = "fetch_stats.json"

MAX_ARTICLES_PER_FEED = 100
MAX_AGE_HOURS = 10
ALLOW_MISSING_DATES = True
ALLOW_OLDER = False
MAX_FEED_ITEMS = 500

BATCH_SIZE = 50
BATCH_INTERVAL_SECONDS = 5

# -- PROMPT --------------------------------------------------------------------

PROMPT = """Classify numbered English news titles as SIGNAL or NOISE. Return SIGNAL indices only. Bar: SUPER HIGH.

NOISE — any of:
- Sports, entertainment, celebrity, lifestyle
- Single discrete incident: arrest, clash, crime, accident, fire, death, protest
- Tribute, anniversary, opinion, editorial, praise, or criticism
- Affects only one district, institution, or individual

SIGNAL — Bangladesh — national scale only:
- Central bank actions, national budget, fuel/utility/forex changes, trade/remittance data, IMF/World Bank actions on BD
- Cabinet/parliament/supreme court decisions, nationwide policy rollouts, election commission rulings
- Countrywide infrastructure failure, or national/divisional disaster or health emergency
- Foreign affairs: bilateral talks, sanctions on BD, cross-border disputes (Teesta, Rohingya), BD at UN/IMF/WTO, foreign loans formally signed

SIGNAL — International — concrete cross-border impact only:
- Active armed conflict between states, or war/ceasefire declared
- UNSC resolution, IMF/WB program approval, WTO ruling, NATO formal decision, IAEA finding, ICC/ICJ verdict
- Formal multilateral treaty signed or collapsed
- Single-country decision disrupting global energy supply, major financial system collapse, verified nuclear milestone
- Foreign domestic politics, elections, leadership changes → NOISE unless direct cross-border consequence is explicit in the title

WHEN IN DOUBT → NOISE.
DEDUPLICATION: Same story across multiple titles → keep lowest index only.

Output: {{"signal": [0-based indices]}}. Valid JSON, no markdown.

Titles:
{titles}"""

# -- CONSTANTS -----------------------------------------------------------------

MEDIA_NS = "http://search.yahoo.com/mrss/"
MEDIA_TAG = "{%s}" % MEDIA_NS

ET.register_namespace("media", MEDIA_NS)

BD_TZ = timezone(timedelta(hours=6))

STATS = {
    "per_feed": {},
    "per_method": {
        "KL": 0,
        "DIRECT": 0
    },
    "total_fetched": 0,
    "total_passed_age": 0,
    "total_new": 0,
    "total_signal_mistral": 0,
    "total_signal": 0,
    "total_signal_deduped": 0,
    "timestamp": None,
}

# -- I/O -----------------------------------------------------------------------

def load_processed_articles():
    if Path(PROCESSED_FILE).exists():
        try:
            with open(PROCESSED_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)

            return {
                "article_ids": data.get("article_ids", []),
                "article_links": data.get("article_links", []),
                "last_updated": data.get("last_updated"),
            }

        except Exception:
            pass

    return {
        "article_ids": [],
        "article_links": [],
        "last_updated": None
    }


def save_processed_articles(data):
    data["article_ids"] = list(
        dict.fromkeys(data.get("article_ids", []))
    )

    data["article_links"] = list(
        dict.fromkeys(data.get("article_links", []))
    )

    data["last_updated"] = datetime.utcnow().isoformat()

    with open(PROCESSED_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_selected_articles(articles):
    existing = []

    if Path(SELECTED_FILE).exists():
        try:
            with open(SELECTED_FILE, "r", encoding="utf-8") as f:
                existing = json.load(f)

        except Exception:
            pass

    existing_links = {a.get("link") for a in existing}

    merged = existing + [
        a for a in articles
        if a.get("link") not in existing_links
    ]

    with open(SELECTED_FILE, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)


def save_stats():
    STATS["timestamp"] = datetime.utcnow().isoformat()

    existing = {}

    if Path(STATS_FILE).exists():
        try:
            with open(STATS_FILE, "r", encoding="utf-8") as f:
                existing = json.load(f)

        except Exception:
            pass

    existing.update(STATS)

    with open(STATS_FILE, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)

# -- UTILITIES -----------------------------------------------------------------

def normalize_link(link, base=None):
    if not link:
        return ""

    link = link.strip()

    if link.startswith("//"):
        link = "https:" + link

    if base and not urlparse(link).netloc:
        link = urljoin(base, link)

    link = re.sub(r"([?&])utm_[^=]+=[^&]+", r"\1", link)
    link = re.sub(r"([?&])fbclid=[^&]+", r"\1", link)
    link = re.sub(r"[?&]$", "", link)

    return link.split("#")[0]


def parse_date(entry):
    for key in (
        "published_parsed",
        "updated_parsed",
        "created_parsed",
        "issued_parsed"
    ):
        st = entry.get(key)

        if st:
            try:
                return (
                    datetime.fromtimestamp(time.mktime(st), tz=timezone.utc),
                    False
                )
            except Exception:
                pass

    for key in (
        "published",
        "updated",
        "created",
        "dc_date",
        "issued"
    ):
        val = entry.get(key)

        if isinstance(val, str) and val.strip():

            try:
                dt = parsedate_to_datetime(val)

                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)

                return (dt.astimezone(timezone.utc), False)

            except Exception:
                pass

            if dateutil_parser:
                try:
                    dt = dateutil_parser.parse(val)

                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)

                    return (dt.astimezone(timezone.utc), False)

                except Exception:
                    pass

    if ALLOW_MISSING_DATES:
        return (datetime.now(timezone.utc), True)

    return None, False


IMG_SRC_RE = re.compile(r'<img[^>]+src=["\']([^"\']+)["\']', re.I)


def find_image_in_html(html, base=None):
    if not html:
        return None

    match = IMG_SRC_RE.search(html)

    if not match:
        return None

    return normalize_link(match.group(1).strip(), base=base)


def get_mime_for_url(url):
    if not url:
        return "image/jpeg"

    path = urlparse(url).path.lower()

    if path.endswith(".png"):
        return "image/png"

    if path.endswith(".gif"):
        return "image/gif"

    if path.endswith(".webp"):
        return "image/webp"

    if path.endswith(".svg"):
        return "image/svg+xml"

    return "image/jpeg"


def extract_image_url(entry, base_link=None):

    mt = entry.get("media_thumbnail")

    if mt:
        if isinstance(mt, list) and mt and mt[0].get("url"):
            return normalize_link(mt[0]["url"], base=base_link)

        if isinstance(mt, dict) and mt.get("url"):
            return normalize_link(mt["url"], base=base_link)

    mc = entry.get("media_content")

    if mc:
        if isinstance(mc, list) and mc and mc[0].get("url"):
            return normalize_link(mc[0]["url"], base=base_link)

        if isinstance(mc, dict) and mc.get("url"):
            return normalize_link(mc["url"], base=base_link)

    enc = entry.get("enclosures")

    if enc and isinstance(enc, list):
        for e in enc:
            href = e.get("href") or e.get("url") or e.get("link")
            typ = e.get("type", "")

            if href and (
                typ.startswith("image/")
                or re.search(r"\.(jpg|jpeg|png|gif|webp|svg)$", href, re.I)
            ):
                return normalize_link(href, base=base_link)

    links = entry.get("links")

    if links and isinstance(links, list):
        for l in links:
            if l.get("rel") == "enclosure":
                href = l.get("href")

                if href:
                    return normalize_link(href, base=base_link)

    content = entry.get("content")

    if content:
        if isinstance(content, list):
            for c in content:
                if isinstance(c, dict) and c.get("value"):
                    found = find_image_in_html(c.get("value"), base=base_link)

                    if found:
                        return found

        elif isinstance(content, str):
            found = find_image_in_html(content, base=base_link)

            if found:
                return found

    for key in (
        "summary",
        "description",
        "summary_detail",
        "description_detail"
    ):
        val = entry.get(key)

        if isinstance(val, dict):
            val = val.get("value")

        if isinstance(val, str) and val:
            found = find_image_in_html(val, base=base_link)

            if found:
                return found

    return None

# -- FETCHING ------------------------------------------------------------------

def fetch_via_kl(kl_endpoint, target_feed_url, timeout=20):
    if not kl_endpoint:
        return None

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/xml, text/xml, */*"
    }

    payload = {"url": target_feed_url}

    try:
        resp = requests.post(
            kl_endpoint,
            json=payload,
            headers=headers,
            timeout=timeout
        )

        if resp.status_code == 200 and resp.text:
            return feedparser.parse(resp.text)

    except Exception:
        pass

    try:
        resp = requests.get(
            kl_endpoint,
            params={"url": target_feed_url},
            headers=headers,
            timeout=timeout
        )

        if resp.status_code == 200 and resp.text:
            return feedparser.parse(resp.text)

    except Exception:
        pass

    return None


def fetch_feed(url):
    url_norm = url.strip()
    method_used = "DIRECT"

    if url_norm in KL_API_FEEDS:
        kl_endpoint = os.environ.get("KL")
        feed = None

        if kl_endpoint:
            feed = fetch_via_kl(kl_endpoint, url_norm)

            if feed:
                method_used = "KL"

        if not feed:
            feed = feedparser.parse(url_norm)

    else:
        feed = feedparser.parse(url_norm)

    entries_count = len(getattr(feed, "entries", []))

    STATS["per_feed"].setdefault(
        url_norm,
        {"fetched": 0, "passed_age": 0, "capped": 0}
    )

    STATS["per_feed"][url_norm]["fetched"] += entries_count

    STATS["per_method"].setdefault(method_used, 0)
    STATS["per_method"][method_used] += entries_count
    STATS["total_fetched"] += entries_count

    return feed


def fetch_all_feeds():
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=MAX_AGE_HOURS)

    bd_now = datetime.now(BD_TZ)
    bd_now_str = bd_now.strftime("%a, %d %b %Y %H:%M:%S +0600")

    all_articles = []

    for url in FEED_URLS:

        feed = fetch_feed(url)
        feed_items = []

        for e in feed.entries:

            dt, inferred = parse_date(e)

            if not dt:
                continue

            if not ALLOW_OLDER and dt < cutoff:
                continue

            desc = ""

            if e.get("summary"):
                desc = e.get("summary")

            elif e.get("description"):
                desc = e.get("description")

            elif e.get("content") and isinstance(e.get("content"), list):
                desc = "\n".join(
                    [
                        c.get("value", "")
                        for c in e.get("content")
                        if isinstance(c, dict)
                    ]
                )

            else:
                det = (
                    e.get("summary_detail")
                    or e.get("description_detail")
                )

                if isinstance(det, dict):
                    desc = det.get("value", "") or ""

            link = normalize_link(e.get("link") or "")
            article_id = e.get("id") or link or ""
            image_url = extract_image_url(e, base_link=link)

            article = {
                "id": str(article_id),
                "title": e.get("title", "") or "",
                "link": link,
                "description": desc or "",
                "published": bd_now_str,
                "source": url,
            }

            if inferred:
                article["published_inferred"] = True

            if image_url:
                article["thumbnail"] = image_url
                article["thumbnail_type"] = get_mime_for_url(image_url)

            feed_items.append(article)

        passed = len(feed_items)
        capped = min(passed, MAX_ARTICLES_PER_FEED)

        STATS["per_feed"][url]["passed_age"] = passed
        STATS["per_feed"][url]["capped"] = capped
        STATS["total_passed_age"] += passed

        all_articles.extend(feed_items[:MAX_ARTICLES_PER_FEED])

    return all_articles


def get_new_articles(all_articles, processed_data):
    processed_ids = set(processed_data.get("article_ids", []))
    processed_links = set(processed_data.get("article_links", []))

    new = []

    for a in all_articles:
        aid = a.get("id")
        alink = a.get("link")

        if (
            aid and aid not in processed_ids
            and alink and alink not in processed_links
        ):
            new.append(a)

        elif (
            alink and alink not in processed_links
            and aid not in processed_ids
        ):
            new.append(a)

    return new

# -- CLASSIFICATION ------------------------------------------------------------

def extract_signal_indices(text):
    text = text.replace("```json", "").replace("```", "").strip()

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)

    if match:
        try:
            obj = json.loads(match.group(0))

            if isinstance(obj, dict):
                indices = obj.get("signal", [])

                if isinstance(indices, list):
                    return [i for i in indices if isinstance(i, int)]

        except Exception:
            pass

    match = re.search(r'"signal"\s*:\s*(\[.*?\])', text, flags=re.DOTALL)

    if match:
        try:
            indices = json.loads(match.group(1))

            if isinstance(indices, list):
                return [i for i in indices if isinstance(i, int)]

        except Exception:
            pass

    return []


def send_to_mistral(articles):
    """
    Batched Mistral calls — BATCH_SIZE titles per request.
    Waits BATCH_INTERVAL_SECONDS after each response before the next call.
    Returns global (0-based) SIGNAL indices across all batches.
    """
    api_key = os.environ.get("MS")

    if not api_key:
        print(
            "Mistral classification error: "
            "MS environment variable is not set."
        )
        return []

    if not articles:
        return []

    client = Mistral(api_key=api_key)

    batches = [
        articles[i:i + BATCH_SIZE]
        for i in range(0, len(articles), BATCH_SIZE)
    ]

    total_batches = len(batches)
    all_signal_indices = []

    print(
        f"  Classifying {len(articles)} articles "
        f"in {total_batches} batch(es) of {BATCH_SIZE}..."
    )

    for batch_num, batch in enumerate(batches):
        offset = batch_num * BATCH_SIZE

        titles_text = "\n".join(
            [f"{i}. {a.get('title', '')}" for i, a in enumerate(batch)]
        )

        formatted_prompt = PROMPT.format(titles=titles_text)

        try:
            response = client.chat.complete(
                model=MISTRAL_MODEL,
                messages=[
                    {"role": "user", "content": formatted_prompt}
                ],
                response_format={"type": "json_object"},
            )

            text = response.choices[0].message.content or ""
            batch_indices = extract_signal_indices(text)

            # Validate within-batch bounds, then convert to global indices
            global_indices = [
                offset + i
                for i in batch_indices
                if isinstance(i, int) and 0 <= i < len(batch)
            ]

            all_signal_indices.extend(global_indices)

            print(
                f"  Batch {batch_num + 1}/{total_batches}: "
                f"{len(batch)} titles → {len(global_indices)} signal"
            )

        except Exception as e:
            print(
                f"  Mistral classification error "
                f"(batch {batch_num + 1}/{total_batches}): {e}"
            )

        # Wait between batches, not after the last one
        if batch_num < total_batches - 1:
            time.sleep(BATCH_INTERVAL_SECONDS)

    return all_signal_indices

# -- XML -----------------------------------------------------------------------

def _fresh_channel(root, feed_title, feed_description):
    channel = ET.SubElement(root, "channel")

    ET.SubElement(channel, "title").text = feed_title
    ET.SubElement(channel, "link").text = (
        "https://yourusername.github.io/yourrepo/"
    )
    ET.SubElement(channel, "description").text = feed_description

    return channel


def _load_or_create(output_file, feed_title, feed_description):
    ET.register_namespace("media", MEDIA_NS)

    if Path(output_file).exists():
        try:
            tree = ET.parse(output_file)
            root = tree.getroot()
            channel = root.find("channel")

            if channel is not None:
                return (tree, root, channel)

            channel = _fresh_channel(root, feed_title, feed_description)
            return (tree, root, channel)

        except ET.ParseError:
            pass

    root = ET.Element("rss", {"version": "2.0"})
    tree = ET.ElementTree(root)
    channel = _fresh_channel(root, feed_title, feed_description)

    return (tree, root, channel)


def generate_xml_feed(
    articles,
    output_file,
    feed_title=None,
    feed_description=None
):
    feed_title = feed_title or "Curated News"
    feed_description = feed_description or "AI-curated news feed"

    (tree, root, channel) = _load_or_create(
        output_file, feed_title, feed_description
    )

    existing_links = set()

    for item in channel.findall("item"):
        link_el = item.find("link")

        if link_el is not None and link_el.text:
            existing_links.add(link_el.text.strip())

    added = 0

    for a in articles:
        link = (a.get("link") or "").strip()

        if not link or link in existing_links:
            continue

        item = ET.SubElement(channel, "item")

        ET.SubElement(item, "title").text = a.get("title", "") or ""
        ET.SubElement(item, "link").text = link

        guid_val = a.get("id") or link
        is_permalink = "true" if guid_val.startswith("http") else "false"

        ET.SubElement(item, "guid", {"isPermaLink": is_permalink}).text = guid_val

        ET.SubElement(item, "description").text = a.get("description", "") or ""

        if a.get("published"):
            ET.SubElement(item, "pubDate").text = a["published"]

        thumb = a.get("thumbnail")

        if thumb:
            ET.SubElement(item, MEDIA_TAG + "thumbnail", {"url": thumb})

            mime = a.get("thumbnail_type") or get_mime_for_url(thumb)

            ET.SubElement(
                item,
                "enclosure",
                {"url": thumb, "type": mime, "length": "0"}
            )

        existing_links.add(link)
        added += 1

    all_items = channel.findall("item")
    overflow = len(all_items) - MAX_FEED_ITEMS

    if overflow > 0:
        for old_item in all_items[:overflow]:
            channel.remove(old_item)

    now_text = datetime.utcnow().strftime("%a, %d %b %Y %H:%M:%S +0000")

    last_build = channel.find("lastBuildDate")

    if last_build is None:
        ET.SubElement(channel, "lastBuildDate").text = now_text
    else:
        last_build.text = now_text

    try:
        ET.indent(tree, space="  ")
    except AttributeError:
        pass

    tree.write(output_file, encoding="unicode", xml_declaration=False)

    with open(output_file, "r+", encoding="utf-8") as fh:
        body = fh.read()
        fh.seek(0)
        fh.write('<?xml version="1.0" encoding="UTF-8"?>\n' + body)
        fh.truncate()

    return added

# -- STATS ---------------------------------------------------------------------

def print_stats():
    print("\nFetch statistics:")
    print(f"  Timestamp:            {STATS.get('timestamp')}")
    print(f"  Total fetched:        {STATS['total_fetched']}")
    print(
        f"  Passed age cut:       {STATS['total_passed_age']}  "
        f"(within {MAX_AGE_HOURS}h)"
    )
    print(f"  New (unseen):         {STATS['total_new']}")
    print(f"  Signal (Mistral):     {STATS['total_signal_mistral']}")
    print(
        f"  Signal (after dedup): {STATS['total_signal_deduped']}  "
        f"-> {OUTPUT_XML}"
    )

    print("  Per-method:")

    for method, cnt in STATS["per_method"].items():
        print(f"    {method}: {cnt}")

    print("  Per-feed:")

    for feed, d in STATS["per_feed"].items():
        print(f"    {feed}")
        print(
            f"      fetched={d.get('fetched', 0)}  "
            f"passed_age={d.get('passed_age', 0)}  "
            f"capped={d.get('capped', 0)}"
        )

    print("")

# -- MAIN ----------------------------------------------------------------------

def main():
    processed_data = load_processed_articles()
    all_articles = fetch_all_feeds()
    new_articles = get_new_articles(all_articles, processed_data)

    STATS["total_new"] = len(new_articles)

    mistral_indices = send_to_mistral(new_articles)

    # Keep only valid indices and remove duplicates
    mistral_indices = list(
        dict.fromkeys(
            i
            for i in mistral_indices
            if isinstance(i, int) and 0 <= i < len(new_articles)
        )
    )

    STATS["total_signal_mistral"] = len(mistral_indices)
    STATS["total_signal"] = len(mistral_indices)

    if not mistral_indices:
        print(
            "Mistral returned no signal indices. "
            "Skipping all file writes."
        )

        STATS["timestamp"] = datetime.utcnow().isoformat()
        save_stats()
        print_stats()
        return

    signal_articles = [new_articles[i] for i in mistral_indices]

    mistral_index_set = set(mistral_indices)

    excluded_articles = [
        new_articles[i]
        for i in range(len(new_articles))
        if i not in mistral_index_set
    ]

    STATS["total_signal_deduped"] = len(signal_articles)

    generate_xml_feed(
        signal_articles,
        output_file=OUTPUT_XML,
        feed_title="Curated News",
        feed_description=(
            "AI-curated signal: "
            "Bangladesh affairs and international hard news"
        ),
    )

    generate_xml_feed(
        excluded_articles,
        output_file=EXCLUDED_XML,
        feed_title="Excluded News",
        feed_description=(
            "Articles excluded after "
            "Mistral classification"
        ),
    )

    save_selected_articles(signal_articles)

    processed_data.setdefault("article_ids", []).extend(
        [a["id"] for a in new_articles if a.get("id")]
    )

    processed_data.setdefault("article_links", []).extend(
        [a["link"] for a in new_articles if a.get("link")]
    )

    save_processed_articles(processed_data)

    STATS["timestamp"] = datetime.utcnow().isoformat()
    save_stats()
    print_stats()


if __name__ == "__main__":
    main()
