#!/usr/bin/env python3
"""
RSS Feed Processor with Gemini API Integration (robust date/content handling + thumbnails)

All articles from all feeds go to one Gemini call.
Gemini classifies each headline into signal, longread, or noise.
A second Gemini call deduplicates near-identical titles within each bucket.

Outputs:
  curated_feed.xml  - signal articles
  longread.xml      - longread articles
Stats:
  fetch_stats.json
"""

import feedparser
import json
import os
import time
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path
import xml.etree.ElementTree as ET
from google import genai
from email.utils import parsedate_to_datetime
from urllib.parse import urljoin, urlparse

import requests

try:
    from dateutil import parser as dateutil_parser
except Exception:
    dateutil_parser = None

# -- FEEDS ---------------------------------------------------------------------

FEED_URLS = [
    "https://www.newyorker.com/feed/rss",
    "https://feeds.feedburner.com/TheAtlantic",
    "https://evilgodfahim.github.io/tm/feeds/feed.xml",
    "https://evilgodfahim.github.io/ftint/combined.xml",
    "https://evilgodfahim.github.io/gd/merged.xml",
    "https://evilgodfahim.github.io/nytint/combined.xml",
    "https://evilgodfahim.github.io/wpint/combined.xml",
    "https://evilgodfahim.github.io/wsjint/combined.xml",
    "https://evilgodfahim.github.io/wl/pau.xml",
    "https://evilgodfahim.github.io/yn/feeds/feed.xml",
    "https://en.prothomalo.com/feed/",
    "https://evilgodfahim.github.io/fen/feeds/feed.xml",
    "https://evilgodfahim.github.io/tbs/articles.xml",
    "https://evilgodfahim.github.io/dstar/feeds/feed.xml",
]

EXISTING_API_FEEDS = {
    "https://www.newyorker.com/feed/rss",
    "https://feeds.feedburner.com/TheAtlantic",
    "https://evilgodfahim.github.io/tm/feeds/feed.xml",
    "https://evilgodfahim.github.io/ftint/combined.xml",
    "https://evilgodfahim.github.io/gd/merged.xml",
    "https://evilgodfahim.github.io/nytint/combined.xml",
    "https://evilgodfahim.github.io/wpint/combined.xml",
    "https://evilgodfahim.github.io/wsjint/combined.xml",
    "https://evilgodfahim.github.io/wl/pau.xml",
    "https://evilgodfahim.github.io/yn/feeds/feed.xml",
    "https://en.prothomalo.com/feed/",
    "https://evilgodfahim.github.io/fen/feeds/feed.xml",
    "https://evilgodfahim.github.io/tbs/articles.xml",
    "https://evilgodfahim.github.io/dstar/feeds/feed.xml",
}

KL_API_FEEDS = set()

# -- CONFIG --------------------------------------------------------------------

GEMINI_MODEL          = "gemini-3-flash-preview"
DEDUP_MODEL           = "gemini-2.5-flash"
PROCESSED_FILE        = "processed_articles.json"
SELECTED_FILE         = "selected_articles.json"
OUTPUT_XML            = "curated_feed.xml"
LONGREAD_XML          = "longread.xml"
STATS_FILE            = "fetch_stats.json"
MAX_ARTICLES_PER_FEED = 50
MAX_AGE_HOURS         = 4
ALLOW_MISSING_DATES   = True
ALLOW_OLDER           = False
MAX_FEED_ITEMS        = 500          # rolling cap per output file

# -- PROMPT --------------------------------------------------------------------

PROMPT = """You are a news classification engine. Classify each headline into exactly one bucket.
SIGNAL — news that matters globally or within Bangladesh: major international events, geopolitical developments involving multiple countries, or Bangladesh developments that meaningfully affect a large portion of the population (major policy shifts, economic crises, political upheaval, governance changes). Isolated incidents, local events, or routine Bangladesh news do not qualify.
LONGREAD — worth reading but not urgent: high-quality in-depth reporting, investigations, features, or thoughtful essays on culture, science, history, or society that reward careful reading. Excludes celebrity profiles, trend pieces, and routine human-interest stories.
NOISE — everything else: any non-Bangladesh country's internal politics, elections, policy disputes, business news, or market moves — plus isolated Bangladesh incidents, sports, entertainment, celebrity gossip, lifestyle, routine official statements, and clickbait.
Rules:
- If a headline could fit both SIGNAL and LONGREAD, always choose SIGNAL.
- Use only the headline text. Indices are 0-based.
- Omit all noise indices from the output entirely.
- Return only valid JSON. No markdown, no backticks, no preamble.
Tricky cases to guide you:
- Bangladesh policy or economic decision with broad national impact → SIGNAL.
- An isolated Bangladesh incident or local event → NOISE, not SIGNAL.
- A routine Bangladesh government statement with no new development → NOISE.
- Any other country's domestic politics or policy with no cross-border impact → NOISE.
- A geopolitical event involving multiple countries or international bodies → SIGNAL.
- National business or market news from any non-Bangladesh country → NOISE unless it signals a global crisis.
- A think-piece on an international subject with genuine global scope → SIGNAL, not LONGREAD.
- A detailed profile or feature on a person with no global or broad Bangladesh consequence → LONGREAD, not SIGNAL.

Examples:
Input: ["US and China sign landmark trade agreement", "Premier League club sacks manager", "How the Ottoman Empire collapsed", "Bangladesh central bank raises interest rates amid inflation crisis", "UK Conservative Party elects new leader", "UN warns of imminent famine across the Horn of Africa"]
Output: {{"signal": [0, 3, 5], "longread": [2]}}

Input: ["India and Pakistan exchange fire across Line of Control", "Dhaka garment workers strike shuts down hundreds of factories", "The secret history of Antarctic exploration", "Australia holds federal election", "Celebrity couple announces divorce", "IMF approves emergency loan for Bangladesh"]
Output: {{"signal": [0, 1, 5], "longread": [2]}}

Input: ["Gaza ceasefire collapses as fighting resumes", "Bangladesh government slashes fuel subsidies nationwide", "A deep dive into the life of a Sundarbans honey collector", "France passes new immigration law", "How microplastics are entering the human bloodstream", "Local man wins national baking competition"]
Output: {{"signal": [0, 1], "longread": [2, 4]}}

Article titles:
{titles}
"""

DEDUP_PROMPT = """You are a news deduplication engine. You will receive a numbered list of article titles.
Your task: identify groups of titles that cover the same story or event (near-duplicates, rephrased versions, or very similar headlines). For each such group, keep only the FIRST occurrence (lowest index) and discard the rest.
Titles that cover clearly distinct topics must all be kept.

Rules:
- Return only the indices (0-based) of titles to KEEP, as a JSON array of integers.
- Always keep at least one title from each duplicate group (the one with the lowest index).
- If all titles are unique, return all indices.
- Return only valid JSON. No markdown, no backticks, no preamble. Example output: [0, 1, 3, 5]

Article titles:
{titles}
"""

# -- CONSTANTS -----------------------------------------------------------------

MEDIA_NS    = "http://search.yahoo.com/mrss/"
MEDIA_TAG   = "{%s}" % MEDIA_NS          # shorthand: "{http://...}"
ET.register_namespace("media", MEDIA_NS)

BD_TZ = timezone(timedelta(hours=6))

STATS = {
    "per_feed":              {},
    "per_method":            {"KL": 0, "DIRECT": 0},
    "total_fetched":         0,
    "total_passed_age":      0,
    "total_new":             0,
    "total_signal":          0,
    "total_longread":        0,
    "total_signal_deduped":  0,
    "total_longread_deduped":0,
    "timestamp":             None,
}

# -- I/O -----------------------------------------------------------------------

def load_processed_articles():
    if Path(PROCESSED_FILE).exists():
        try:
            with open(PROCESSED_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            return {
                "article_ids":   data.get("article_ids", []),
                "article_links": data.get("article_links", []),
                "last_updated":  data.get("last_updated"),
            }
        except Exception:
            pass
    return {"article_ids": [], "article_links": [], "last_updated": None}


def save_processed_articles(data):
    data["article_ids"]   = list(dict.fromkeys(data.get("article_ids", [])))
    data["article_links"] = list(dict.fromkeys(data.get("article_links", [])))
    data["last_updated"]  = datetime.utcnow().isoformat()
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
    merged = existing + [a for a in articles if a.get("link") not in existing_links]
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
    link = re.sub(r"([?&])fbclid=[^&]+",    r"\1", link)
    link = re.sub(r"[?&]$", "", link)
    return link.split("#")[0]


def parse_date(entry):
    for key in ("published_parsed", "updated_parsed", "created_parsed", "issued_parsed"):
        st = entry.get(key)
        if st:
            try:
                dt = datetime.fromtimestamp(time.mktime(st), tz=timezone.utc)
                return dt, False
            except Exception:
                pass
    for key in ("published", "updated", "created", "dc_date", "issued"):
        val = entry.get(key)
        if isinstance(val, str) and val.strip():
            try:
                dt = parsedate_to_datetime(val)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.astimezone(timezone.utc), False
            except Exception:
                pass
            if dateutil_parser:
                try:
                    dt = dateutil_parser.parse(val)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    return dt.astimezone(timezone.utc), False
                except Exception:
                    pass
    if ALLOW_MISSING_DATES:
        return datetime.now(timezone.utc), True
    return None, False


IMG_SRC_RE = re.compile(r'<img[^>]+src=["\']([^"\']+)["\']', re.I)


def find_image_in_html(html, base=None):
    if not html:
        return None
    m = IMG_SRC_RE.search(html)
    if not m:
        return None
    return normalize_link(m.group(1).strip(), base=base)


def get_mime_for_url(url):
    if not url:
        return "image/jpeg"
    path = urlparse(url).path.lower()
    if path.endswith(".png"):  return "image/png"
    if path.endswith(".gif"):  return "image/gif"
    if path.endswith(".webp"): return "image/webp"
    if path.endswith(".svg"):  return "image/svg+xml"
    return "image/jpeg"


def extract_image_url(entry, base_link=None):
    mt = entry.get("media_thumbnail")
    if mt:
        if isinstance(mt, list) and mt[0].get("url"):
            return normalize_link(mt[0]["url"], base=base_link)
        if isinstance(mt, dict) and mt.get("url"):
            return normalize_link(mt["url"], base=base_link)

    mc = entry.get("media_content")
    if mc:
        if isinstance(mc, list) and mc[0].get("url"):
            return normalize_link(mc[0]["url"], base=base_link)
        if isinstance(mc, dict) and mc.get("url"):
            return normalize_link(mc["url"], base=base_link)

    enc = entry.get("enclosures")
    if enc and isinstance(enc, list):
        for e in enc:
            href = e.get("href") or e.get("url") or e.get("link")
            typ  = e.get("type", "")
            if href and (typ.startswith("image/") or re.search(r'\.(jpg|jpeg|png|gif|webp|svg)$', href, re.I)):
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

    for key in ("summary", "description", "summary_detail", "description_detail"):
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
    headers = {"Content-Type": "application/json", "Accept": "application/xml, text/xml, */*"}
    payload = {"url": target_feed_url}
    try:
        resp = requests.post(kl_endpoint, json=payload, headers=headers, timeout=timeout)
        if resp.status_code == 200 and resp.text:
            return feedparser.parse(resp.text)
    except Exception:
        pass
    try:
        resp = requests.get(kl_endpoint, params={"url": target_feed_url}, headers=headers, timeout=timeout)
        if resp.status_code == 200 and resp.text:
            return feedparser.parse(resp.text)
    except Exception:
        pass
    return None


def fetch_feed(url):
    url_norm    = url.strip()
    method_used = "DIRECT"

    if url_norm in EXISTING_API_FEEDS:
        feed        = feedparser.parse(url_norm)
        method_used = "DIRECT"
    elif url_norm in KL_API_FEEDS:
        kl_endpoint = os.environ.get("KL")
        feed        = None
        if kl_endpoint:
            feed = fetch_via_kl(kl_endpoint, url_norm)
            if feed:
                method_used = "KL"
        if not feed:
            feed        = feedparser.parse(url_norm)
            method_used = "DIRECT"
    else:
        feed        = feedparser.parse(url_norm)
        method_used = "DIRECT"

    entries_count = len(getattr(feed, "entries", []))
    STATS["per_feed"].setdefault(url_norm, {"fetched": 0, "passed_age": 0, "capped": 0})
    STATS["per_feed"][url_norm]["fetched"] += entries_count
    STATS["per_method"].setdefault(method_used, 0)
    STATS["per_method"][method_used] += entries_count
    STATS["total_fetched"]            += entries_count

    return feed


def fetch_all_feeds():
    now        = datetime.now(timezone.utc)
    cutoff     = now - timedelta(hours=MAX_AGE_HOURS)
    bd_now     = datetime.now(BD_TZ)
    bd_now_str = bd_now.strftime("%a, %d %b %Y %H:%M:%S +0600")
    all_articles = []

    for url in FEED_URLS:
        feed       = fetch_feed(url)
        feed_items = []

        for e in feed.entries:
            dt, inferred = parse_date(e)
            if not dt:
                continue
            if (not ALLOW_OLDER) and dt < cutoff:
                continue

            desc = ""
            if e.get("summary"):
                desc = e.get("summary")
            elif e.get("description"):
                desc = e.get("description")
            elif e.get("content") and isinstance(e.get("content"), list):
                desc = "\n".join([c.get("value", "") for c in e.get("content") if isinstance(c, dict)])
            else:
                det = e.get("summary_detail") or e.get("description_detail")
                if isinstance(det, dict):
                    desc = det.get("value", "") or ""

            link       = normalize_link(e.get("link") or "")
            article_id = e.get("id") or link or ""
            image_url  = extract_image_url(e, base_link=link)

            article = {
                "id":          str(article_id),
                "title":       e.get("title", "") or "",
                "link":        link,
                "description": desc or "",
                "published":   bd_now_str,
                "source":      url,
            }
            if inferred:
                article["published_inferred"] = True
            if image_url:
                article["thumbnail"]      = image_url
                article["thumbnail_type"] = get_mime_for_url(image_url)

            feed_items.append(article)

        passed = len(feed_items)
        capped = min(passed, MAX_ARTICLES_PER_FEED)
        STATS["per_feed"][url]["passed_age"] = passed
        STATS["per_feed"][url]["capped"]     = capped
        STATS["total_passed_age"]           += passed
        all_articles.extend(feed_items[:MAX_ARTICLES_PER_FEED])

    return all_articles


def get_new_articles(all_articles, processed_data):
    processed_ids   = set(processed_data.get("article_ids", []))
    processed_links = set(processed_data.get("article_links", []))
    new = []
    for a in all_articles:
        aid   = a.get("id")
        alink = a.get("link")
        if (aid and aid not in processed_ids) and (alink and alink not in processed_links):
            new.append(a)
        elif alink and alink not in processed_links and aid not in processed_ids:
            new.append(a)
    return new

# -- GEMINI --------------------------------------------------------------------

def extract_json_object(text):
    """Parse {"signal": [...], "longread": [...]} from Gemini response."""
    text = text.replace("```json", "").replace("```", "").strip()
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            if isinstance(obj, dict):
                return {
                    "signal":   [i for i in obj.get("signal",   []) if isinstance(i, int)],
                    "longread": [i for i in obj.get("longread", []) if isinstance(i, int)],
                }
        except Exception:
            pass
    result = {"signal": [], "longread": []}
    for key in ("signal", "longread"):
        m = re.search(rf'"{key}"\s*:\s*(\[.*?\])', text, flags=re.DOTALL)
        if m:
            try:
                result[key] = [i for i in json.loads(m.group(1)) if isinstance(i, int)]
            except Exception:
                pass
    return result


def send_to_gemini(articles):
    """Single Gemini 3 Flash call. Returns {"signal": [...], "longread": [...]}."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key or not articles:
        return {"signal": [], "longread": []}

    try:
        client = genai.Client(api_key=api_key)

        titles_text = "\n".join([f"{i}. {a.get('title', '')}" for i, a in enumerate(articles)])

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=f"Article titles:\n{titles_text}",
            config={
                "system_instruction": PROMPT,
                "response_mime_type": "application/json",
            },
        )

        if hasattr(response, "parsed") and response.parsed:
            return {
                "signal":   [i for i in response.parsed.get("signal",   []) if isinstance(i, int)],
                "longread": [i for i in response.parsed.get("longread", []) if isinstance(i, int)],
            }

        return extract_json_object(response.text)

    except Exception as e:
        print(f"Gemini classification error: {e}")
        return {"signal": [], "longread": []}


def deduplicate_articles(articles):
    """
    Send article titles to Gemini 2.5 Flash.
    Returns a deduplicated subset of `articles`, preserving order.
    Near-identical or same-story titles are collapsed to the first occurrence.
    Falls back to returning all articles unchanged on any error.
    """
    if not articles:
        return articles

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return articles

    try:
        client = genai.Client(api_key=api_key)

        titles_text = "\n".join(
            [f"{i}. {a.get('title', '')}" for i, a in enumerate(articles)]
        )

        response = client.models.generate_content(
            model=DEDUP_MODEL,
            contents=DEDUP_PROMPT.format(titles=titles_text),
            config={"response_mime_type": "application/json"},
        )

        # Parse response — expect a plain JSON array of ints, e.g. [0, 1, 3, 5]
        raw = response.text if hasattr(response, "text") else ""
        raw = raw.replace("```json", "").replace("```", "").strip()

        # Try direct array parse first
        keep_indices = None
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                keep_indices = [i for i in parsed if isinstance(i, int) and 0 <= i < len(articles)]
        except Exception:
            pass

        # Fallback: extract array from inside any surrounding object
        if keep_indices is None:
            m = re.search(r"\[[\d,\s]+\]", raw)
            if m:
                try:
                    keep_indices = [
                        i for i in json.loads(m.group(0))
                        if isinstance(i, int) and 0 <= i < len(articles)
                    ]
                except Exception:
                    pass

        if keep_indices is None:
            print("Dedup: could not parse response, keeping all articles.")
            return articles

        # Preserve original ordering; indices returned by model are already ordered,
        # but sort just in case.
        keep_indices = sorted(set(keep_indices))
        deduped = [articles[i] for i in keep_indices]
        dropped = len(articles) - len(deduped)
        if dropped:
            print(f"Dedup: removed {dropped} near-duplicate title(s).")
        return deduped

    except Exception as e:
        print(f"Gemini dedup error: {e}")
        return articles

# -- XML -----------------------------------------------------------------------

def _fresh_channel(root, feed_title, feed_description):
    """Add a blank <channel> to root and return it."""
    channel = ET.SubElement(root, "channel")
    ET.SubElement(channel, "title").text       = feed_title
    ET.SubElement(channel, "link").text        = "https://yourusername.github.io/yourrepo/"
    ET.SubElement(channel, "description").text = feed_description
    return channel


def _load_or_create(output_file, feed_title, feed_description):
    """
    Return (tree, root, channel).

    Tries to parse an existing file.  If the file is absent, empty, or
    corrupt a fresh tree is built from scratch.  The namespace prefix
    'media' is always re-registered so ElementTree writes it correctly.
    """
    ET.register_namespace("media", MEDIA_NS)

    if Path(output_file).exists():
        try:
            tree    = ET.parse(output_file)
            root    = tree.getroot()
            channel = root.find("channel")
            if channel is not None:
                return tree, root, channel
            channel = _fresh_channel(root, feed_title, feed_description)
            return tree, root, channel
        except ET.ParseError:
            pass

    root    = ET.Element("rss", {"version": "2.0"})
    tree    = ET.ElementTree(root)
    channel = _fresh_channel(root, feed_title, feed_description)
    return tree, root, channel


def generate_xml_feed(articles, output_file, feed_title=None, feed_description=None):
    """
    Append new unique articles to the existing RSS <channel>.
    Enforces a MAX_FEED_ITEMS rolling cap — oldest items (top of list) are
    dropped first once the cap is exceeded.
    Creates the file from scratch if it does not exist.
    """
    feed_title       = feed_title       or "Curated News"
    feed_description = feed_description or "AI-curated news feed"

    tree, root, channel = _load_or_create(output_file, feed_title, feed_description)

    existing_links: set[str] = set()
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
        ET.SubElement(item, "title").text       = a.get("title", "") or ""
        ET.SubElement(item, "link").text        = link
        guid_val     = a.get("id") or link
        is_permalink = "true" if guid_val.startswith("http") else "false"
        ET.SubElement(item, "guid", {"isPermaLink": is_permalink}).text = guid_val
        ET.SubElement(item, "description").text = a.get("description", "") or ""
        if a.get("published"):
            ET.SubElement(item, "pubDate").text = a["published"]

        thumb = a.get("thumbnail")
        if thumb:
            ET.SubElement(
                item,
                MEDIA_TAG + "thumbnail",
                {"url": thumb},
            )
            mime = a.get("thumbnail_type") or get_mime_for_url(thumb)
            ET.SubElement(item, "enclosure", {"url": thumb, "type": mime, "length": "0"})

        existing_links.add(link)
        added += 1

    all_items = channel.findall("item")
    overflow  = len(all_items) - MAX_FEED_ITEMS
    if overflow > 0:
        for old_item in all_items[:overflow]:
            channel.remove(old_item)

    now_text   = datetime.utcnow().strftime("%a, %d %b %Y %H:%M:%S +0000")
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
    print(f"  Timestamp:           {STATS.get('timestamp')}")
    print(f"  Total fetched:       {STATS['total_fetched']}  (raw entries from all feeds)")
    print(f"  Passed age cut:      {STATS['total_passed_age']}  (within {MAX_AGE_HOURS}h window)")
    print(f"  New (unseen):        {STATS['total_new']}")
    print(f"  Signal (classified): {STATS['total_signal']}")
    print(f"  Signal (after dedup):{STATS['total_signal_deduped']}  -> {OUTPUT_XML}")
    print(f"  Longread (classified):{STATS['total_longread']}")
    print(f"  Longread (after dedup):{STATS['total_longread_deduped']}  -> {LONGREAD_XML}")
    print("  Per-method (raw fetch):")
    for method, cnt in STATS["per_method"].items():
        print(f"    {method}: {cnt}")
    print("  Per-feed breakdown:")
    for feed, d in STATS["per_feed"].items():
        print(f"    {feed}")
        print(f"      fetched={d.get('fetched',0)}  passed_age={d.get('passed_age',0)}  sent_to_pipeline={d.get('capped',0)}")
    print("")

# -- MAIN ----------------------------------------------------------------------

def main():
    processed_data = load_processed_articles()
    all_articles   = fetch_all_feeds()
    new_articles   = get_new_articles(all_articles, processed_data)

    STATS["total_new"] = len(new_articles)

    # --- Step 1: classify with Gemini 3 Flash --------------------------------
    result = send_to_gemini(new_articles)

    signal_indices   = [i for i in result.get("signal",   []) if isinstance(i, int) and 0 <= i < len(new_articles)]
    longread_indices = [i for i in result.get("longread", []) if isinstance(i, int) and 0 <= i < len(new_articles)]

    # Signal wins on overlap
    signal_set       = set(signal_indices)
    longread_indices = [i for i in longread_indices if i not in signal_set]

    signal_articles   = [new_articles[i] for i in signal_indices]
    longread_articles = [new_articles[i] for i in longread_indices]

    STATS["total_signal"]   = len(signal_articles)
    STATS["total_longread"] = len(longread_articles)

    # --- Step 2: deduplicate signal only with Gemini 2.5 Flash ---------------
    print(f"Deduplicating {len(signal_articles)} signal article(s)...")
    signal_articles = deduplicate_articles(signal_articles)

    STATS["total_signal_deduped"]   = len(signal_articles)
    STATS["total_longread_deduped"] = len(longread_articles)  # no dedup, same as classified

    # --- Step 3: write XML feeds ---------------------------------------------
    generate_xml_feed(
        signal_articles,
        output_file=OUTPUT_XML,
        feed_title="Curated News",
        feed_description="AI-curated signal: international affairs and Bangladesh news",
    )
    generate_xml_feed(
        longread_articles,
        output_file=LONGREAD_XML,
        feed_title="Longread",
        feed_description="Quality in-depth reading: features, analysis, investigations",
    )

    save_selected_articles(signal_articles + longread_articles)

    processed_data.setdefault("article_ids",   []).extend([a["id"]   for a in new_articles if a.get("id")])
    processed_data.setdefault("article_links", []).extend([a["link"] for a in new_articles if a.get("link")])
    save_processed_articles(processed_data)

    STATS["timestamp"] = datetime.utcnow().isoformat()
    save_stats()
    print_stats()


if __name__ == "__main__":
    main()
