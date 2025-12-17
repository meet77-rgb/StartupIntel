import os
import json
import time
import csv
import urllib.parse
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from typing import Any, List, Dict, TypedDict

import requests
import tldextract
import trafilatura
from ddgs import DDGS
from dotenv import load_dotenv
from tqdm import tqdm

from langgraph.graph import StateGraph, END
from openai import OpenAI

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY missing in .env")

# =========================================================
# CONFIG
# =========================================================
SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
DAYS_BACK = 30
MAX_ARTICLES = 30

GDELT_QUERY = (
    'startup (raised OR raises OR secured OR funding OR "seed round" '
    'OR "Series A" OR "Series B" OR "pre-seed")'
)

CAREER_PATHS = ["/careers", "/jobs", "/join-us", "/hiring", "/open-roles"]

TECH_KEYWORDS = [
    "software engineer", "backend", "frontend", "full stack",
    "ml engineer", "ai engineer", "data scientist", "devops"
]

BLACKLIST_DOMAINS = (
    "linkedin.com", "crunchbase.com", "pitchbook.com",
    "facebook.com", "twitter.com", "wikipedia.org"
)


SEEN_COMPANIES = set()
DOMAIN_CACHE = {}

# =========================================================
# DATA MODELS
# =========================================================
@dataclass
class StartupSignalRow:
    company: str
    domain: str
    amount_raised: str
    round_type: str
    investors: str
    lead_investor: str
    country: str
    hiring_tech: str
    careers_url: str
    funding_date: str
    source_url: str


class GraphState(TypedDict):
    articles: List[Dict[str, Any]]
    rows: List[StartupSignalRow]

# =========================================================
# UTILS
# =========================================================
def safe_get(url: str, timeout: int = 20):
    try:
        return requests.get(
            url,
            timeout=timeout,
            headers={"User-Agent": "Mozilla/5.0"},
            allow_redirects=True,
        )
    except Exception:
        return None


def normalize_domain(url: str) -> str:
    ext = tldextract.extract(url)
    if not ext.domain or not ext.suffix:
        return ""
    return f"https://{ext.domain}.{ext.suffix}"


def normalize_llm_field(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return ", ".join(str(v) for v in value if v)
    if isinstance(value, dict):
        return json.dumps(value)
    return str(value).strip()


def normalize_llm_records(raw: Any) -> List[Dict[str, Any]]:
    if isinstance(raw, dict):
        return [raw]
    if isinstance(raw, list):
        return [r for r in raw if isinstance(r, dict)]
    return []

# =========================================================
# SEARCH — GDELT
# =========================================================
def gdelt_search():
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=DAYS_BACK)

    def fmt(dt): return dt.strftime("%Y%m%d%H%M%S")

    params = {
        "query": GDELT_QUERY,
        "mode": "ArtList",
        "format": "json",
        "maxrecords": MAX_ARTICLES,
        "startdatetime": fmt(start),
        "enddatetime": fmt(end),
    }

    url = "https://api.gdeltproject.org/api/v2/doc/doc?" + urllib.parse.urlencode(params)
    r = safe_get(url)

    if not r or not r.text or not r.text.lstrip().startswith("{"):
        print("⚠️ GDELT returned non-JSON response")
        return []

    return r.json().get("articles", [])

# =========================================================
# EXTRACTION
# =========================================================
def extract_article_text(url: str) -> str:
    r = safe_get(url)
    if not r or not r.text:
        return ""
    return trafilatura.extract(r.text) or ""


def openai_extract(text: str, source_url: str) -> List[Dict[str, str]]:
    client = OpenAI(api_key=OPENAI_API_KEY)

    prompt = f"""
Extract startup funding information.

Return JSON ONLY.
If multiple startups exist, return a LIST.

Each object:
company, amount_raised, round_type,
investors, lead_investor, country, funding_date

ARTICLE:
{text[:12000]}
"""

    resp = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt,
        temperature=0.0,
    )

    raw = resp.output_text.strip().strip("```").replace("json", "")
    parsed = json.loads(raw)

    records = normalize_llm_records(parsed)
    out = []

    for r in records:
        out.append({
            "company": normalize_llm_field(r.get("company")),
            "amount_raised": normalize_llm_field(r.get("amount_raised")),
            "round_type": normalize_llm_field(r.get("round_type")),
            "investors": normalize_llm_field(r.get("investors")),
            "lead_investor": normalize_llm_field(r.get("lead_investor")),
            "country": normalize_llm_field(r.get("country")),
            "funding_date": normalize_llm_field(r.get("funding_date")),
        })

    return out

# =========================================================
# DOMAIN + CAREERS
# =========================================================
def find_domain(company: str) -> str:
    if company in DOMAIN_CACHE:
        return DOMAIN_CACHE[company]

    with DDGS() as ddgs:
        for r in ddgs.text(f"{company} official website", max_results=5):
            url = r.get("href", "")
            if url and not any(b in url for b in BLACKLIST_DOMAINS):
                domain = normalize_domain(url)
                DOMAIN_CACHE[company] = domain
                return domain

    DOMAIN_CACHE[company] = ""
    return ""


def find_careers(domain: str) -> str:
    for p in CAREER_PATHS:
        url = domain.rstrip("/") + p
        if safe_get(url):
            return url
    return ""


def is_hiring_tech(careers_url: str) -> str:
    r = safe_get(careers_url)
    if not r or not r.text:
        return "No"
    t = r.text.lower()
    return "Yes" if any(k in t for k in TECH_KEYWORDS) else "No"

# =========================================================
# OUTPUTS
# =========================================================
def save_to_csv(rows: List[StartupSignalRow], filename="startup_signals.csv"):
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=asdict(rows[0]).keys())
        writer.writeheader()
        for r in rows:
            writer.writerow(asdict(r))
    print(f"📁 Local CSV saved: {filename}")


def write_to_google_sheet(rows: List[StartupSignalRow]):
    creds = None
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)

    if not creds or not creds.valid:
        flow = InstalledAppFlow.from_client_secrets_file(
            "client_secret.json", SCOPES
        )
        creds = flow.run_local_server(port=0)
        with open("token.json", "w") as f:
            f.write(creds.to_json())

    service = build("sheets", "v4", credentials=creds)

    sheet = service.spreadsheets().create(
        body={"properties": {"title": "Startup Signals (Deduplicated)"}}
    ).execute()

    sheet_id = sheet["spreadsheetId"]

    headers = list(asdict(rows[0]).keys())
    values = [headers] + [list(asdict(r).values()) for r in rows]

    service.spreadsheets().values().update(
        spreadsheetId=sheet_id,
        range="Sheet1!A1",
        valueInputOption="RAW",
        body={"values": values},
    ).execute()

    print(f"📊 Google Sheet:")
    print(f"https://docs.google.com/spreadsheets/d/{sheet_id}")

# =========================================================
# LANGGRAPH NODES
# =========================================================
def node_search(state: GraphState):
    return {"articles": gdelt_search(), "rows": []}


def node_process(state: GraphState):
    rows = []
    print("\n📰 Processing articles...\n")

    for a in tqdm(state["articles"], desc="Articles processed", unit="article"):
        url = a.get("url")
        if not url:
            continue

        text = extract_article_text(url)
        if len(text) < 500:
            continue

        try:
            records = openai_extract(text, url)
        except Exception as e:
            print(f"⚠️ OpenAI failed for {url}: {e}")
            continue

        for data in records:
            company = data.get("company")
            if not company:
                continue

            key = company.lower().strip()
            if key in SEEN_COMPANIES:
                continue
            SEEN_COMPANIES.add(key)

            domain = find_domain(company)
            careers = find_careers(domain)
            hiring = is_hiring_tech(careers)

            rows.append(StartupSignalRow(
                company=company,
                domain=domain,
                amount_raised=data.get("amount_raised", ""),
                round_type=data.get("round_type", ""),
                investors=data.get("investors", ""),
                lead_investor=data.get("lead_investor", ""),
                country=data.get("country", ""),
                hiring_tech=hiring,
                careers_url=careers,
                funding_date=data.get("funding_date", ""),
                source_url=url,
            ))

            print(f"✅ Added startup: {company}")

        time.sleep(0.2)

    print(f"\n✅ Finished processing {len(rows)} unique startups.")
    return {"articles": state["articles"], "rows": rows}


def node_output(state: GraphState):
    rows = state["rows"]
    if not rows:
        print("⚠️ No startups found.")
        return state

    save_to_csv(rows)
    write_to_google_sheet(rows)
    return state

# =========================================================
# MAIN
# =========================================================
def main():
    graph = StateGraph(GraphState)
    graph.add_node("search", node_search)
    graph.add_node("process", node_process)
    graph.add_node("output", node_output)

    graph.set_entry_point("search")
    graph.add_edge("search", "process")
    graph.add_edge("process", "output")
    graph.add_edge("output", END)

    app = graph.compile()
    app.invoke({"articles": [], "rows": []})


if __name__ == "__main__":
    main()
