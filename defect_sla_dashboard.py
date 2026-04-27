"""
Client Defect SLA Tracker — standalone Streamlit app.

Runs independently via:  streamlit run defect_sla_dashboard.py

Scope: MDLRN client-reported defects (Jira filter 20391). Tracks SLA breach
state, aging, ownership, client attribution, and 30-day flow. Internal audience
(Engineering Manager + team). Not client-facing.

Intentionally self-contained — duplicates a small slice of Jira helper code
from dashboard.py rather than sharing, so the two dashboards can evolve
independently.
"""

from __future__ import annotations

import json
import re
import unicodedata
from datetime import date, datetime, timedelta
from typing import Any

import altair as alt  # noqa: F401  (kept in requirements; may be used for future charts)
import pandas as pd
import pytz
import requests
import streamlit as st
import streamlit.components.v1 as components
from requests.auth import HTTPBasicAuth
from streamlit_autorefresh import st_autorefresh


# =====================
# Constants — SLA config, palette, client slug map
# =====================

IST = pytz.timezone("Asia/Kolkata")

# Priority → (clock kind, budget). Clock kind is "calendar_hours" or "working_days".
SLA_CONFIG: dict[str, tuple[str, float]] = {
    "P0": ("calendar_hours", 8),
    "P1": ("working_days", 3),
    "P2": ("working_days", 7),
    "P3": ("working_days", 15),
}

PRIORITY_ORDER = ["P0", "P1", "P2", "P3"]

# Canonical SLA palette (from spec — do not drift)
COLOR = {
    "red_fill":   "#A32D2D",
    "red_bg":     "#FCEBEB",
    "red_text":   "#501313",
    "amber_fill": "#EF9F27",
    "amber_bg":   "#FAEEDA",
    "amber_text": "#412402",
    "green_fill": "#97C459",
    "green_bg":   "#EAF3DE",
    "green_text": "#173404",
    "p0_bg": "#FCEBEB", "p0_text": "#501313",
    "p1_bg": "#FAECE7", "p1_text": "#4A1B0C",
    "p2_bg": "#FAEEDA", "p2_text": "#412402",
    "p3_bg": "#F1EFE8", "p3_text": "#2C2C2A",
    "text_primary":   "#0f172a",
    "text_secondary": "#64748b",
    "text_tertiary":  "#94a3b8",
    "page_bg":   "#f8fafc",
    "card_bg":   "#ffffff",
    "border":    "#e2e8f0",
    "accent":    "#6366f1",
}

LABEL_STYLE = {
    "breached": (COLOR["red_fill"],   COLOR["red_bg"],   COLOR["red_text"]),
    "at_risk":  (COLOR["amber_fill"], COLOR["amber_bg"], COLOR["amber_text"]),
    "on_track": (COLOR["green_fill"], COLOR["green_bg"], COLOR["green_text"]),
    "met":      (COLOR["green_fill"], COLOR["green_bg"], COLOR["green_text"]),
}

PRIORITY_STYLE = {
    "P0": (COLOR["p0_bg"], COLOR["p0_text"]),
    "P1": (COLOR["p1_bg"], COLOR["p1_text"]),
    "P2": (COLOR["p2_bg"], COLOR["p2_text"]),
    "P3": (COLOR["p3_bg"], COLOR["p3_text"]),
}

# Known client slugs → canonical names. Grow as new clients appear.
CLIENT_SLUGS: dict[str, str] = {
    "rajasthanhospital": "Rajasthan Hospital",
    "motherhood": "Motherhood",
    "rainbow": "Rainbow",
}

# Summary-prefix tokens to strip before landing on the real client name.
SUMMARY_STRIP_TOKENS = {"live", "observation", "hrms", "bug", "issue", "defect"}

URL_CLIENT_RE = re.compile(r"https?://([a-z0-9-]+)\.medlern\.com", re.IGNORECASE)
PRIORITY_RE = re.compile(r"\bP[0-3]\b")

JIRA_LABEL = "MedLern_Client_Reported"

REQUIRED_JIRA_SECRETS = [
    "JIRA_DOMAIN",
    "JIRA_EMAIL",
    "JIRA_API_TOKEN",
    "JIRA_DEFECT_FILTER_ID",
]


def _get_secret_value(key: str, default: str = "") -> str:
    try:
        value = st.secrets.get(key, default)
    except Exception:
        return default
    if value is None:
        return default
    return str(value).strip()


def validate_jira_config() -> tuple[list[str], dict[str, str]]:
    config = {key: _get_secret_value(key) for key in REQUIRED_JIRA_SECRETS}
    missing = [key for key, value in config.items() if not value]
    return missing, config


def render_missing_config(missing_keys: list[str]) -> None:
    st.error("Jira configuration is incomplete.")
    st.markdown(
        f"""
<div style="background:{COLOR["card_bg"]};border:1px solid {COLOR["border"]};
            border-radius:8px;padding:16px 18px;margin-top:8px">
  <div style="font-size:14px;font-weight:700;color:{COLOR["text_primary"]};margin-bottom:8px">
    Missing required Streamlit secrets
  </div>
  <div style="font-size:13px;color:{COLOR["text_secondary"]};margin-bottom:12px">
    Add these values to local Streamlit secrets or the deployment configuration before loading Jira data.
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.code("\n".join(missing_keys), language="text")


# =====================
# Jira fetch helpers (pattern copied from dashboard.py — kept local on purpose)
# =====================

@st.cache_data(ttl=900, show_spinner=False)
def _jira_search_all(jira_domain: str, email: str, token: str,
                     jql: str, fields: str) -> list[dict]:
    """Paginated search against /rest/api/3/search/jql with nextPageToken."""
    url = f"https://{jira_domain}/rest/api/3/search/jql"
    auth = HTTPBasicAuth(email, token)
    headers = {"Accept": "application/json"}

    all_issues: list[dict] = []
    seen_ids: set[str] = set()
    seen_tokens: set[str] = set()
    page_token: str | None = None
    MAX_PAGES = 200

    for _ in range(MAX_PAGES):
        params: dict[str, Any] = {"jql": jql, "fields": fields, "maxResults": 100}
        if page_token:
            params["nextPageToken"] = page_token
            params["pageToken"] = page_token

        resp = requests.get(url, headers=headers, auth=auth, params=params, timeout=30)
        resp.raise_for_status()
        payload = resp.json()

        added = 0
        for it in payload.get("issues", []) or []:
            iid = str(it.get("id") or "")
            if iid and iid not in seen_ids:
                seen_ids.add(iid)
                all_issues.append(it)
                added += 1

        next_token = payload.get("nextPageToken")
        is_last = bool(payload.get("isLast", False))
        if added == 0 or not next_token or next_token in seen_tokens or is_last:
            break
        seen_tokens.add(next_token)
        page_token = next_token

    return all_issues


@st.cache_data(ttl=86400, show_spinner=False)
def _discover_field_ids(jira_domain: str, email: str, token: str) -> dict[str, str]:
    """Map friendly names ('ui_api_app', 'client') → Jira custom field IDs.

    Cached 24h — field IDs rarely change. Falls back to empty dict on API errors
    so the rest of the dashboard still renders.
    """
    try:
        resp = requests.get(
            f"https://{jira_domain}/rest/api/3/field",
            headers={"Accept": "application/json"},
            auth=HTTPBasicAuth(email, token),
            timeout=30,
        )
        resp.raise_for_status()
    except Exception:
        return {}

    mapping: dict[str, str] = {}
    for f in resp.json() or []:
        fid = f.get("id") or ""
        name = (f.get("name") or "").strip().lower()
        if not fid or not name:
            continue
        if name in ("ui/api/app", "ui / api / app", "ui api app", "uiapiapp"):
            mapping["ui_api_app"] = fid
        elif name == "client":
            mapping["client"] = fid
    return mapping


def _normalize_str(v: Any) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    s = unicodedata.normalize("NFKC", str(v))
    return re.sub(r"\s+", " ", s).strip()


def _normalize_developer(dev_field: Any, assignee_field: Any) -> str:
    """Resolve developer name with fallback chain.

    Priority: Developer custom field (customfield_11012) → Jira assignee →
    literal 'Unassigned'. Custom field handles dict / list / scalar shapes.
    """
    if isinstance(dev_field, dict):
        name = dev_field.get("displayName") or dev_field.get("name") or ""
        if name:
            return name
    elif isinstance(dev_field, list) and dev_field:
        names = [
            d.get("displayName") or d.get("name") or ""
            if isinstance(d, dict) else str(d)
            for d in dev_field
        ]
        joined = ", ".join(n for n in names if n)
        if joined:
            return joined
    elif dev_field:
        return str(dev_field)

    if isinstance(assignee_field, dict):
        name = assignee_field.get("displayName") or assignee_field.get("name") or ""
        if name:
            return name

    return "Unassigned"


def _assignee_name(assignee_field: Any) -> str:
    """Plain Jira assignee displayName, or 'Unassigned'."""
    if isinstance(assignee_field, dict):
        name = assignee_field.get("displayName") or assignee_field.get("name") or ""
        if name:
            return name
    return "Unassigned"


def _normalize_priority(priority_field: Any) -> str:
    """Extract P0/P1/P2/P3 from various Jira priority representations."""
    if not priority_field:
        return ""
    name = priority_field.get("name", "") if isinstance(priority_field, dict) else str(priority_field)
    if not name:
        return ""
    m = PRIORITY_RE.search(name)
    if m:
        return m.group(0)
    # Fallback for text priorities
    lowered = name.strip().lower()
    fallback = {
        "highest": "P0", "critical": "P0", "blocker": "P0",
        "high":    "P1", "major":    "P1",
        "medium":  "P2", "normal":   "P2",
        "low":     "P3", "lowest":   "P3", "minor": "P3", "trivial": "P3",
    }
    return fallback.get(lowered, "")


def _adf_to_text(adf: Any) -> str:
    """Flatten Atlassian Document Format (v3) into plain text. Includes link hrefs."""
    if adf is None:
        return ""
    if isinstance(adf, str):
        return adf
    if not isinstance(adf, dict):
        return ""

    parts: list[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("type") == "text" and node.get("text"):
                parts.append(node["text"])
            for mark in node.get("marks", []) or []:
                if mark.get("type") == "link":
                    href = (mark.get("attrs") or {}).get("href", "")
                    if href:
                        parts.append(f" {href} ")
            for child in node.get("content", []) or []:
                walk(child)
        elif isinstance(node, list):
            for child in node:
                walk(child)

    walk(adf)
    return " ".join(parts)


def _parse_jira_dt(s: str | None) -> pd.Timestamp | None:
    """Parse Jira ISO timestamp → tz-aware IST Timestamp. Returns None if unparseable."""
    if not s:
        return None
    try:
        ts = pd.to_datetime(s, utc=True, errors="coerce")
        if pd.isna(ts):
            return None
        return ts.tz_convert(IST)
    except Exception:
        return None


# =====================
# Data loading
# =====================

@st.cache_data(ttl=900, show_spinner=False)
def load_defect_data() -> tuple[pd.DataFrame, datetime]:
    """Fetch all tickets in filter=JIRA_DEFECT_FILTER_ID (open + closed).

    Returns (raw_df, fetched_at_ist). Single API call — open/closed partitioning
    happens downstream. Cached for 15 minutes; manual refresh clears the cache.
    """
    jira_domain = st.secrets["JIRA_DOMAIN"]
    email = st.secrets["JIRA_EMAIL"]
    token = st.secrets["JIRA_API_TOKEN"]
    filter_id = st.secrets["JIRA_DEFECT_FILTER_ID"]

    field_map = _discover_field_ids(jira_domain, email, token)
    ui_api_app_id = field_map.get("ui_api_app")

    base_fields = [
        "key", "summary", "status", "priority",
        "created", "customfield_10988", "updated",
        "assignee", "components", "description", "labels",
        "customfield_11012",  # Developer (from productivity dashboard)
    ]
    if ui_api_app_id:
        base_fields.append(ui_api_app_id)
    fields_csv = ",".join(base_fields)

    jql = f"filter={filter_id}"
    issues = _jira_search_all(jira_domain, email, token, jql, fields_csv)

    rows: list[dict] = []
    for issue in issues:
        f = issue.get("fields", {}) or {}
        ui_api_app_val = ""
        if ui_api_app_id:
            raw = f.get(ui_api_app_id)
            if isinstance(raw, dict):
                ui_api_app_val = raw.get("value") or raw.get("name") or ""
            elif isinstance(raw, list) and raw:
                first = raw[0]
                ui_api_app_val = first.get("value") if isinstance(first, dict) else str(first)
            elif raw:
                ui_api_app_val = str(raw)

        components_raw = f.get("components") or []
        component_names = [
            c.get("name") for c in components_raw
            if isinstance(c, dict) and c.get("name")
        ]

        rows.append({
            "key": issue.get("key") or "",
            "summary": _normalize_str(f.get("summary", "")),
            "status": _normalize_str((f.get("status") or {}).get("name", "")),
            "priority": _normalize_priority(f.get("priority")),
            "created_raw": f.get("created"),
            "resolved_raw": f.get("customfield_10988"),
            "updated_raw": f.get("updated"),
            "developer": _normalize_developer(
                f.get("customfield_11012"), f.get("assignee"),
            ),
            "assignee": _assignee_name(f.get("assignee")),
            "description_text": _adf_to_text(f.get("description")),
            "labels": f.get("labels") or [],
            "components": component_names,
            "type": _normalize_str(ui_api_app_val),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df, datetime.now(IST)

    df["created"] = df["created_raw"].apply(_parse_jira_dt)
    df["resolved"] = df["resolved_raw"].apply(_parse_jira_dt)
    df["updated"] = df["updated_raw"].apply(_parse_jira_dt)
    df = df.drop(columns=["created_raw", "resolved_raw", "updated_raw"])

    return df, datetime.now(IST)


# =====================
# SLA math — pure functions, no Streamlit imports
# =====================

def calendar_hours(start: datetime, end: datetime) -> float:
    """Elapsed calendar hours between two datetimes. Weekends count."""
    if end <= start:
        return 0.0
    return (end - start).total_seconds() / 3600


def working_days(start: datetime, end: datetime) -> float:
    """Elapsed working days (Mon–Fri, fractional).

    Counts only weekday time. No public-holiday calendar yet — TODO: plug in
    an IST holiday list (e.g., Republic Day, Diwali) and subtract those days.
    """
    if end <= start:
        return 0.0

    total = 0.0
    cur = start
    while cur.date() < end.date():
        if cur.weekday() < 5:
            eod = datetime.combine(cur.date(), datetime.min.time(), tzinfo=cur.tzinfo) + timedelta(days=1)
            total += (eod - cur).total_seconds() / 86400
        cur = datetime.combine(cur.date(), datetime.min.time(), tzinfo=cur.tzinfo) + timedelta(days=1)
    if end.weekday() < 5:
        sod = datetime.combine(end.date(), datetime.min.time(), tzinfo=end.tzinfo)
        total += (end - sod).total_seconds() / 86400
    return total


def compute_sla(priority: str, created: datetime, status: str,
                closed_at: datetime | None, now: datetime) -> dict:
    """Compute SLA elapsed, budget, % consumed, and label for a ticket.

    Rules:
      • Clock starts at `created`.
      • Clock stops ONLY when `status == "Closed"`. No pauses — "Resolved",
        "In Review", "Waiting on Client" do not stop the clock.
      • P0 uses calendar hours; P1/P2/P3 use working days (Mon–Fri IST).
      • label:
          - open tickets:  pct>100 → breached, 75≤pct≤100 → at_risk, else on_track
          - closed tickets: elapsed≤budget → met, else breached
    """
    if priority not in SLA_CONFIG or created is None:
        return {"elapsed": 0.0, "budget": 0.0, "pct": 0, "label": "unknown", "unit": ""}

    kind, budget = SLA_CONFIG[priority]
    is_closed = (status or "").strip().lower() == "closed"
    end = closed_at if (is_closed and closed_at) else now

    elapsed = calendar_hours(created, end) if kind == "calendar_hours" else working_days(created, end)
    pct = (elapsed / budget * 100) if budget else 0

    if is_closed:
        label = "met" if elapsed <= budget else "breached"
    elif pct > 100:
        label = "breached"
    elif pct >= 75:
        label = "at_risk"
    else:
        label = "on_track"

    return {
        "elapsed": elapsed,
        "budget": float(budget),
        "pct": round(pct),
        "label": label,
        "unit": "h" if kind == "calendar_hours" else "wd",
    }


# =====================
# Client extraction
# =====================

def extract_client(description_text: str, summary: str) -> tuple[str, str]:
    """Pick a client name for a ticket. Returns (client_name, extraction_source).

    Priority:
      1. URL match against description: `https?://<slug>.medlern.com`.
         If slug is in CLIENT_SLUGS → use canonical name; else title-case the slug.
      2. Summary prefix before first ':', with known non-client tokens stripped.
      3. Fallback: literal 'CLIENT NOT TAGGED'.
    """
    # 1. URL match
    if description_text:
        m = URL_CLIENT_RE.search(description_text)
        if m:
            slug = m.group(1).lower()
            canonical = CLIENT_SLUGS.get(slug)
            if canonical:
                return canonical, "description_url"
            # Best-guess title-case of unknown slug
            return slug.replace("-", " ").title(), "description_url"

    # 2. Summary prefix
    if summary:
        # Strip leading bracketed tags like "[Observation]"
        cleaned = re.sub(r"^\s*\[[^\]]+\]\s*", "", summary).strip()
        if ":" in cleaned:
            head, _, _ = cleaned.partition(":")
            # Walk colon-separated prefixes: strip known non-client tokens
            for token in [t.strip() for t in head.split(":") if t.strip()]:
                if token.lower() in SUMMARY_STRIP_TOKENS:
                    continue
                # Peel one more layer: summary like "HRMS: YMCH: ..." —
                # head is "HRMS" first. Walk the full prefix chain instead.
                break

            # Full prefix chain walk
            prefix_chain = [t.strip() for t in cleaned.split(":")[:-1] if t.strip()]
            for token in prefix_chain:
                bare = re.sub(r"[\[\]]", "", token).strip()
                if bare.lower() in SUMMARY_STRIP_TOKENS:
                    continue
                if bare:
                    return bare, "summary_prefix"

    return "CLIENT NOT TAGGED", "not_tagged"


# =====================
# Derived dataframe builder
# =====================

@st.cache_data(ttl=300, show_spinner=False)
def build_derived(df: pd.DataFrame, now: datetime) -> pd.DataFrame:
    """Add sla_*, client, age_days, extraction_source columns to the raw df."""
    if df.empty:
        return df.assign(
            client="", extraction_source="", age_days=0.0,
            sla_elapsed=0.0, sla_budget=0.0, sla_pct=0, sla_label="unknown", sla_unit="",
        )

    out = df.copy()

    sla_rows = out.apply(
        lambda r: compute_sla(
            r["priority"],
            r["created"].to_pydatetime() if pd.notna(r["created"]) else None,
            r["status"],
            r["resolved"].to_pydatetime() if pd.notna(r["resolved"]) else None,
            now,
        ),
        axis=1,
    )
    out["sla_elapsed"] = [s["elapsed"] for s in sla_rows]
    out["sla_budget"] = [s["budget"] for s in sla_rows]
    out["sla_pct"] = [s["pct"] for s in sla_rows]
    out["sla_label"] = [s["label"] for s in sla_rows]
    out["sla_unit"] = [s["unit"] for s in sla_rows]

    client_rows = out.apply(
        lambda r: extract_client(r["description_text"] or "", r["summary"] or ""),
        axis=1,
    )
    out["client"] = [c[0] for c in client_rows]
    out["extraction_source"] = [c[1] for c in client_rows]

    def _age(row: pd.Series) -> float:
        created = row.get("created")
        if pd.isna(created):
            return 0.0
        is_closed = str(row.get("status") or "").strip().lower() == "closed"
        resolved = row.get("resolved")
        end = resolved if (is_closed and pd.notna(resolved)) else now
        created_dt = created.to_pydatetime() if hasattr(created, "to_pydatetime") else created
        end_dt = end.to_pydatetime() if hasattr(end, "to_pydatetime") else end
        return working_days(created_dt, end_dt)

    out["age_days"] = out.apply(_age, axis=1)
    return out


# =====================
# CSS + section header (adapted from dashboard.py)
# =====================

def _inject_sla_css_additions() -> None:
    """SLA-specific CSS additions. Called when embedded in the unified dashboard
    so we avoid overriding the shared base styles (button colors, fonts, etc.)."""
    st.markdown(f"""
<style>
/* Hide the invisible iframe that streamlit-autorefresh injects */
div[data-testid="stElementContainer"]:has(iframe[title*="autorefresh"]),
div[data-testid="element-container"]:has(iframe[title*="autorefresh"]) {{
    display: none !important;
}}
iframe[title*="autorefresh"] {{ display: none !important; }}

/* Fresh-chip uses a distinctive teal */
.st-key-chip_fresh .stButton > button {{
    background: #0e7490 !important; color: #fff !important;
    border: 1px solid #0e7490 !important;
}}
.st-key-chip_fresh .stButton > button:hover {{ background: #155e75 !important; }}
.st-key-chip_fresh .stButton > button[kind="primary"] {{
    background: #0891b2 !important; border-color: #0891b2 !important;
    box-shadow: 0 0 0 2px rgba(8,145,178,.25) !important;
}}

/* SLA-met chip — green */
.st-key-chip_met .stButton > button {{
    background: #15803d !important; color: #fff !important;
    border: 1px solid #15803d !important;
}}
.st-key-chip_met .stButton > button:hover {{ background: #166534 !important; }}
.st-key-chip_met .stButton > button[kind="primary"] {{
    background: #16a34a !important; border-color: #16a34a !important;
    box-shadow: 0 0 0 2px rgba(22,163,74,.25) !important;
}}

/* Clickable KPI cards — transparent overlay buttons */
div:has(> :is(.st-key-btn_open_popup, .st-key-btn_reported_popup,
              .st-key-btn_closed_popup,
              .st-key-btn_breached_popup, .st-key-btn_met_popup)) {{
    position: relative !important;
}}
:is(.st-key-btn_open_popup, .st-key-btn_reported_popup,
    .st-key-btn_closed_popup,
    .st-key-btn_breached_popup, .st-key-btn_met_popup) {{
    position: absolute !important; top: 0 !important; left: 0 !important;
    right: 0 !important; bottom: 0 !important; z-index: 10 !important;
    margin: 0 !important; padding: 0 !important; pointer-events: none !important;
}}
:is(.st-key-btn_open_popup, .st-key-btn_reported_popup,
    .st-key-btn_closed_popup,
    .st-key-btn_breached_popup, .st-key-btn_met_popup) > div {{ height: 100% !important; margin: 0 !important; }}
:is(.st-key-btn_open_popup, .st-key-btn_reported_popup,
    .st-key-btn_closed_popup,
    .st-key-btn_breached_popup, .st-key-btn_met_popup) > div > button {{
    height: 100% !important; width: 100% !important;
    background: transparent !important; border: none !important;
    box-shadow: none !important; outline: none !important;
    padding: 0 !important; cursor: pointer !important;
    border-radius: 12px !important; pointer-events: auto !important;
}}
:is(.st-key-btn_open_popup, .st-key-btn_reported_popup,
    .st-key-btn_closed_popup,
    .st-key-btn_breached_popup, .st-key-btn_met_popup) > div > button:hover {{
    background: transparent !important;
}}
</style>
""", unsafe_allow_html=True)


def _inject_css() -> None:
    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {{ font-family: 'Inter', sans-serif !important; }}

.stApp {{ background-color: {COLOR["page_bg"]} !important; }}
.main .block-container {{ padding: 1.5rem 2rem 3rem !important; max-width: 100% !important; }}

header[data-testid="stHeader"],
#MainMenu, footer,
[data-testid="stToolbar"],
[data-testid="stStatusWidget"] {{ display: none !important; }}
.stMainBlockContainer {{ padding-top: 0 !important; }}

/* Collapse the invisible iframe that streamlit-autorefresh injects */
div[data-testid="stElementContainer"]:has(iframe[title*="autorefresh"]),
div[data-testid="element-container"]:has(iframe[title*="autorefresh"]) {{
    display: none !important;
}}
iframe[title*="autorefresh"] {{ display: none !important; }}

h1 {{ font-size: 22px !important; font-weight: 700 !important; color: {COLOR["text_primary"]} !important; }}
h2, h3 {{ color: {COLOR["text_primary"]} !important; }}

.stButton > button {{
    background: {COLOR["accent"]} !important; color: #fff !important;
    border: none !important; border-radius: 6px !important;
    font-weight: 600 !important; font-size: 13px !important;
    padding: 0.4rem 1.1rem !important; transition: background .2s !important;
}}
.stButton > button:hover {{ background: #4f46e5 !important; }}

/* Fresh-chip uses a distinctive teal so it stands out from the palette reds/ambers */
.st-key-chip_fresh .stButton > button {{
    background: #0e7490 !important;
    color: #fff !important;
    border: 1px solid #0e7490 !important;
}}
.st-key-chip_fresh .stButton > button:hover {{ background: #155e75 !important; }}
.st-key-chip_fresh .stButton > button[kind="primary"] {{
    background: #0891b2 !important;
    border-color: #0891b2 !important;
    box-shadow: 0 0 0 2px rgba(8,145,178,.25) !important;
}}

/* SLA-met chip — green to match the SLA 'met' status color */
.st-key-chip_met .stButton > button {{
    background: #15803d !important;
    color: #fff !important;
    border: 1px solid #15803d !important;
}}
.st-key-chip_met .stButton > button:hover {{ background: #166534 !important; }}
.st-key-chip_met .stButton > button[kind="primary"] {{
    background: #16a34a !important;
    border-color: #16a34a !important;
    box-shadow: 0 0 0 2px rgba(22,163,74,.25) !important;
}}

[data-testid="stSelectbox"] > div > div {{
    background: {COLOR["card_bg"]} !important; border: 1px solid {COLOR["border"]} !important;
    border-radius: 6px !important; color: {COLOR["text_primary"]} !important;
}}

[data-testid="stExpander"] {{
    background: {COLOR["card_bg"]} !important; border: 1px solid {COLOR["border"]} !important;
    border-radius: 10px !important; overflow: hidden !important;
    box-shadow: 0 1px 3px rgba(0,0,0,.06) !important;
}}
[data-testid="stExpander"] summary {{
    color: {COLOR["text_primary"]} !important; font-weight: 600 !important; font-size: 14px !important;
    padding: 12px 16px !important;
}}
[data-testid="stExpander"] summary:hover {{ background: #f1f5f9 !important; }}

hr {{ border-color: {COLOR["border"]} !important; margin: 1.5rem 0 !important; }}

[data-testid="stCaptionContainer"], .stCaption {{ color: {COLOR["text_secondary"]} !important; }}

[data-testid="stAlert"] {{
    background: {COLOR["card_bg"]} !important; border-radius: 8px !important;
    border-left-color: {COLOR["accent"]} !important;
}}

[data-testid="stMultiSelect"] > div {{
    background: {COLOR["card_bg"]} !important; border: 1px solid {COLOR["border"]} !important;
    border-radius: 6px !important;
}}

.stSpinner > div {{ border-top-color: {COLOR["accent"]} !important; }}

::-webkit-scrollbar {{ width: 5px; height: 5px; }}
::-webkit-scrollbar-track {{ background: #f1f5f9; }}
::-webkit-scrollbar-thumb {{ background: #cbd5e1; border-radius: 3px; }}
::-webkit-scrollbar-thumb:hover {{ background: #94a3b8; }}

[data-testid="stSelectbox"] > div > div {{
    box-shadow: 0 1px 3px rgba(0,0,0,.05) !important;
}}

/* Clickable KPI cards — transparent overlay button, position:absolute so it
   takes zero space in normal flow and matches the card's exact dimensions. */
div:has(> :is(.st-key-btn_open_popup, .st-key-btn_reported_popup,
              .st-key-btn_closed_popup,
              .st-key-btn_breached_popup, .st-key-btn_met_popup)) {{
    position: relative !important;
}}
:is(.st-key-btn_open_popup, .st-key-btn_reported_popup,
    .st-key-btn_closed_popup,
    .st-key-btn_breached_popup, .st-key-btn_met_popup) {{
    position: absolute !important;
    top: 0 !important; left: 0 !important;
    right: 0 !important; bottom: 0 !important;
    z-index: 10 !important;
    margin: 0 !important; padding: 0 !important;
    pointer-events: none !important;
}}
:is(.st-key-btn_open_popup, .st-key-btn_reported_popup,
    .st-key-btn_closed_popup,
    .st-key-btn_breached_popup, .st-key-btn_met_popup) > div {{
    height: 100% !important;
    margin: 0 !important;
}}
:is(.st-key-btn_open_popup, .st-key-btn_reported_popup,
    .st-key-btn_closed_popup,
    .st-key-btn_breached_popup, .st-key-btn_met_popup) > div > button {{
    height: 100% !important;
    width: 100% !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
    padding: 0 !important;
    cursor: pointer !important;
    border-radius: 12px !important;
    pointer-events: auto !important;
}}
:is(.st-key-btn_open_popup, .st-key-btn_reported_popup,
    .st-key-btn_closed_popup,
    .st-key-btn_breached_popup, .st-key-btn_met_popup) > div > button:hover {{
    background: transparent !important;
}}
</style>
""", unsafe_allow_html=True)


def _section_header(title: str, subtitle: str = "") -> None:
    sub_html = (
        f'<p style="font-size:13px;color:{COLOR["text_secondary"]};'
        f'margin:4px 0 0 15px;font-weight:400">{subtitle}</p>'
        if subtitle else ""
    )
    st.markdown(f"""
<div style="margin:8px 0 16px">
  <div style="display:flex;align-items:center;gap:10px">
    <div style="width:3px;height:22px;background:{COLOR["accent"]};
                border-radius:2px;flex-shrink:0"></div>
    <span style="font-size:17px;font-weight:700;color:{COLOR["text_primary"]};
                 letter-spacing:-.01em">{title}</span>
  </div>
  {sub_html}
</div>""", unsafe_allow_html=True)


# =====================
# UI builders
# =====================

PERIOD_TYPES = ["Current Month", "Current Year", "This Quarter", "Last Quarter", "Custom"]


def _get_period_bounds(period_type: str, today: date) -> tuple[date, date]:
    """Absolute (start, end) for a named period. Upper bound clipped to today."""
    if period_type == "Current Year":
        start = date(today.year, 1, 1)
        end = date(today.year, 12, 31)
    elif period_type == "This Quarter":
        q_month = 3 * ((today.month - 1) // 3) + 1
        start = date(today.year, q_month, 1)
        next_q = q_month + 3
        end = (date(today.year + 1, 1, 1) - timedelta(days=1)
               if next_q > 12 else date(today.year, next_q, 1) - timedelta(days=1))
    elif period_type == "Last Quarter":
        cur_q = 3 * ((today.month - 1) // 3) + 1
        end = date(today.year, cur_q, 1) - timedelta(days=1)
        lq = 3 * ((end.month - 1) // 3) + 1
        start = date(end.year, lq, 1)
    else:  # Current Month (default)
        start = today.replace(day=1)
        if today.month == 12:
            end = date(today.year + 1, 1, 1) - timedelta(days=1)
        else:
            end = date(today.year, today.month + 1, 1) - timedelta(days=1)
    return start, min(end, today)


def _render_header_bar(jira_domain: str, total_open: int, fetched_at: datetime,
                       today: date) -> tuple[date, date]:
    """Top header bar. Returns (start_date, end_date)."""
    sync_str = fetched_at.strftime("%d %b %Y · %H:%M %Z")

    st.markdown(f"""
<div style="display:flex;align-items:center;gap:12px;padding:6px 0 14px;
            border-bottom:1px solid {COLOR["border"]};margin-bottom:4px">
  <div style="width:36px;height:36px;
              background:linear-gradient(135deg,{COLOR["red_fill"]},{COLOR["amber_fill"]});
              border-radius:9px;display:flex;align-items:center;justify-content:center;
              font-size:16px;flex-shrink:0;color:#fff;font-weight:700;
              box-shadow:0 2px 6px rgba(163,45,45,.3)">!</div>
  <div style="flex:1">
    <div style="font-size:18px;font-weight:700;color:{COLOR["text_primary"]};
                letter-spacing:-0.3px;line-height:1.2">Client Issues Tracker</div>
    <div style="font-size:11px;color:{COLOR["text_tertiary"]};margin-top:2px;
                font-weight:500;letter-spacing:0.04em;text-transform:uppercase">
      MDLRN · {total_open} open defects · synced {sync_str}
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    def _period_label(pt: str) -> str:
        if pt == "Custom":
            return "Custom"
        s, e = _get_period_bounds(pt, today)
        return f"{pt}  ({s.strftime('%d %b')} – {e.strftime('%d %b %Y')})"

    period_labels = [_period_label(pt) for pt in PERIOD_TYPES]

    st.markdown(
        f"<div style='margin-top:12px'>"
        f"<p style='font-size:11px;font-weight:600;color:{COLOR['text_secondary']};"
        f"letter-spacing:.06em;text-transform:uppercase;margin-bottom:6px'>"
        f"Period</p></div>",
        unsafe_allow_html=True,
    )
    _period_col, _ = st.columns([2, 3])
    with _period_col:
        selected_label = st.selectbox(
            "Period", period_labels,
            index=0,  # Current Month default
            label_visibility="collapsed", key="period_choice",
        )
        period_type = PERIOD_TYPES[period_labels.index(selected_label)]

    # Custom date pickers (only when Custom is selected)
    if period_type == "Custom":
        d1, d2, _blank = st.columns([2, 2, 3])
        with d1:
            custom_start = st.date_input(
                "From", value=today.replace(day=1), key="custom_from",
            )
        with d2:
            custom_end = st.date_input("To", value=today, key="custom_to")
        start_date = custom_start
        end_date = min(custom_end, today)
    else:
        start_date, end_date = _get_period_bounds(period_type, today)

    return start_date, end_date


@st.cache_data(ttl=300, show_spinner=False)
def _kpi_snapshot_at(all_df: pd.DataFrame, period_start: date, period_end: date,
                     as_of: datetime) -> dict:
    """Reconstruct KPI counts as of `as_of`.

    Live metrics (open, at_risk): tickets not yet closed as of as_of.
    Period metrics (breached, met, avg_resolution, compliance): tickets closed
    within [period_start, min(period_end, as_of.date())], matching the same
    resolution-date pool used by the main KPI tiles.
    """
    empty = {
        "open": 0, "at_risk": 0, "breached": 0,
        "met": 0, "n_closed": 0,
        "avg_resolution_days": None, "compliance_pct": 100.0,
        "reported": 0,
    }
    if all_df.empty:
        return empty

    filtered = all_df[all_df["created"] <= as_of].copy()
    if filtered.empty:
        return empty

    is_closed_final = filtered["status"].str.lower() == "closed"
    closed_by = (
        is_closed_final
        & filtered["resolved"].notna()
        & (filtered["resolved"] <= as_of)
    )
    open_snap = filtered[~closed_by]
    closed_snap = filtered[closed_by]

    # Live: open + at_risk (same compute_sla loop, no breached — breached is period-scoped)
    n_at_risk = n_on_track = 0
    for _, r in open_snap.iterrows():
        sla = compute_sla(
            r["priority"],
            r["created"].to_pydatetime() if pd.notna(r["created"]) else None,
            "open", None, as_of,
        )
        if sla["label"] == "at_risk":
            n_at_risk += 1
        elif sla["label"] == "on_track":
            n_on_track += 1

    # Period: closed in [period_start, min(period_end, as_of.date())]
    effective_end = min(period_end, as_of.date())
    closed_period = closed_snap[
        (closed_snap["resolved"].dt.date >= period_start)
        & (closed_snap["resolved"].dt.date <= effective_end)
    ]
    n_closed = len(closed_period)
    if n_closed > 0:
        n_met = int((closed_period["sla_label"] == "met").sum())
        n_breached = int((closed_period["sla_label"] == "breached").sum())
        avg_res: float | None = float(
            closed_period.apply(
                lambda r: working_days(r["created"].to_pydatetime(), r["resolved"].to_pydatetime()), axis=1
            ).mean()
        )
        period_tracked = closed_period[
            closed_period["sla_label"].isin(["on_track", "at_risk", "breached", "met"])
        ]
        _n_pt = len(period_tracked)
        compliance = round(
            int((period_tracked["sla_label"].isin(["on_track", "at_risk", "met"])).sum()) / _n_pt * 100, 1
        ) if _n_pt else 100.0
    else:
        n_met = n_breached = 0
        avg_res = None
        compliance = 100.0

    reported_snap = int((
        filtered["created"].notna()
        & (filtered["created"].dt.date >= period_start)
        & (filtered["created"].dt.date <= effective_end)
    ).sum())

    return {
        "open": len(open_snap),
        "at_risk": n_at_risk,
        "breached": n_breached,
        "met": n_met,
        "n_closed": n_closed,
        "avg_resolution_days": avg_res,
        "compliance_pct": compliance,
        "reported": reported_snap,
    }



def _fmt_wow_delta(diff: float | int | None, unit: str, invert: bool) -> tuple[str, str, str]:
    """Format week-over-week delta. Returns (text, bg, fg). `invert=True` means
    higher values are worse for this metric (e.g. open count, breach count)."""
    if diff is None:
        return "—", "#f1f5f9", COLOR["text_tertiary"]
    if abs(diff) < 0.05:
        return "no change", "#f1f5f9", COLOR["text_secondary"]
    is_good = (diff > 0) if not invert else (diff < 0)
    arrow = "↑" if diff > 0 else "↓"
    sign = "+" if diff > 0 else ""
    if isinstance(diff, float) and abs(diff - round(diff)) > 0.05:
        body = f"{sign}{diff:.1f}{unit}"
    else:
        body = f"{sign}{int(round(diff))}{unit}"
    bg = "#dcfce7" if is_good else "#fee2e2"
    fg = "#16a34a" if is_good else "#dc2626"
    return f"{body} {arrow}", bg, fg


def _render_health_banner(open_df: pd.DataFrame, closed_recent_df: pd.DataFrame,
                          start_date: date, end_date: date) -> None:
    """Top-of-page health status banner with enriched executive summary."""
    n_open = len(open_df)
    n_at_risk = int((open_df["sla_label"] == "at_risk").sum())
    n_closed = len(closed_recent_df)

    # SLA breach + compliance use the same pool as SLA Met: tickets RESOLVED in the period.
    period_tracked = closed_recent_df[closed_recent_df["sla_label"].isin(["on_track", "at_risk", "breached", "met"])]
    n_breached = int((closed_recent_df["sla_label"] == "breached").sum())
    _n_pt = len(period_tracked)
    compliance_pct = round(
        int((period_tracked["sla_label"].isin(["on_track", "at_risk", "met"])).sum()) / _n_pt * 100, 1
    ) if _n_pt else 100.0

    if compliance_pct >= 95:
        status, icon = "HEALTHY", "✓"
        fill, bg, border = COLOR["green_fill"], COLOR["green_bg"], "#86efac"
    elif compliance_pct >= 80:
        status, icon = "AT RISK", "⚠"
        fill, bg, border = COLOR["amber_fill"], COLOR["amber_bg"], "#fcd34d"
    else:
        status, icon = "CRITICAL", "✕"
        fill, bg, border = COLOR["red_fill"], COLOR["red_bg"], "#fca5a5"

    p0_breach = int(((closed_recent_df["sla_label"] == "breached") & (closed_recent_df["priority"] == "P0")).sum())
    p1_breach = int(((closed_recent_df["sla_label"] == "breached") & (closed_recent_df["priority"] == "P1")).sum())

    if n_breached == 0 and n_at_risk == 0:
        defect_word = "defect" if n_open == 1 else "defects"
        summary = f"All {n_open} open {defect_word} are on track."
    elif n_breached == 0:
        summary = f"{n_open} open defect{'s' if n_open != 1 else ''} — {n_at_risk} approaching SLA deadline."
    else:
        parts = ([f"{p0_breach} P0"] if p0_breach else []) + ([f"{p1_breach} P1"] if p1_breach else [])
        breach_detail = f" ({', '.join(parts)})" if parts else ""
        summary = (
            f"{n_open} open defect{'s' if n_open != 1 else ''} — "
            f"{n_breached} breach SLA{breach_detail}, {n_at_risk} at risk."
        )
    closed_note = f" {n_closed} resolved in the last 30 days." if n_closed else ""

    # Extra metrics
    oldest_age = float(open_df["age_days"].max()) if not open_df.empty else 0.0
    oldest_str = f"{oldest_age * 8:.0f}h" if oldest_age < 1 else f"{oldest_age:.0f}d"

    avg_res_str = "—"
    if not closed_recent_df.empty:
        wd_series = closed_recent_df.apply(
            lambda r: working_days(r["created"].to_pydatetime(), r["resolved"].to_pydatetime()), axis=1
        )
        avg_res_str = f"{float(wd_series.mean()):.1f}d"

    tagged = open_df[open_df["client"] != "CLIENT NOT TAGGED"]
    n_clients = int(tagged["client"].nunique()) if not tagged.empty else 0
    clients_chip = (
        f'<span style="background:{COLOR["page_bg"]};color:{COLOR["text_primary"]};'
        f'padding:3px 10px;border-radius:20px;font-size:11px;border:1px solid {COLOR["border"]}">'
        f'clients affected: <strong>{n_clients}</strong></span>'
    ) if n_clients else ""

    comp_bg = fill + "22"
    period_label = f"{start_date.strftime('%d %b')} – {end_date.strftime('%d %b %Y')}"
    scope_lbl = (
        f'font-size:9px;color:{COLOR["text_tertiary"]};text-transform:uppercase;'
        f'letter-spacing:.08em;font-weight:600;margin-bottom:5px'
    )
    chip = (
        f'background:{COLOR["page_bg"]};padding:3px 10px;border-radius:20px;'
        f'font-size:11px;border:1px solid {COLOR["border"]}'
    )
    stats_html = f"""
<div style="margin-top:10px">
  <div style="{scope_lbl}">Live snapshot — all open defects</div>
  <div style="display:flex;flex-wrap:wrap;gap:6px">
    <span style="{chip};color:{COLOR["text_primary"]}">⏱ oldest open: <strong>{oldest_str}</strong></span>
    {clients_chip}
  </div>
</div>
<div style="margin-top:8px">
  <div style="{scope_lbl}">{period_label}</div>
  <div style="display:flex;flex-wrap:wrap;gap:6px">
    <span style="background:{comp_bg};color:{fill};padding:3px 10px;border-radius:20px;
                 font-size:11px;font-weight:700">SLA {compliance_pct:.0f}%</span>
    <span style="{chip};color:{COLOR["text_secondary"]}">target 95%</span>
    <span style="{chip};color:{COLOR["text_primary"]}">✓ avg close: <strong>{avg_res_str}</strong></span>
    <span style="{chip};color:{COLOR["text_secondary"]}">closed: <strong>{n_closed}</strong></span>
  </div>
</div>"""

    # Per-priority breakdown — only priorities with open tickets
    pri_cells = ""
    for pri, pri_bg, pri_text in [
        ("P0", COLOR["p0_bg"], COLOR["p0_text"]),
        ("P1", COLOR["p1_bg"], COLOR["p1_text"]),
        ("P2", COLOR["p2_bg"], COLOR["p2_text"]),
        ("P3", COLOR["p3_bg"], COLOR["p3_text"]),
    ]:
        pri_df = open_df[open_df["priority"] == pri]
        if pri_df.empty:
            continue
        n_b = int((pri_df["sla_label"] == "breached").sum())
        n_r = int((pri_df["sla_label"] == "at_risk").sum())
        n_t = int((pri_df["sla_label"] == "on_track").sum())
        n_total = len(pri_df)
        detail_parts = []
        if n_b:
            detail_parts.append(f'<span style="color:{COLOR["red_fill"]};font-weight:700">{n_b} breach</span>')
        if n_r:
            detail_parts.append(f'<span style="color:{COLOR["amber_fill"]};font-weight:600">{n_r} at-risk</span>')
        if n_t:
            detail_parts.append(f'<span style="color:{COLOR["text_secondary"]}">{n_t} on-track</span>')
        if not detail_parts:
            detail_parts.append(f'<span style="color:{COLOR["text_secondary"]}">{n_total} open</span>')
        detail = " · ".join(detail_parts)
        pri_cells += (
            f'<div style="display:flex;align-items:baseline;gap:5px;padding-right:20px">'
            f'<span style="background:{pri_bg};color:{pri_text};font-size:10px;font-weight:700;'
            f'padding:2px 7px;border-radius:6px;letter-spacing:.04em;flex-shrink:0">{pri}</span>'
            f'<span style="font-size:12px">{detail}</span>'
            f'</div>'
        )

    priority_row = (
        f'<div style="margin-top:10px;padding-top:10px;border-top:1px solid {border}">'
        f'<div style="{scope_lbl}">Live snapshot — by priority</div>'
        f'<div style="display:flex;flex-wrap:wrap;gap:4px">{pri_cells}</div>'
        f'</div>'
    ) if pri_cells else ""

    st.markdown(f"""
<div style="background:{bg};border:1.5px solid {border};border-radius:12px;
            padding:16px 20px;margin-bottom:12px;box-shadow:0 1px 4px rgba(0,0,0,.05)">
  <div style="display:flex;align-items:center;gap:16px">
    <div style="background:{fill};color:#fff;border-radius:8px;padding:10px 14px;
                font-size:12px;font-weight:700;letter-spacing:.1em;white-space:nowrap;
                min-width:90px;text-align:center;flex-shrink:0">{icon} {status}</div>
    <div style="font-size:13px;font-weight:600;color:{COLOR["text_primary"]};line-height:1.5">
      {summary}{closed_note}
    </div>
  </div>
  {stats_html}
  {priority_row}
</div>
""", unsafe_allow_html=True)



def _render_kpi_cards(open_df: pd.DataFrame, closed_recent_df: pd.DataFrame,
                      prev_snap: dict, start_date: date, end_date: date,
                      all_df: pd.DataFrame, jira_domain: str,
                      reported_df: pd.DataFrame | None = None) -> None:
    n_open = len(open_df)

    if reported_df is None:
        reported_df = all_df[
            all_df["created"].notna()
            & (all_df["created"].dt.date >= start_date)
            & (all_df["created"].dt.date <= end_date)
        ].copy()
    n_reported = len(reported_df)

    if not closed_recent_df.empty:
        wd_series = closed_recent_df.apply(
            lambda r: working_days(r["created"].to_pydatetime(), r["resolved"].to_pydatetime()), axis=1
        )
        avg_days_num: float | None = float(wd_series.mean())
        avg_resolution = f"{avg_days_num:.1f}d"
        n_closed = len(closed_recent_df)
        n_met = int((closed_recent_df["sla_label"] == "met").sum())
        met_pct = round(n_met / n_closed * 100) if n_closed else 0
    else:
        avg_resolution = "—"
        avg_days_num = None
        n_closed = 0
        n_met = 0
        met_pct = 0

    breached_df = closed_recent_df[closed_recent_df["sla_label"] == "breached"].copy()
    met_df      = closed_recent_df[closed_recent_df["sla_label"] == "met"].copy()

    n_breached = len(breached_df)
    p0_breach = int(((breached_df["priority"] == "P0")).sum())
    p1_breach = int(((breached_df["priority"] == "P1")).sum())
    period_tracked = closed_recent_df[closed_recent_df["sla_label"].isin(["on_track", "at_risk", "breached", "met"])]
    _n_pt = len(period_tracked)
    compliance_pct = round(
        int((period_tracked["sla_label"].isin(["on_track", "at_risk", "met"])).sum()) / _n_pt * 100, 1
    ) if _n_pt else 100.0

    comp_accent = (
        COLOR["green_fill"] if compliance_pct >= 95
        else (COLOR["amber_fill"] if compliance_pct >= 80 else COLOR["red_fill"])
    )
    met_secondary = (
        f"{n_met} of {n_closed} resolved within SLA ({met_pct}%)"
        if n_closed else "No resolved defects in period"
    )

    prev_avg = prev_snap.get("avg_resolution_days")
    avg_diff = (avg_days_num - prev_avg) if (avg_days_num is not None and prev_avg is not None) else None
    deltas = {
        "open":       (n_open     - prev_snap.get("open", 0),              "",  True),
        "reported":   (n_reported - prev_snap.get("reported", 0),          "",  True),
        "closed":     (n_closed   - prev_snap.get("n_closed", 0),           "",  False),
        "compliance": (compliance_pct - prev_snap.get("compliance_pct", 100.0), "%", False),
        "breached":   (n_breached - prev_snap.get("breached", 0),          "",  True),
        "met":        (n_met      - prev_snap.get("met", 0),               "",  False),
    }

    live_scope   = "live · all open"
    period_scope = f"{start_date.strftime('%d %b')} – {end_date.strftime('%d %b %Y')}"
    created_scope = f"created · {period_scope}"
    resolved_scope = f"resolved · {period_scope}"

    # Tile tuple: (label, scope, value, secondary, accent, delta_key, btn_key, popup_func, popup_df)
    # btn_key=None means the card is not clickable.
    closed_secondary = (
        f"{n_met} within SLA · {n_breached} breached"
        if n_closed else "No resolved defects in period"
    )
    row1 = [
        ("Open defects",   live_scope,     str(n_open),              "All open, any created date",       COLOR["text_secondary"], "open",       "btn_open_popup",     _show_open_dialog,     open_df),
        ("Reported",       created_scope,  str(n_reported),          "Created in selected period",       COLOR["accent"],         "reported",   "btn_reported_popup", _show_reported_dialog, reported_df),
        ("Closed",         resolved_scope, str(n_closed),            closed_secondary,                   COLOR["text_secondary"], "closed",     "btn_closed_popup",   _show_closed_dialog,   closed_recent_df),
    ]
    breach_parts = [f"{p0_breach} P0" if p0_breach else None, f"{p1_breach} P1" if p1_breach else None]
    breach_secondary = (
        " · ".join(p for p in breach_parts if p) + " critical"
        if any(breach_parts) else "No P0 or P1 breaches"
    ) if n_breached else "No breaches in period"
    row2 = [
        ("SLA breached",   resolved_scope, str(n_breached),  breach_secondary,                                       COLOR["red_fill"],       "breached", "btn_breached_popup", _show_breached_dialog, breached_df),
        ("SLA met",        resolved_scope, str(n_met),       met_secondary,                                          COLOR["green_fill"],     "met",      "btn_met_popup",      _show_met_dialog,      met_df),
        ("SLA compliance", resolved_scope, f"{compliance_pct:.0f}%", "Closed in period · target 95%",               comp_accent,             "compliance", None,               None,                  None),
    ]

    def _render_row(tiles: list) -> None:
        cols = st.columns(3)
        for col, (label, scope, value, secondary, accent, delta_key, btn_key, popup_func, popup_df) in zip(cols, tiles):
            diff, unit, invert = deltas[delta_key]
            delta_text, delta_bg, delta_fg = _fmt_wow_delta(diff, unit, invert)
            delta_html = (
                f'<div style="display:inline-block;background:{delta_bg};border-radius:10px;'
                f'padding:2px 7px;font-size:10px;color:{delta_fg};font-weight:600;margin-bottom:4px">'
                f'{delta_text} <span style="opacity:.7;font-weight:500">vs 7d ago</span></div>'
            )
            with col:
                st.markdown(f"""
<div style="background:{COLOR["card_bg"]};border-radius:12px;padding:16px 16px;
            border:1px solid {COLOR["border"]};border-top:3px solid {accent};
            box-shadow:0 2px 8px rgba(0,0,0,.07)">
  <div style="font-size:11px;color:{COLOR["text_secondary"]};letter-spacing:.07em;
              text-transform:uppercase;font-weight:600;margin-bottom:2px">{label}</div>
  <div style="font-size:10px;color:{COLOR["text_tertiary"]};letter-spacing:.04em;
              font-weight:500;margin-bottom:7px">{scope}</div>
  <div style="font-size:26px;font-weight:700;color:{COLOR["text_primary"]};
              line-height:1;margin-bottom:7px;letter-spacing:-.02em">{value}</div>
  {delta_html}
  <div style="font-size:11px;color:{COLOR["text_secondary"]};font-weight:500;margin-top:4px">{secondary}</div>
</div>
""", unsafe_allow_html=True)
                if btn_key:
                    if st.button(" ", key=btn_key, use_container_width=True):
                        popup_func(popup_df, jira_domain)

    _render_row(row1)
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    _render_row(row2)


def _render_priority_bars(open_df: pd.DataFrame) -> None:
    """Stacked horizontal bar per priority showing breached/at_risk/on_track counts."""
    priority_hints = {
        "P0": "Critical · SLA: 8 calendar hours",
        "P1": "High · SLA: 3 working days",
        "P2": "Medium · SLA: 7 working days",
        "P3": "Low · SLA: 15 working days",
    }

    rows_html = ""
    for pri in PRIORITY_ORDER:
        subset = open_df[open_df["priority"] == pri]
        n_total = len(subset)
        n_breach = int((subset["sla_label"] == "breached").sum())
        n_risk = int((subset["sla_label"] == "at_risk").sum())
        n_track = int((subset["sla_label"] == "on_track").sum())

        if n_total == 0:
            bar = (
                f'<div style="height:28px;background:#f1f5f9;border-radius:6px;opacity:.5;'
                f'display:flex;align-items:center;justify-content:center;'
                f'font-size:11px;color:{COLOR["text_tertiary"]}">No open tickets</div>'
            )
        else:
            segments = []
            for (count, fill, text_clr) in [
                (n_breach, COLOR["red_fill"],   "#fff"),
                (n_risk,   COLOR["amber_fill"], "#fff"),
                (n_track,  COLOR["green_fill"], "#fff"),
            ]:
                if count <= 0:
                    continue
                pct = count / n_total * 100
                segments.append(
                    f'<div style="width:{pct:.2f}%;background:{fill};'
                    f'display:flex;align-items:center;justify-content:center;'
                    f'font-size:12px;font-weight:700;color:{text_clr}">{count}</div>'
                )
            bar = (
                f'<div style="height:28px;border-radius:6px;overflow:hidden;'
                f'display:flex;border:1px solid {COLOR["border"]}">{"".join(segments)}</div>'
            )

        rows_html += f"""
<div style="margin-bottom:14px">
  <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:6px">
    <div>
      <span style="font-size:13px;font-weight:700;color:{COLOR["text_primary"]}">{pri}</span>
      <span style="font-size:11px;color:{COLOR["text_tertiary"]};margin-left:8px">{priority_hints[pri]}</span>
    </div>
    <span style="font-size:12px;color:{COLOR["text_secondary"]};font-weight:500">{n_total} open</span>
  </div>
  {bar}
</div>
"""

    legend = (
        f'<div style="display:flex;gap:14px;margin-top:6px;font-size:11px;color:{COLOR["text_secondary"]}">'
        f'<span><span style="display:inline-block;width:10px;height:10px;background:{COLOR["red_fill"]};border-radius:2px;margin-right:4px"></span>Breached</span>'
        f'<span><span style="display:inline-block;width:10px;height:10px;background:{COLOR["amber_fill"]};border-radius:2px;margin-right:4px"></span>At risk</span>'
        f'<span><span style="display:inline-block;width:10px;height:10px;background:{COLOR["green_fill"]};border-radius:2px;margin-right:4px"></span>On track</span>'
        f'</div>'
    )

    st.markdown(f"""
<div style="background:{COLOR["card_bg"]};border:1px solid {COLOR["border"]};
            border-radius:12px;padding:18px 20px;box-shadow:0 1px 4px rgba(0,0,0,.06)">
  <div style="font-size:13px;font-weight:700;color:{COLOR["text_primary"]};margin-bottom:14px">
    SLA status by priority
  </div>
  {rows_html}
  {legend}
</div>
""", unsafe_allow_html=True)


def _render_aging_buckets(open_df: pd.DataFrame) -> None:
    """Aging histogram for open tickets across 5 age buckets."""
    buckets = [
        ("0–1d",    0,    1,   COLOR["green_fill"]),
        ("2–7d",    1,    7,   "#D2D250"),
        ("8–30d",   7,    30,  COLOR["amber_fill"]),
        ("31–180d", 30,   180, "#D96F2B"),
        ("180+d",   180,  1e9, COLOR["red_fill"]),
    ]

    counts = []
    for label, lo, hi, _ in buckets:
        n = int(((open_df["age_days"] > lo) & (open_df["age_days"] <= hi)).sum())
        counts.append(n)
    max_c = max(counts) if counts else 0

    rows_html = ""
    for (label, _, _, fill), n in zip(buckets, counts):
        width_pct = (n / max_c * 100) if max_c else 0
        rows_html += f"""
<div style="display:flex;align-items:center;gap:10px;margin-bottom:10px">
  <div style="width:70px;font-size:12px;color:{COLOR["text_secondary"]};font-weight:500">{label}</div>
  <div style="flex:1;background:#f1f5f9;border-radius:4px;height:16px;position:relative">
    <div style="width:{width_pct:.1f}%;background:{fill};height:16px;border-radius:4px;min-width:{2 if n else 0}px"></div>
  </div>
  <div style="width:30px;text-align:right;font-size:13px;font-weight:700;color:{COLOR["text_primary"]}">{n}</div>
</div>
"""

    oldest_caption = ""
    if not open_df.empty:
        oldest = open_df.sort_values("age_days", ascending=False).iloc[0]
        oldest_caption = (
            f'<div style="font-size:11px;color:{COLOR["text_tertiary"]};'
            f'margin-top:10px;padding-top:10px;border-top:1px solid {COLOR["border"]}">'
            f'Oldest: <span style="font-family:ui-monospace,monospace;color:{COLOR["text_secondary"]}">'
            f'{oldest["key"]}</span> · {oldest["age_days"]:.0f} days old'
            f'</div>'
        )

    st.markdown(f"""
<div style="background:{COLOR["card_bg"]};border:1px solid {COLOR["border"]};
            border-radius:12px;padding:18px 20px;box-shadow:0 1px 4px rgba(0,0,0,.06)">
  <div style="font-size:13px;font-weight:700;color:{COLOR["text_primary"]};margin-bottom:14px">
    Aging of open tickets
  </div>
  {rows_html}
  {oldest_caption}
</div>
""", unsafe_allow_html=True)


def _render_median_resolution(closed_recent_df: pd.DataFrame) -> None:
    """Per-priority median resolution stat row, last 30d closed tickets."""
    if closed_recent_df.empty:
        medians_html = " · ".join(f"{p} median: —" for p in PRIORITY_ORDER)
    else:
        parts = []
        for pri in PRIORITY_ORDER:
            subset = closed_recent_df[closed_recent_df["priority"] == pri]
            if subset.empty:
                parts.append(f"<b>{pri}</b> median: —")
                continue
            wd_vals = subset.apply(
                lambda r: working_days(r["created"].to_pydatetime(), r["resolved"].to_pydatetime()), axis=1
            )
            parts.append(f"<b>{pri}</b> median: {float(wd_vals.median()):.1f}d")
        medians_html = " · ".join(parts)

    st.markdown(f"""
<div style="font-size:11px;color:{COLOR["text_secondary"]};margin-top:10px;
            padding:8px 14px;background:{COLOR["card_bg"]};
            border:1px solid {COLOR["border"]};border-radius:8px">
  <span style="color:{COLOR["text_tertiary"]};margin-right:8px">MEDIAN RESOLUTION · LAST 30D:</span>
  {medians_html}
</div>
""", unsafe_allow_html=True)


STATUS_STYLE_RULES: list[tuple[tuple[str, ...], tuple[str, str]]] = [
    (("closed", "done", "resolved", "fixed", "released"),  ("#dcfce7", "#15803d")),
    (("in progress", "in development", "development", "coding", "working"),
                                                            ("#dbeafe", "#1d4ed8")),
    (("review", "code review", "peer review"),             ("#ede9fe", "#6d28d9")),
    (("qa", "testing", "ready for qa", "in qa", "verification"),
                                                            ("#fef3c7", "#b45309")),
    (("blocked", "on hold", "waiting", "pending"),         ("#fee2e2", "#b91c1c")),
    (("open", "to do", "backlog", "new", "created", "reopened"),
                                                            ("#e0e7ef", "#334155")),
]


def _status_style(status: str) -> tuple[str, str]:
    """Return (bg, fg) for a status name. Falls back to neutral grey-blue."""
    s = (status or "").strip().lower()
    if not s:
        return ("#f1f5f9", "#64748b")
    for keywords, colors in STATUS_STYLE_RULES:
        if any(k in s for k in keywords):
            return colors
    return ("#e0e7ef", "#334155")


def _pill(text: str, bg: str, fg: str, mono: bool = False) -> str:
    fam = "font-family:ui-monospace,SFMono-Regular,monospace;" if mono else ""
    return (
        f'<span style="background:{bg};color:{fg};padding:2px 8px;'
        f'border-radius:10px;font-size:11px;font-weight:600;{fam}">{text}</span>'
    )


def _fmt_sla_duration(value: float, unit: str) -> str:
    value = max(float(value or 0), 0.0)
    if unit == "h":
        return f"{value:.1f}h" if value < 10 and value % 1 else f"{value:.0f}h"
    if unit == "wd":
        return f"{value:.1f}wd"
    return "—"


def _sla_time_caption(row: pd.Series) -> str:
    unit = row.get("sla_unit", "")
    budget = float(row.get("sla_budget", 0) or 0)
    elapsed = float(row.get("sla_elapsed", 0) or 0)
    if not unit or budget <= 0:
        return ""

    label = row.get("sla_label", "")
    delta = budget - elapsed
    if label == "breached":
        return f"breached by {_fmt_sla_duration(abs(delta), unit)}"
    if label in ("at_risk", "on_track"):
        return f"{_fmt_sla_duration(delta, unit)} remaining"
    if label == "met":
        return f"{_fmt_sla_duration(delta, unit)} headroom"
    return ""


def _render_ticket_table(open_df: pd.DataFrame, jira_domain: str,
                          showing_closed: bool = False) -> None:
    """Main ticket table. For open tickets sorts by sla_pct desc; when
    `showing_closed` is True, the caller pre-sorts (usually by resolved desc)."""
    if open_df.empty:
        st.info("No tickets match the current filter.")
        return

    if showing_closed:
        df = open_df.reset_index(drop=True)
    else:
        df = open_df.sort_values("sla_pct", ascending=False).reset_index(drop=True)

    th = (
        f"padding:10px 14px;font-size:11px;font-weight:600;"
        f"color:{COLOR['text_secondary']};text-transform:uppercase;"
        f"letter-spacing:.07em;text-align:left;border-bottom:1px solid {COLOR['border']};"
        f"white-space:nowrap;background:{COLOR['page_bg']}"
    )
    td_base = (
        f"padding:10px 14px;font-size:13px;color:{COLOR['text_primary']};"
        f"border-bottom:1px solid #f1f5f9;vertical-align:middle"
    )

    rows_html = ""
    for _, r in df.iterrows():
        label = r["sla_label"]
        if label == "breached":
            row_bg = "#FDF5F5"
        elif label == "at_risk":
            row_bg = "#FFFBF2"
        elif label == "met":
            row_bg = "#F3FAF5"
        else:
            row_bg = "#ffffff"

        td = td_base + f";background:{row_bg}"

        pri = r["priority"] or ""
        pri_pill = _pill(pri, *PRIORITY_STYLE[pri]) if pri in PRIORITY_STYLE else _pill("—", COLOR["p3_bg"], COLOR["p3_text"])

        client = r["client"]
        if client == "CLIENT NOT TAGGED":
            client_html = (
                f'<span style="font-style:italic;color:{COLOR["amber_text"]};'
                f'background:{COLOR["amber_bg"]};padding:2px 8px;border-radius:10px;font-size:11px">'
                f'NOT TAGGED</span>'
            )
        else:
            client_html = f'<span style="color:{COLOR["text_primary"]}">{client}</span>'

        type_val = r["type"] or "—"
        type_color = COLOR["text_primary"] if r["type"] else COLOR["text_tertiary"]

        summary = r["summary"] or ""
        summary_short = summary if len(summary) <= 110 else summary[:107] + "…"
        summary_esc = summary.replace('"', "&quot;")

        age_days = r["age_days"]
        if age_days < 1:
            age_str = f"{age_days * 8:.0f}h"
        else:
            age_str = f"{age_days:.0f}d"

        status_bg, status_fg = _status_style(r["status"])
        status_html = (
            f'<span style="background:{status_bg};color:{status_fg};padding:2px 8px;'
            f'border-radius:10px;font-size:11px;font-weight:600;white-space:nowrap">'
            f'{r["status"]}</span>'
        )

        pct = r["sla_pct"]
        if label == "breached":
            sla_bg, sla_fg, sla_text = "#fee2e2", "#b91c1c", f"{pct}% over"
        elif label == "at_risk":
            sla_bg, sla_fg, sla_text = "#fef3c7", "#b45309", f"{pct}% used"
        elif label == "on_track":
            sla_bg, sla_fg, sla_text = "#dcfce7", "#15803d", f"{pct}% used"
        elif label == "met":
            sla_bg, sla_fg, sla_text = "#dcfce7", "#15803d", f"✓ met · {pct}%"
        else:
            sla_bg, sla_fg, sla_text = "#f1f5f9", "#64748b", "—"
        sla_caption = _sla_time_caption(r)
        sla_caption_html = (
            f'<span style="font-size:10px;color:{COLOR["text_secondary"]};white-space:nowrap">'
            f'{sla_caption}</span>'
            if sla_caption else ""
        )
        sla_html = (
            f'<div style="display:flex;flex-direction:column;gap:2px;white-space:nowrap">'
            f'<span style="background:{sla_bg};color:{sla_fg};padding:2px 10px;'
            f'border-radius:10px;font-size:11px;font-weight:700;white-space:nowrap">'
            f'{sla_text}</span>{sla_caption_html}</div>'
        )

        key_link = (
            f'<a href="https://{jira_domain}/browse/{r["key"]}" target="_blank" '
            f'style="color:{COLOR["accent"]};text-decoration:none;font-weight:600;'
            f'font-family:ui-monospace,SFMono-Regular,monospace;font-size:12px">'
            f'{r["key"]}</a>'
        )

        rows_html += (
            f'<tr>'
            f'<td style="{td};white-space:nowrap">{key_link}</td>'
            f'<td style="{td};min-width:320px;max-width:520px;line-height:1.4" '
            f'title="{summary_esc}">{summary_short}</td>'
            f'<td style="{td};white-space:nowrap">{pri_pill}</td>'
            f'<td style="{td};font-size:12px;white-space:nowrap">{client_html}</td>'
            f'<td style="{td};color:{type_color};font-size:12px;white-space:nowrap">{type_val}</td>'
            f'<td style="{td};font-size:12px;white-space:nowrap">{r["developer"]}</td>'
            f'<td style="{td};font-size:12px;white-space:nowrap">{r["assignee"]}</td>'
            f'<td style="{td};white-space:nowrap">{status_html}</td>'
            f'<td style="{td};font-size:12px;color:{COLOR["text_secondary"]};white-space:nowrap">{age_str}</td>'
            f'<td style="{td};white-space:nowrap">{sla_html}</td>'
            f'</tr>'
        )

    table_html = f"""
<div style="background:{COLOR["card_bg"]};border-radius:12px;border:1px solid {COLOR["border"]};
            overflow:hidden;overflow-x:auto;box-shadow:0 1px 4px rgba(0,0,0,.06)">
  <table style="width:100%;border-collapse:collapse">
    <thead>
      <tr>
        <th style="{th}">Key</th>
        <th style="{th};width:100%">Summary</th>
        <th style="{th}">Pri</th>
        <th style="{th}">Client</th>
        <th style="{th}">Type</th>
        <th style="{th}">Developer</th>
        <th style="{th}">Assignee</th>
        <th style="{th}">Status</th>
        <th style="{th}">Age</th>
        <th style="{th}">SLA</th>
      </tr>
    </thead>
    <tbody>{rows_html}</tbody>
  </table>
</div>
"""
    st.markdown(table_html, unsafe_allow_html=True)


def _render_stuck_workflow_callout(open_df: pd.DataFrame, now: datetime) -> None:
    """Callout for tickets with an old resolution date but non-closed status."""
    if open_df.empty:
        return
    cutoff = now - timedelta(days=30)
    stuck = open_df[
        open_df["resolved"].notna()
        & (open_df["resolved"] < cutoff)
    ]
    if stuck.empty:
        return

    keys = ", ".join(stuck["key"].tolist())
    oldest_date = stuck["resolved"].min().strftime("%d %b %Y")
    st.markdown(f"""
<div style="font-size:12px;color:{COLOR["text_secondary"]};font-style:italic;
            margin-top:10px;padding:10px 14px;background:{COLOR["page_bg"]};
            border:1px dashed {COLOR["border"]};border-radius:8px">
  <strong style="font-style:normal;color:{COLOR["text_primary"]}">Stuck in workflow:</strong>
  {keys} have resolution dates from {oldest_date} but are still in non-closed status.
  Workflow hygiene issue.
</div>
""", unsafe_allow_html=True)


def _render_component_hotspots(open_df: pd.DataFrame, top_n: int = 8) -> None:
    """Top components by open-ticket volume. A ticket in multiple components
    counts once per component — intentional, shows spread of pain."""
    empty_card = f"""
<div style="background:{COLOR["card_bg"]};border:1px solid {COLOR["border"]};
            border-radius:12px;padding:18px;box-shadow:0 1px 4px rgba(0,0,0,.06)">
  <div style="color:{COLOR["text_tertiary"]};font-size:13px;font-style:italic">
    No components tagged on open tickets.
  </div>
</div>
"""
    if open_df.empty or "components" not in open_df.columns:
        st.markdown(empty_card, unsafe_allow_html=True)
        return

    exploded = open_df.explode("components")
    exploded = exploded[exploded["components"].notna() & (exploded["components"] != "")]
    if exploded.empty:
        st.markdown(empty_card, unsafe_allow_html=True)
        return

    counts = exploded["components"].value_counts().head(top_n)
    # Breach count per component (ticket counted once per component it's in)
    breach_counts = (
        exploded[exploded["sla_label"] == "breached"]["components"]
        .value_counts()
    )
    max_c = int(counts.max())

    rows = ""
    for comp, n in counts.items():
        n_breach = int(breach_counts.get(comp, 0))
        w = (n / max_c * 100) if max_c else 0
        bar_color = COLOR["red_fill"] if n_breach else COLOR["accent"]
        breach_badge = (
            f'<span style="font-size:10px;color:{COLOR["red_text"]};'
            f'background:{COLOR["red_bg"]};padding:1px 6px;border-radius:8px;'
            f'margin-left:6px;font-weight:600">{n_breach} breached</span>'
            if n_breach else ""
        )
        rows += (
            f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:10px">'
            f'<div style="width:120px;font-size:12px;color:{COLOR["text_primary"]};'
            f'font-weight:500;overflow:hidden;text-overflow:ellipsis;white-space:nowrap"'
            f' title="{comp}">{comp}</div>'
            f'<div style="flex:1;background:#f1f5f9;border-radius:4px;height:14px;min-width:40px">'
            f'<div style="width:{w:.1f}%;background:{bar_color};height:14px;border-radius:4px"></div>'
            f'</div>'
            f'<div style="width:24px;text-align:right;font-size:12px;font-weight:700;'
            f'color:{COLOR["text_primary"]}">{int(n)}</div>'
            f'{breach_badge}'
            f'</div>'
        )

    st.markdown(f"""
<div style="background:{COLOR["card_bg"]};border:1px solid {COLOR["border"]};
            border-radius:12px;padding:14px 18px;box-shadow:0 1px 4px rgba(0,0,0,.06)">
  {rows}
</div>
""", unsafe_allow_html=True)


def _render_breakdowns(open_df: pd.DataFrame) -> None:
    """Three columns: developer breach load, component hotspots, UI/API/APP type."""
    col1, col2, col3 = st.columns(3)

    # Developer breach load
    with col1:
        st.markdown(f'<div style="font-size:13px;font-weight:700;color:{COLOR["text_primary"]};margin-bottom:10px">Breach load by developer</div>', unsafe_allow_html=True)
        if open_df.empty:
            st.markdown(f'<div style="color:{COLOR["text_tertiary"]};font-size:12px">No data</div>', unsafe_allow_html=True)
        else:
            g = (
                open_df.groupby("developer")["sla_label"]
                .value_counts().unstack(fill_value=0)
                .reindex(columns=["breached", "at_risk", "on_track"], fill_value=0)
                .reset_index()
                .sort_values(["breached", "at_risk"], ascending=[False, False])
            )
            rows = ""
            for _, r in g.iterrows():
                if r["breached"] > 0:
                    pill = _pill(f"{int(r['breached'])} breached", COLOR["red_bg"], COLOR["red_text"])
                elif r["at_risk"] > 0:
                    pill = _pill(f"{int(r['at_risk'])} at risk", COLOR["amber_bg"], COLOR["amber_text"])
                else:
                    pill = _pill(f"{int(r['on_track'])} on track", COLOR["green_bg"], COLOR["green_text"])
                rows += (
                    f'<div style="display:flex;justify-content:space-between;'
                    f'align-items:center;padding:8px 0;border-bottom:1px solid {COLOR["border"]}">'
                    f'<span style="font-size:13px;color:{COLOR["text_primary"]}">{r["developer"]}</span>'
                    f'{pill}</div>'
                )
            st.markdown(f"""
<div style="background:{COLOR["card_bg"]};border:1px solid {COLOR["border"]};
            border-radius:12px;padding:14px 18px;box-shadow:0 1px 4px rgba(0,0,0,.06)">
  {rows}
</div>
""", unsafe_allow_html=True)

    # Component hotspots
    with col2:
        st.markdown(f'<div style="font-size:13px;font-weight:700;color:{COLOR["text_primary"]};margin-bottom:10px">Component hotspots</div>', unsafe_allow_html=True)
        _render_component_hotspots(open_df)

    # UI/API/APP type
    with col3:
        st.markdown(f'<div style="font-size:13px;font-weight:700;color:{COLOR["text_primary"]};margin-bottom:10px">By type (UI / API / APP)</div>', unsafe_allow_html=True)
        has_type = open_df["type"].str.strip().ne("").any() if not open_df.empty else False
        if not has_type:
            st.markdown(f"""
<div style="background:{COLOR["card_bg"]};border:1px solid {COLOR["border"]};
            border-radius:12px;padding:18px;box-shadow:0 1px 4px rgba(0,0,0,.06)">
  <div style="color:{COLOR["text_tertiary"]};font-size:14px">— · — · —</div>
  <div style="font-size:11px;color:{COLOR["text_tertiary"]};margin-top:8px;font-style:italic">
    Field empty in all tickets. Will populate once QE backfills.
  </div>
</div>
""", unsafe_allow_html=True)
        else:
            counts = open_df[open_df["type"].str.strip().ne("")]["type"].value_counts()
            max_c = int(counts.max()) if not counts.empty else 0
            rows = ""
            for type_name, n in counts.items():
                w = (n / max_c * 100) if max_c else 0
                rows += (
                    f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:10px">'
                    f'<div style="width:60px;font-size:12px;color:{COLOR["text_primary"]};font-weight:500">{type_name}</div>'
                    f'<div style="flex:1;background:#f1f5f9;border-radius:4px;height:14px">'
                    f'<div style="width:{w:.1f}%;background:{COLOR["accent"]};height:14px;border-radius:4px"></div>'
                    f'</div>'
                    f'<div style="width:26px;text-align:right;font-size:12px;font-weight:700;color:{COLOR["text_primary"]}">{int(n)}</div>'
                    f'</div>'
                )
            st.markdown(f"""
<div style="background:{COLOR["card_bg"]};border:1px solid {COLOR["border"]};
            border-radius:12px;padding:14px 18px;box-shadow:0 1px 4px rgba(0,0,0,.06)">
  {rows}
</div>
""", unsafe_allow_html=True)


def _render_monthly_trend(all_df: pd.DataFrame, now: datetime) -> None:
    """Line chart: bugs raised vs resolved. Last-month view is week-granular; others monthly."""
    today = now.date()
    first_this_month = today.replace(day=1)
    last_month_end = first_this_month - timedelta(days=1)
    last_month_start = last_month_end.replace(day=1)

    hdr_col, toggle_col = st.columns([3, 2])
    with toggle_col:
        st.markdown("<div style='height:22px'></div>", unsafe_allow_html=True)
        view = st.radio(
            "monthly_trend_view",
            ["Last month", "Last 6M", "This year", "Last year"],
            horizontal=True,
            label_visibility="collapsed",
            key="monthly_trend_view",
        )

    if view == "Last month":
        range_start, range_end = last_month_start, last_month_end
        subtitle = f"Weekly — {last_month_start.strftime('%B %Y')}"
        partial_note = ""
        weekly = True
    elif view == "Last 6M":
        y, m = today.year, today.month - 5
        while m <= 0:
            m += 12
            y -= 1
        range_start, range_end = date(y, m, 1), today
        subtitle = "Monthly — last 6 months"
        partial_note = (
            f"* {today.strftime('%B %Y')} is a partial month "
            f"(data up to {today.strftime('%-d %b')})"
        )
        weekly = False
    elif view == "This year":
        range_start, range_end = today.replace(month=1, day=1), today
        subtitle = f"Monthly — {today.year}"
        partial_note = (
            f"* {today.strftime('%B %Y')} is a partial month "
            f"(data up to {today.strftime('%-d %b')})"
        )
        weekly = False
    else:
        range_start, range_end = date(today.year - 1, 1, 1), date(today.year - 1, 12, 31)
        subtitle = f"Monthly — {today.year - 1}"
        partial_note = ""
        weekly = False

    with hdr_col:
        _section_header("Trend", subtitle)

    if all_df.empty:
        st.info("No ticket data available.")
        return

    days_in_last_month = last_month_end.day
    month_abbr = last_month_start.strftime("%b")

    def _week_label(day: int) -> str:
        wk = (day - 1) // 7
        w_start = wk * 7 + 1
        w_end = min((wk + 1) * 7, days_in_last_month)
        return f"{w_start}–{w_end} {month_abbr}"

    def _week_sort(day: int) -> str:
        return str((day - 1) // 7)

    def _build_series(date_col: str, metric_name: str) -> pd.DataFrame:
        sub = all_df[
            all_df[date_col].notna()
            & all_df[date_col].dt.date.between(range_start, range_end)
        ].copy()
        if sub.empty:
            return pd.DataFrame(columns=["x_label", "sort_key", "count", "metric"])
        if weekly:
            sub["x_label"] = sub[date_col].dt.day.apply(_week_label)
            sub["sort_key"] = sub[date_col].dt.day.apply(_week_sort)
        else:
            sub["x_label"] = sub[date_col].dt.strftime("%B %Y")
            sub["sort_key"] = sub[date_col].dt.to_period("M").astype(str)
        grouped = sub.groupby(["x_label", "sort_key"]).size().reset_index(name="count")
        grouped["metric"] = metric_name
        return grouped

    raised_df = _build_series("created", "Raised")
    resolved_df = _build_series("resolved", "Resolved")
    chart_df = pd.concat([raised_df, resolved_df], ignore_index=True)

    if chart_df.empty:
        st.info("No data in the selected range.")
        return

    # Fill zeros so both lines span the same x-axis periods
    all_periods = chart_df[["x_label", "sort_key"]].drop_duplicates()
    full_grid = all_periods.merge(
        pd.DataFrame({"metric": ["Raised", "Resolved"]}), how="cross"
    )
    chart_df = (
        full_grid.merge(chart_df, on=["x_label", "sort_key", "metric"], how="left")
        .fillna({"count": 0})
        .assign(count=lambda d: d["count"].astype(int))
    )
    sort_order = chart_df.sort_values("sort_key")["x_label"].unique().tolist()

    x_enc = alt.X(
        "x_label:O", title=None, sort=sort_order,
        axis=alt.Axis(labelAngle=-30, labelColor=COLOR["text_secondary"],
                      domainColor=COLOR["border"], tickColor=COLOR["border"]),
    )
    y_enc = alt.Y(
        "count:Q", title="Tickets",
        axis=alt.Axis(gridColor=COLOR["border"], labelColor=COLOR["text_secondary"],
                      titleColor=COLOR["text_secondary"], tickMinStep=1),
    )
    color_enc = alt.Color(
        "metric:N",
        scale=alt.Scale(domain=["Raised", "Resolved"],
                        range=[COLOR["amber_fill"], COLOR["green_fill"]]),
        legend=alt.Legend(orient="top-right", title=None,
                          labelColor=COLOR["text_secondary"], labelFontSize=12),
    )
    tooltip_enc = [
        alt.Tooltip("x_label:O", title="Period"),
        alt.Tooltip("metric:N", title=""),
        alt.Tooltip("count:Q", title="Count"),
    ]

    base = alt.Chart(chart_df).encode(x=x_enc, y=y_enc, color=color_enc, tooltip=tooltip_enc)
    line  = base.mark_line(strokeWidth=2.5, interpolate="monotone")
    points = base.mark_point(size=60, filled=True)

    chart = (
        alt.layer(line, points)
        .resolve_scale(color="shared")
        .properties(height=300, background="transparent")
        .configure_view(strokeWidth=0, fill="transparent")
    )

    st.altair_chart(chart, use_container_width=True)

    if partial_note:
        st.markdown(
            f'<div style="font-size:11px;color:{COLOR["text_tertiary"]};'
            f'margin-top:-8px;padding-left:4px">{partial_note}</div>',
            unsafe_allow_html=True,
        )


@st.dialog("Open defects", width="large")
def _show_open_dialog(open_df: pd.DataFrame, jira_domain: str) -> None:
    n = len(open_df)
    if n == 0:
        st.info("No open defects.")
        return
    st.caption(f"{n} open defect{'s' if n != 1 else ''} — sorted by SLA severity.")
    _render_ticket_table(
        open_df.sort_values("sla_pct", ascending=False),
        jira_domain,
        showing_closed=False,
    )


@st.dialog("Reported defects", width="large")
def _show_reported_dialog(reported_df: pd.DataFrame, jira_domain: str) -> None:
    n = len(reported_df)
    if n == 0:
        st.info("No defects reported in this period.")
        return
    st.caption(f"{n} defect{'s' if n != 1 else ''} created in the selected period — sorted newest first.")
    _render_ticket_table(
        reported_df.sort_values("created", ascending=False),
        jira_domain,
        showing_closed=True,
    )


@st.dialog("Closed defects", width="large")
def _show_closed_dialog(closed_df: pd.DataFrame, jira_domain: str) -> None:
    n = len(closed_df)
    if n == 0:
        st.info("No defects resolved in this period.")
        return
    st.caption(f"{n} defect{'s' if n != 1 else ''} resolved in the selected period — sorted newest first.")
    _render_ticket_table(
        closed_df.sort_values("resolved", ascending=False),
        jira_domain,
        showing_closed=True,
    )


@st.dialog("SLA breached", width="large")
def _show_breached_dialog(breached_df: pd.DataFrame, jira_domain: str) -> None:
    n = len(breached_df)
    if n == 0:
        st.info("No SLA breaches in this period.")
        return
    st.caption(f"{n} ticket{'s' if n != 1 else ''} breached SLA — sorted by priority.")
    _render_ticket_table(
        breached_df.sort_values(["priority", "resolved"], ascending=[True, False]),
        jira_domain,
        showing_closed=True,
    )


@st.dialog("SLA met", width="large")
def _show_met_dialog(met_df: pd.DataFrame, jira_domain: str) -> None:
    n = len(met_df)
    if n == 0:
        st.info("No tickets resolved within SLA in this period.")
        return
    st.caption(f"{n} ticket{'s' if n != 1 else ''} resolved within SLA — sorted newest first.")
    _render_ticket_table(
        met_df.sort_values("resolved", ascending=False),
        jira_domain,
        showing_closed=True,
    )


def _render_trend_footer(all_df: pd.DataFrame, start: date, end: date,
                         jira_domain: str) -> None:
    """Compact period-activity strip rendered below the KPI tiles."""
    created_dates = all_df["created"].dt.date
    resolved_dates = all_df["resolved"].dt.date
    in_range_created = (created_dates >= start) & (created_dates <= end)
    in_range_closed = (
        (all_df["status"].str.lower() == "closed")
        & all_df["resolved"].notna()
        & (resolved_dates >= start) & (resolved_dates <= end)
    )

    reported_df = all_df[in_range_created].copy()
    reported = len(reported_df)
    closed = int(in_range_closed.sum())
    net = reported - closed
    open_all = all_df[all_df["status"].str.lower() != "closed"]
    oldest_age = int(open_all["age_days"].max()) if not open_all.empty else 0

    net_color = COLOR["red_fill"] if net > 0 else COLOR["green_fill"]
    net_icon = "▲" if net > 0 else ("▼" if net < 0 else "•")
    net_str = f"{net_icon} {'+' if net > 0 else ''}{net}"
    range_label = f"{start.strftime('%d %b')} – {end.strftime('%d %b %Y')}"

    chip = (
        f"background:{COLOR['page_bg']};border:1px solid {COLOR['border']};"
        f"border-radius:8px;padding:7px 14px;font-size:12px;"
        f"color:{COLOR['text_secondary']};font-weight:500;text-align:center"
    )

    col_rep, col_closed, col_net, col_oldest, col_spacer = st.columns([1, 1, 1, 1, 3])

    with col_rep:
        if st.button(
            f"🆕 Reported   {reported}",
            key="btn_reported_popup",
            use_container_width=True,
            help="Click to view all defects reported in this period",
        ):
            _show_reported_dialog(reported_df, jira_domain)

    with col_closed:
        st.markdown(
            f'<div style="{chip}">✓ Closed &nbsp;<strong style="color:{COLOR["text_primary"]}">{closed}</strong></div>',
            unsafe_allow_html=True,
        )

    with col_net:
        st.markdown(
            f'<div style="{chip}">Net &nbsp;<strong style="color:{net_color}">{net_str}</strong></div>',
            unsafe_allow_html=True,
        )

    with col_oldest:
        st.markdown(
            f'<div style="{chip}">⏱ Oldest &nbsp;<strong style="color:{COLOR["text_primary"]}">{oldest_age}d</strong></div>',
            unsafe_allow_html=True,
        )

    with col_spacer:
        st.markdown(
            f'<div style="display:flex;justify-content:flex-end;align-items:center;height:100%">'
            f'<span style="font-size:11px;color:{COLOR["text_tertiary"]};font-style:italic">'
            f'Period: {range_label}</span></div>',
            unsafe_allow_html=True,
        )


def _render_raw_expander(df: pd.DataFrame, filtered_view: pd.DataFrame) -> None:
    with st.expander("Raw data + extraction audit", expanded=False):
        st.markdown(
            f'<p style="font-size:12px;color:{COLOR["text_secondary"]};margin-bottom:8px">'
            f'Full dataframe with computed SLA + client extraction columns. '
            f'Use <code>extraction_source</code> to audit the client extractor — <code>not_tagged</code> rows are candidates for the slug map.'
            f'</p>',
            unsafe_allow_html=True,
        )
        display_cols = [
            "key", "summary", "priority", "status", "client", "extraction_source",
            "developer", "assignee", "type", "created", "age_days",
            "sla_pct", "sla_label", "sla_elapsed", "sla_budget", "sla_unit",
        ]
        present = [c for c in display_cols if c in df.columns]
        show = df[present].copy()
        if "created" in show.columns:
            show["created"] = pd.to_datetime(show["created"]).dt.strftime("%d-%b-%Y %H:%M")
        for col in ["age_days", "sla_elapsed", "sla_budget"]:
            if col in show.columns:
                show[col] = show[col].round(2)
        st.dataframe(show, use_container_width=True, hide_index=True)

        # CSV export of current filtered view
        if not filtered_view.empty:
            csv_cols = [c for c in display_cols if c in filtered_view.columns]
            csv_df = filtered_view[csv_cols].copy()
            if "created" in csv_df.columns:
                csv_df["created"] = pd.to_datetime(csv_df["created"]).dt.strftime("%Y-%m-%d %H:%M")
            st.download_button(
                "⬇ Download filtered view (CSV)",
                data=csv_df.to_csv(index=False).encode("utf-8"),
                file_name=f"defect_sla_{datetime.now(IST).strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                key="csv_dl",
            )


def _render_share_section(open_df: pd.DataFrame, closed_recent_df: pd.DataFrame,
                          jira_domain: str) -> None:
    """Outlook-ready HTML report + copy-to-clipboard button."""
    HDR_BG = "#1e3a5f"
    td_base = "font-size:13px;border:1px solid #c8d0d8;padding:7px 12px;font-weight:normal;color:#111111"

    def hdr(text: str) -> str:
        return (
            f'<td bgcolor="{HDR_BG}" style="background:{HDR_BG};padding:8px 12px;'
            f'font-size:13px;border:1px solid #c8d0d8;text-align:left">'
            f'<span style="color:#ffffff;font-weight:bold">{text}</span></td>'
        )

    n_open = len(open_df)
    breached_df = open_df[open_df["sla_label"] == "breached"]
    at_risk_df = open_df[open_df["sla_label"] == "at_risk"].sort_values("sla_pct", ascending=False)
    n_breach = len(breached_df)
    n_risk = len(at_risk_df)

    p0b = int((breached_df["priority"] == "P0").sum())
    p1b = int((breached_df["priority"] == "P1").sum())
    p2b = int((breached_df["priority"] == "P2").sum())
    p3b = int((breached_df["priority"] == "P3").sum())

    # Top 5 at-risk rows
    risk_rows_html = ""
    for i, (_, r) in enumerate(at_risk_df.head(5).iterrows()):
        cell = td_base + (";background:#f4f6f8" if i % 2 else "")
        risk_rows_html += (
            f'<tr>'
            f'<td style="{cell}"><a href="https://{jira_domain}/browse/{r["key"]}" style="color:#1e3a5f">{r["key"]}</a></td>'
            f'<td style="{cell}">{r["priority"]}</td>'
            f'<td style="{cell}">{r["client"]}</td>'
            f'<td style="{cell}">{r["developer"]}</td>'
            f'<td style="{cell}">{r["sla_pct"]}% used</td>'
            f'</tr>'
        )
    if not risk_rows_html:
        risk_rows_html = f'<tr><td colspan="5" style="{td_base}">None at risk.</td></tr>'

    # Client breakdown
    client_rows_html = ""
    if not open_df.empty:
        g = (
            open_df.groupby("client")["sla_label"]
            .value_counts().unstack(fill_value=0)
            .reindex(columns=["breached", "at_risk", "on_track"], fill_value=0)
            .reset_index()
            .sort_values(["breached", "at_risk"], ascending=[False, False])
        )
        for i, (_, r) in enumerate(g.iterrows()):
            cell = td_base + (";background:#f4f6f8" if i % 2 else "")
            client_rows_html += (
                f'<tr>'
                f'<td style="{cell}">{r["client"]}</td>'
                f'<td style="{cell}">{int(r["breached"])}</td>'
                f'<td style="{cell}">{int(r["at_risk"])}</td>'
                f'<td style="{cell}">{int(r["on_track"])}</td>'
                f'</tr>'
            )
    if not client_rows_html:
        client_rows_html = f'<tr><td colspan="4" style="{td_base}">No open tickets.</td></tr>'

    report_html = (
        f'<div style="font-family:Calibri,Arial,sans-serif;color:#111111;font-weight:normal">'
        f'<p style="font-size:18px;font-weight:bold;margin-bottom:2px;color:#111111">Client Issues Report</p>'
        f'<p style="font-size:13px;color:#555555;font-weight:normal;margin-top:0;margin-bottom:14px">'
        f'MDLRN · {datetime.now(IST).strftime("%d %b %Y")}</p>'

        f'<p style="font-size:13px;font-weight:bold;margin-bottom:6px;color:#111111">Summary</p>'
        f'<table style="border-collapse:collapse;margin-bottom:18px" cellpadding="0" cellspacing="0">'
        f'<tr>{hdr("Open")}{hdr("At risk")}{hdr("Breached")}{hdr("P0/P1/P2/P3 breach")}{hdr("Closed")}</tr>'
        f'<tr><td style="{td_base}">{n_open}</td>'
        f'<td style="{td_base}">{n_risk}</td>'
        f'<td style="{td_base}">{n_breach}</td>'
        f'<td style="{td_base}">{p0b} / {p1b} / {p2b} / {p3b}</td>'
        f'<td style="{td_base}">{len(closed_recent_df)}</td></tr>'
        f'</table>'

        f'<p style="font-size:13px;font-weight:bold;margin-bottom:6px;color:#111111">Top 5 at-risk tickets</p>'
        f'<table style="border-collapse:collapse;margin-bottom:18px;width:100%" cellpadding="0" cellspacing="0">'
        f'<tr>{hdr("Key")}{hdr("Pri")}{hdr("Client")}{hdr("Developer")}{hdr("SLA")}</tr>'
        f'{risk_rows_html}'
        f'</table>'

        f'<p style="font-size:13px;font-weight:bold;margin-bottom:6px;color:#111111">Per-client breakdown</p>'
        f'<table style="border-collapse:collapse;width:100%" cellpadding="0" cellspacing="0">'
        f'<tr>{hdr("Client")}{hdr("Breached")}{hdr("At risk")}{hdr("On track")}</tr>'
        f'{client_rows_html}'
        f'</table>'

        f'<p style="font-size:11px;color:#999999;font-weight:normal;margin-top:10px">'
        f'Generated {datetime.now(IST).strftime("%d %b %Y %H:%M %Z")}</p>'
        f'</div>'
    )

    html_json = json.dumps(report_html)
    components.html(f"""
<button onclick="copyReport(this)"
  style="background:#4f46e5;color:#fff;border:none;padding:10px 22px;
         border-radius:6px;font-size:14px;cursor:pointer;font-family:sans-serif;
         font-weight:600;letter-spacing:.02em">
  📋 Copy Outlook Report
</button>
<span id="msg" style="margin-left:14px;font-size:13px;font-family:sans-serif"></span>
<script>
function copyReport(btn) {{
  const html = {html_json};
  navigator.clipboard.write([
    new ClipboardItem({{ 'text/html': new Blob([html], {{type: 'text/html'}}) }})
  ]).then(() => {{
    const msg = document.getElementById('msg');
    msg.style.color = '#22c55e';
    msg.textContent = '✓ Copied — paste into Outlook';
    setTimeout(() => msg.textContent = '', 3000);
  }}).catch(() => {{
    const msg = document.getElementById('msg');
    msg.style.color = '#ef4444';
    msg.textContent = 'Copy failed — use the raw data expander instead';
  }});
}}
</script>
""", height=52)


# =====================
# main()
# =====================

def render_defect_sla(inject_base_css: bool = True) -> None:
    """Render defect SLA content.

    Called by the unified dashboard.py (inject_base_css=False — base styles
    already injected) or standalone via main() (inject_base_css=True).
    """
    if inject_base_css:
        _inject_css()
        # Standalone mode: own auto-refresh. Embedded: dashboard.py handles it.
        st_autorefresh(interval=30 * 60 * 1000, key="defect_sla_autorefresh")
    else:
        _inject_sla_css_additions()

    missing_keys, jira_config = validate_jira_config()
    if missing_keys:
        render_missing_config(missing_keys)
        st.stop()

    now = datetime.now(IST)
    _now_min = now.replace(second=0, microsecond=0)
    jira_domain = jira_config["JIRA_DOMAIN"]

    raw_df, fetched_at = load_defect_data()

    if raw_df.empty:
        st.warning(
            "No tickets returned by filter. Verify JIRA_DEFECT_FILTER_ID=20391 in secrets "
            "and that the filter has results."
        )
        st.stop()

    derived_df = build_derived(raw_df, _now_min)

    # Partition by status
    is_closed = derived_df["status"].str.lower() == "closed"
    open_df_all = derived_df[~is_closed].copy()

    # Header + filters — must come before closed_recent_df so we have start/end date
    today = now.date()
    start_date, end_date = _render_header_bar(
        jira_domain, len(open_df_all), fetched_at, today,
    )

    # Closed tickets resolved within the selected date range.
    # Drives: SLA Met, Avg Resolution, Median Resolution, closed count in banner/share.
    closed_recent_df = derived_df[
        is_closed
        & (derived_df["resolved"].dt.date >= start_date)
        & (derived_df["resolved"].dt.date <= end_date)
    ].copy()

    # Open defects = all currently-open tickets regardless of when they were created.
    # Spillovers from prior periods must be counted. The date range only controls
    # the "Period Activity" footer (Reported / Closed counts).
    open_df = open_df_all

    # Reported defects in the selected period — passed to _render_kpi_cards for the dialog.
    reported_df = derived_df[
        derived_df["created"].notna()
        & (derived_df["created"].dt.date >= start_date)
        & (derived_df["created"].dt.date <= end_date)
    ].copy()

    st.markdown(
        f"<div style='height:1px;background:linear-gradient(90deg,{COLOR['accent']},transparent);"
        f"margin:12px 0 16px'></div>",
        unsafe_allow_html=True,
    )

    # Section 1: KPI tiles (with week-over-week delta pills)
    period_label = f"{start_date.strftime('%d %b')} – {end_date.strftime('%d %b %Y')}"
    _section_header(
        "Executive KPIs",
        f"Open defects is a live all-open snapshot. Reported uses created date; SLA breached/met and resolution use resolved date for {period_label}.",
    )
    prev_snap = _kpi_snapshot_at(
        derived_df, start_date, end_date, _now_min - timedelta(days=7),
    )
    _render_kpi_cards(open_df, closed_recent_df, prev_snap, start_date, end_date,
                      derived_df, jira_domain, reported_df)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # Section 2: Priority bars + aging
    left, right = st.columns([2, 1])
    with left:
        _render_priority_bars(open_df)
    with right:
        _render_aging_buckets(open_df)
        _render_median_resolution(closed_recent_df)

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    # Section 3: Main table
    _section_header(
        "Open tickets",
        "Sorted by SLA breach severity (highest % consumed first)",
    )
    table_df = open_df
    _render_ticket_table(table_df, jira_domain)
    _render_stuck_workflow_callout(open_df, now)

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    # Section 4: Monthly trend
    _render_monthly_trend(derived_df, now)

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    # Section 5: Three-column breakdowns
    _section_header("Breakdowns")
    _render_breakdowns(open_df)

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    # Raw data expander
    _render_raw_expander(derived_df, table_df)

    # Share to Outlook
    st.markdown(
        f"<div style='height:1px;background:linear-gradient(90deg,{COLOR['accent']},transparent);"
        f"margin:20px 0'></div>",
        unsafe_allow_html=True,
    )
    _section_header("Share report", "Copy Outlook-ready summary to clipboard")
    _render_share_section(open_df, closed_recent_df, jira_domain)


def main() -> None:
    st.set_page_config(
        page_title="Client Issues Tracker",
        page_icon="🚨",
        layout="wide",
    )
    render_defect_sla()


if __name__ == "__main__":
    main()


# =====================
# v2 candidates (out of scope here)
# =====================
# TODO(v2): Reopen rate / escape rate — needs changelog API calls per ticket.
# TODO(v2): Per-client health card grid — defer until more client volume.
# TODO(v2): Root cause / injected phase Pareto charts — fields not populated yet.
# TODO(v2): Severity field — currently priority-only per spec.
# TODO(v2): Time-in-status heatmap — needs changelog expansion.
# TODO(v2): Public holiday calendar for working_days() — plug in IST holidays
#           (Republic Day, Independence Day, Diwali, etc.) and subtract.
# TODO(v2): Client-facing view — current dashboard is internal only.
