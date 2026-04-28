"""
QA Productivity Dashboard — self-contained Streamlit module.

Rendered inside the main dashboard via:
    from qa_dashboard import render_qa_productivity
    render_qa_productivity(inject_base_css=False)

Scope: QE (tester) productivity measured through Jira changelog status transitions.
Primary data: Jira filter JIRA_QE_FILTER_ID (defaults to 20403).
QE owner field: customfield_11013 ("QE").

Key metric: QA cycle time — hours a bug spends in "Test" status between entering
and leaving to a completion or rework status. Derived from Jira changelog, not
issue fields.
"""

from __future__ import annotations

import html as _html
import re
import unicodedata
from datetime import date, datetime, timedelta
from typing import Any

import altair as alt
import pandas as pd
import pytz
import requests
import streamlit as st
from requests.auth import HTTPBasicAuth

# =====================
# Constants
# =====================

IST = pytz.timezone("Asia/Kolkata")

QA_FILTER_SECRET = "JIRA_QE_FILTER_ID"
QA_FIELD_SECRET = "JIRA_QE_FIELD_ID"
DEV_FIELD_SECRET = "JIRA_DEVELOPER_FIELD_ID"
DEFAULT_QE_FIELD = "customfield_11013"
DEFAULT_DEV_FIELD = "customfield_11012"

QA_TEST_STATUS = "TEST"
QA_COMPLETE_TO = {"ACCEPTED IN QA", "CLOSED"}
QA_REWORK_TO = {"REOPENED", "IN PROGRESS", "OPEN", "IN REVIEW"}

RESOLUTION_VALID = {"DONE"}
RESOLUTION_INVALID = {"DECLINED", "DUPLICATE", "CANNOT REPRODUCE", "WON'T DO", "KNOWN ERROR"}

# Calendar hours per priority
QA_SLA_HOURS: dict[str, float] = {
    "P0": 4,
    "P1": 8,
    "P2": 24,
    "P3": 48,
    "P4": 72,
}

PROD_GREEN_THRESHOLD = 80
PROD_YELLOW_THRESHOLD = 60
QUALITY_GREEN_THRESHOLD = 80
QUALITY_YELLOW_THRESHOLD = 60

PRIORITY_ORDER = ["P0", "P1", "P2", "P3", "P4", "Unknown"]

COLOR = {
    "accent":         "#6366f1",
    "accent_light":   "#c7d2fe",
    "red_fill":       "#A32D2D",
    "red_bg":         "#FCEBEB",
    "red_text":       "#501313",
    "amber_fill":     "#EF9F27",
    "amber_bg":       "#FAEEDA",
    "amber_text":     "#412402",
    "green_fill":     "#97C459",
    "green_bg":       "#EAF3DE",
    "green_text":     "#173404",
    "text_primary":   "#0f172a",
    "text_secondary": "#64748b",
    "text_tertiary":  "#94a3b8",
    "page_bg":        "#f8fafc",
    "card_bg":        "#ffffff",
    "border":         "#e2e8f0",
}


# =====================
# Jira auth helpers
# =====================

def _get_secret(key: str, default: str = "") -> str:
    try:
        v = st.secrets.get(key, default)
    except Exception:
        return default
    return str(v).strip() if v is not None else default


def _auth() -> tuple[str, HTTPBasicAuth, dict]:
    domain = _get_secret("JIRA_DOMAIN")
    email = _get_secret("JIRA_EMAIL")
    token = _get_secret("JIRA_API_TOKEN")
    return domain, HTTPBasicAuth(email, token), {"Accept": "application/json"}


def _normalize_str(v: Any) -> str:
    if pd.isna(v) if not isinstance(v, (dict, list)) else False:
        return ""
    s = unicodedata.normalize("NFKC", str(v))
    return re.sub(r"\s+", " ", s).strip()


def _normalize_person(field: Any) -> str:
    if isinstance(field, dict):
        return field.get("displayName") or field.get("name") or field.get("accountId") or ""
    if isinstance(field, list) and field:
        return ", ".join(
            d.get("displayName", "") if isinstance(d, dict) else str(d)
            for d in field
        )
    return str(field) if field is not None else ""


# =====================
# Jira API — field discovery
# =====================

@st.cache_data(ttl=86400, show_spinner=False)
def _discover_field_id(domain: str, email: str, token: str, field_name_hint: str, default: str) -> str:
    """Return the custom field ID matching field_name_hint (case-insensitive), or default."""
    try:
        resp = requests.get(
            f"https://{domain}/rest/api/3/field",
            auth=HTTPBasicAuth(email, token),
            headers={"Accept": "application/json"},
            timeout=15,
        )
        resp.raise_for_status()
        hint_upper = field_name_hint.upper()
        for f in resp.json():
            if hint_upper in (f.get("name") or "").upper():
                return f["id"]
    except Exception:
        pass
    return default


# =====================
# Jira API — issue search with embedded changelog
# =====================

@st.cache_data(ttl=1800, show_spinner=False)
def _jira_search_with_changelog(domain: str, email: str, token: str, jql: str, fields: str) -> list[dict]:
    """
    Paginated JQL search with expand=changelog.
    Changelog is embedded in each issue under issue["changelog"]["histories"].
    Each history entry has: id, created, author, items[].

    Note: POST /rest/api/3/changelog/bulkfetch was tried first (status-field-only, more
    efficient) but returned empty values[] on this Jira Cloud instance. expand=changelog
    on the search endpoint is heavier (returns full history, not status-only) but is the
    working path here. For ~300-400 bugs with typical <20 status changes each the payload
    size is acceptable. expand=changelog caps at 100 history entries per issue; bugs with
    more than 100 total field changes (extremely rare) would have truncated changelog.
    """
    url = f"https://{domain}/rest/api/3/search/jql"
    auth = HTTPBasicAuth(email, token)
    headers = {"Accept": "application/json"}
    all_issues: list[dict] = []
    seen_ids: set[str] = set()
    seen_tokens: set[str] = set()
    page_token = None

    for _ in range(200):
        params: dict[str, Any] = {
            "jql": jql,
            "fields": fields,
            "expand": "changelog",
            "maxResults": 50,   # smaller page size — each issue now carries changelog payload
        }
        if page_token:
            params["nextPageToken"] = page_token
            params["pageToken"] = page_token

        resp = requests.get(url, headers=headers, auth=auth, params=params, timeout=60)
        resp.raise_for_status()
        payload = resp.json()

        issues = payload.get("issues") or []
        added = 0
        for it in issues:
            iid = str(it.get("id") or "")
            if iid and iid not in seen_ids:
                seen_ids.add(iid)
                all_issues.append(it)
                added += 1

        next_token = payload.get("nextPageToken")
        if added == 0 or not next_token or next_token in seen_tokens or payload.get("isLast"):
            break
        seen_tokens.add(next_token)
        page_token = next_token

    return all_issues


def _extract_changelogs_from_issues(raw_issues: list[dict]) -> dict[str, list]:
    """
    Pull status changelog items out of expand=changelog issue responses.
    Returns dict: issue_id → list of items, each with _created attached.
    """
    result: dict[str, list] = {}
    for issue in raw_issues:
        iid = str(issue.get("id") or "")
        histories = (issue.get("changelog") or {}).get("histories") or []
        items_with_ts = []
        for h in histories:
            ts = h.get("created")
            for item in (h.get("items") or []):
                if item.get("field") == "status" or item.get("fieldId") == "status":
                    item["_created"] = ts
                    items_with_ts.append(item)
        if items_with_ts:
            result[iid] = items_with_ts
    return result


# =====================
# Data loader
# =====================

@st.cache_data(ttl=1800, show_spinner=False)
def load_qe_data(domain: str, email: str, token: str, filter_id: str) -> tuple[pd.DataFrame, dict]:
    """
    Load QE issues from Jira filter and their status changelogs.
    Returns (issues_df, changelogs_dict).
    changelogs_dict maps issue_id (str) → list of status-change items.
    """
    qe_field = _get_secret(QA_FIELD_SECRET) or _discover_field_id(domain, email, token, "QE", DEFAULT_QE_FIELD)
    dev_field = _get_secret(DEV_FIELD_SECRET) or DEFAULT_DEV_FIELD

    fields = (
        f"key,summary,issuetype,status,priority,resolution,created,updated,"
        f"resolutiondate,assignee,components,labels,{dev_field},{qe_field},customfield_10010"
    )
    jql = f"filter={filter_id} ORDER BY created DESC"

    empty_df = pd.DataFrame(columns=[
        "Key", "IssueId", "Summary", "IssueType", "Status", "Priority",
        "Resolution", "ResolutionClass", "Created", "Updated", "ResolvedAt",
        "Assignee", "Components", "QE", "Developer", "StoryPoints",
    ])

    try:
        raw_issues = _jira_search_with_changelog(domain, email, token, jql, fields)
    except Exception as e:
        st.error(f"Failed to load QE issue data: {e}")
        return empty_df, {}

    rows = []
    for issue in raw_issues:
        f = issue.get("fields") or {}
        priority_raw = (f.get("priority") or {}).get("name") or ""
        priority = _parse_priority(priority_raw)

        resolution_name = (f.get("resolution") or {}).get("name") or ""
        res_class = classify_resolution(resolution_name)

        rows.append({
            "Key": issue.get("key") or "",
            "IssueId": str(issue.get("id") or ""),
            "Summary": _normalize_str(f.get("summary") or ""),
            "IssueType": (f.get("issuetype") or {}).get("name") or "",
            "Status": _normalize_str((f.get("status") or {}).get("name") or ""),
            "Priority": priority,
            "Resolution": _normalize_str(resolution_name),
            "ResolutionClass": res_class,
            "Created": f.get("created"),
            "Updated": f.get("updated"),
            "ResolvedAt": f.get("resolutiondate"),
            "Assignee": _normalize_str(_normalize_person(f.get("assignee"))),
            "Components": _first_component(f.get("components")),
            "QE": _normalize_str(_normalize_person(f.get(qe_field))),
            "Developer": _normalize_str(_normalize_person(f.get(dev_field))),
            "StoryPoints": f.get("customfield_10010"),
        })

    if not rows:
        return empty_df, {}

    df = pd.DataFrame(rows)
    df["Created"] = pd.to_datetime(df["Created"], utc=True, errors="coerce")
    df["Updated"] = pd.to_datetime(df["Updated"], utc=True, errors="coerce")
    df["ResolvedAt"] = pd.to_datetime(df["ResolvedAt"], utc=True, errors="coerce")
    df["StoryPoints"] = pd.to_numeric(df["StoryPoints"], errors="coerce").fillna(0)
    df["QE"] = df["QE"].replace("", "(Unassigned)")
    df["Developer"] = df["Developer"].replace("", "(Unassigned)")

    changelogs = _extract_changelogs_from_issues(raw_issues)

    return df, changelogs


# =====================
# Transformation helpers
# =====================

def classify_resolution(name: str) -> str:
    upper = name.strip().upper()
    if upper in RESOLUTION_VALID:
        return "valid"
    if upper in RESOLUTION_INVALID:
        return "invalid"
    return "unresolved"


def _parse_priority(raw: str) -> str:
    """Map Jira priority name to P0–P4 or 'Unknown'."""
    m = re.search(r"P(\d)", raw, re.IGNORECASE)
    if m:
        return f"P{m.group(1)}"
    upper = raw.upper()
    mapping = {
        "BLOCKER": "P0", "CRITICAL": "P0",
        "HIGHEST": "P0", "HIGH": "P1",
        "MEDIUM": "P2", "LOW": "P3",
        "LOWEST": "P4", "TRIVIAL": "P4",
    }
    return mapping.get(upper, "Unknown")


def _first_component(components: Any) -> str:
    if isinstance(components, list) and components:
        return (components[0] or {}).get("name") or ""
    return ""


def extract_status_windows(
    issue_id: str,
    changelog_items: list[dict],
    issue_created: datetime | None,
    current_time: datetime,
) -> list[dict]:
    """
    Build a list of status windows from raw changelog items.
    Each window: {issue_id, from_status, to_status, entered_at, exited_at, duration_hours}.
    The last open window has exited_at=current_time (for aging only, not cycle time).
    """
    # Normalize items — bulk fetch returns items with direct fields;
    # per-issue fallback attaches _created to each item.
    status_items = []
    for item in changelog_items:
        if item.get("field") == "status" or item.get("fieldId") == "status":
            ts_raw = item.get("_created") or item.get("created")
            if ts_raw:
                try:
                    ts = pd.to_datetime(ts_raw, utc=True)
                    status_items.append({
                        "to_status": _normalize_str(item.get("toString") or item.get("to") or ""),
                        "from_status": _normalize_str(item.get("fromString") or item.get("from") or ""),
                        "ts": ts,
                    })
                except Exception:
                    pass

    if not status_items:
        return []

    status_items.sort(key=lambda x: x["ts"])

    windows = []
    for i, item in enumerate(status_items):
        entered_at = item["ts"]
        exited_at = status_items[i + 1]["ts"] if i + 1 < len(status_items) else current_time
        duration_hours = (exited_at - entered_at).total_seconds() / 3600.0
        windows.append({
            "issue_id": issue_id,
            "from_status": item["from_status"],
            "to_status": item["to_status"],
            "entered_at": entered_at,
            "exited_at": exited_at,
            "duration_hours": max(duration_hours, 0.0),
            "is_open": i + 1 >= len(status_items),
        })

    # Prepend the initial window (from issue creation to first transition)
    if issue_created is not None and status_items:
        first_ts = status_items[0]["ts"]
        init_duration = (first_ts - issue_created).total_seconds() / 3600.0
        windows.insert(0, {
            "issue_id": issue_id,
            "from_status": "(initial)",
            "to_status": status_items[0]["from_status"],
            "entered_at": issue_created,
            "exited_at": first_ts,
            "duration_hours": max(init_duration, 0.0),
            "is_open": False,
        })

    return windows


def build_qa_passes(windows: list[dict], issue_key: str, priority: str) -> list[dict]:
    """
    Identify QA passes from status windows.

    A window where to_status == TEST is the time the bug spent IN Test:
      - entered_at  = when it entered Test
      - exited_at   = when it left Test
      - duration_hours = actual QA cycle time for that pass

    The FOLLOWING window (i+1) tells us the exit destination, which
    determines the outcome (completed / rework / in_progress / other).
    """
    passes = []
    pass_num = 0
    sla_hours = QA_SLA_HOURS.get(priority, QA_SLA_HOURS.get("P4", 72))

    for i, w in enumerate(windows):
        if w["to_status"].upper() != QA_TEST_STATUS:
            continue

        pass_num += 1
        duration = w["duration_hours"]
        is_open = w["is_open"]

        if is_open:
            # Still in Test right now
            outcome = "in_progress"
            within_sla = None
            exit_to = ""
        else:
            # Determine where it went after Test
            exit_to = windows[i + 1]["to_status"].upper() if i + 1 < len(windows) else ""
            if exit_to in QA_COMPLETE_TO:
                outcome = "completed"
                within_sla = duration <= sla_hours
            elif exit_to in QA_REWORK_TO:
                outcome = "rework"
                within_sla = None
            else:
                outcome = "other"
                within_sla = None

        passes.append({
            "issue_key": issue_key,
            "pass_num": pass_num,
            "started_at": w["entered_at"],
            "ended_at": w["exited_at"],
            "exit_to_status": exit_to,
            "duration_hours": duration,
            "outcome": outcome,
            "within_sla": within_sla,
            "is_open": is_open,
            "priority": priority,
        })

    return passes


def compute_qe_metrics(
    issues_df: pd.DataFrame,
    all_passes: list[dict],
    start_date: date,
    end_date: date,
    now_utc: datetime,
) -> tuple[dict, pd.DataFrame]:
    """
    Compute per-QE metrics and team-level summary.
    Returns (team_dict, qe_df).

    qe_df display columns: QE | Completed Passes | Avg Passes | QA Productivity % | Rework | QA Quality Score
    qe_df detail columns (for drill-down): Median QA CT (h) | QA SLA % | Validity Rate % | Rework Rate % | In Test | P0/P1 Open
    """
    if issues_df.empty:
        return {}, pd.DataFrame()

    start_dt = pd.Timestamp(start_date, tz="UTC")
    end_dt = pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=1)

    period_issues = issues_df[
        (issues_df["Created"] >= start_dt) & (issues_df["Created"] < end_dt)
    ].copy()

    passes_df = pd.DataFrame(all_passes) if all_passes else pd.DataFrame(
        columns=["issue_key", "pass_num", "started_at", "ended_at", "duration_hours",
                 "outcome", "within_sla", "is_open", "priority"]
    )
    if not passes_df.empty and "started_at" in passes_df.columns:
        passes_df["started_at"] = pd.to_datetime(passes_df["started_at"], utc=True, errors="coerce")
        passes_df["ended_at"] = pd.to_datetime(passes_df["ended_at"], utc=True, errors="coerce")
        period_passes = passes_df[
            (passes_df["started_at"] >= start_dt) & (passes_df["started_at"] < end_dt)
        ].copy()
    else:
        period_passes = passes_df.copy()

    key_to_qe = issues_df.set_index("Key")["QE"].to_dict()
    if not period_passes.empty:
        period_passes["QE"] = period_passes["issue_key"].map(key_to_qe).fillna("(Unassigned)")

    in_test_issues = issues_df[issues_df["Status"].str.upper() == QA_TEST_STATUS]

    all_qes = sorted(set(period_issues["QE"].unique()) | set(
        period_passes["QE"].unique() if not period_passes.empty else []
    ))

    passes_by_key: dict[str, list[dict]] = {}
    for p in all_passes:
        passes_by_key.setdefault(p["issue_key"], []).append(p)

    qe_rows = []
    completed_pass_counts: list[int] = []

    for qe in all_qes:
        qi = period_issues[period_issues["QE"] == qe]
        qp = period_passes[period_passes["QE"] == qe] if not period_passes.empty else pd.DataFrame()
        in_test = in_test_issues[in_test_issues["QE"] == qe]

        resolved = qi[qi["ResolvedAt"].notna()]
        resolved_count = len(resolved)
        valid_bugs = len(qi[qi["ResolutionClass"] == "valid"])
        validity_rate = (valid_bugs / resolved_count * 100) if resolved_count > 0 else None
        in_test_count = len(in_test)

        if not qp.empty:
            completed = qp[qp["outcome"] == "completed"]
            rework = qp[qp["outcome"] == "rework"]
            completed_passes = len(completed)
            rework_count = len(rework)
            total_passes = len(qp[qp["outcome"].isin(["completed", "rework"])])
            rework_rate = (rework_count / total_passes * 100) if total_passes > 0 else 0.0

            durations = completed["duration_hours"].dropna()
            median_ct = float(durations.median()) if len(durations) > 0 else None

            sla_eligible = completed[completed["within_sla"].notna()]
            sla_pct = (
                sla_eligible["within_sla"].sum() / len(sla_eligible) * 100
                if len(sla_eligible) > 0 else None
            )
        else:
            completed_passes = rework_count = 0
            rework_rate = 0.0
            median_ct = sla_pct = None

        p0p1_open = len(qi[
            qi["Priority"].isin(["P0", "P1"]) &
            ~qi["Status"].str.upper().isin({"ACCEPTED IN QA", "CLOSED"})
        ])

        completed_pass_counts.append(completed_passes)
        qe_rows.append({
            "QE": qe,
            "Completed Passes": completed_passes,
            # Avg Passes / QA Productivity % / QA Quality Score filled after team avg is known
            "Rework": rework_count,
            "Median QA CT (h)": round(median_ct, 1) if median_ct is not None else None,
            "QA SLA %": round(sla_pct, 1) if sla_pct is not None else None,
            "Validity Rate %": round(validity_rate, 1) if validity_rate is not None else None,
            "Rework Rate %": round(rework_rate, 1),
            "In Test": in_test_count,
            "P0/P1 Open": p0p1_open,
        })

    if not qe_rows:
        return {}, pd.DataFrame()

    team_avg_passes = float(pd.Series(completed_pass_counts).mean()) if completed_pass_counts else 0.0

    for row in qe_rows:
        row["Avg Passes"] = round(team_avg_passes, 1)
        row["QA Productivity %"] = _qa_productivity_pct(row["Completed Passes"], team_avg_passes)
        row["QA Quality Score"] = _qa_quality_score(row["QA SLA %"], row["Validity Rate %"])

    qe_df = pd.DataFrame(qe_rows)

    team = {
        "Active QEs": len(qe_df),
        "Avg Passes / QE": round(team_avg_passes, 1),
        "Completed Passes": int(qe_df["Completed Passes"].sum()),
        "QA Productivity %": round(float(qe_df["QA Productivity %"].mean()), 1),
        "Rework": int(qe_df["Rework"].sum()),
        "QA Quality Score": int(round(float(qe_df["QA Quality Score"].mean()))),
    }

    return team, qe_df


def _qa_productivity_pct(completed_passes: int, team_avg_passes: float) -> float:
    if team_avg_passes <= 0:
        return 0.0
    return round(min(completed_passes / team_avg_passes, 1.5) / 1.5 * 100, 1)


def _qa_quality_score(sla_pct: float | None, validity_rate: float | None) -> int:
    sla = float(sla_pct) if sla_pct is not None else 50.0
    val = float(validity_rate) if validity_rate is not None else 50.0
    return int(max(0, min(100, round(0.6 * sla + 0.4 * val))))


def _fmt_delta(diff, unit: str, invert: bool):
    """Return (display_str, is_positive_visually) for a numeric delta."""
    if diff is None or abs(diff) < 0.05:
        return "no change", None
    is_pos_visually = (diff > 0) if not invert else (diff < 0)
    arrow = "↑" if diff > 0 else "↓"
    sign = "+" if diff > 0 else ""
    if isinstance(diff, float) and not diff.is_integer():
        body = f"{sign}{diff:.1f}{unit}"
    else:
        body = f"{sign}{int(diff)}{unit}"
    return f"{body} {arrow}", is_pos_visually


# =====================
# Period utilities (mirrors dashboard.py)
# =====================

def _get_period_bounds(period_type: str, today: date) -> tuple[date, date, date, date]:
    if period_type == "Current Week":
        start = today - timedelta(days=today.weekday())
        end = min(start + timedelta(days=6), today)
    elif period_type == "Current Month":
        start = today.replace(day=1)
        end = today
    elif period_type == "Current Quarter":
        q = (today.month - 1) // 3
        start = today.replace(month=q * 3 + 1, day=1)
        end = today
    elif period_type == "Current Year":
        start = today.replace(month=1, day=1)
        end = today
    elif period_type == "Last Month":
        first_this = today.replace(day=1)
        end = first_this - timedelta(days=1)
        start = end.replace(day=1)
    elif period_type == "Last Quarter":
        q = (today.month - 1) // 3
        if q == 0:
            start = today.replace(year=today.year - 1, month=10, day=1)
            end = today.replace(year=today.year - 1, month=12, day=31)
        else:
            start = today.replace(month=(q - 1) * 3 + 1, day=1)
            last_month = today.replace(month=q * 3, day=1)
            end = (last_month + timedelta(days=32)).replace(day=1) - timedelta(days=1)
    else:
        start = today - timedelta(weeks=4)
        end = today

    delta = end - start
    prev_end = start - timedelta(days=1)
    prev_start = prev_end - delta
    return start, min(end, today), prev_start, prev_end


# =====================
# CSS injection
# =====================

_BASE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
* { font-family: 'Inter', sans-serif; }
[data-testid="stAppViewContainer"] { background: #f8fafc; }
[data-testid="stHeader"] { display: none; }
[data-testid="stToolbar"] { display: none; }
footer { display: none; }
.block-container { padding: 1.5rem 2rem 3rem; max-width: 1400px; }
</style>
"""

_QA_CSS = """
<style>
.qa-kpi-card {
    background: #fff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 16px 18px;
    min-height: 90px;
}
.qa-kpi-label {
    font-size: 11px;
    font-weight: 600;
    color: #64748b;
    letter-spacing: .06em;
    text-transform: uppercase;
    margin-bottom: 6px;
}
.qa-kpi-value {
    font-size: 26px;
    font-weight: 700;
    color: #0f172a;
    line-height: 1.1;
}
.qa-kpi-sub {
    font-size: 11px;
    color: #94a3b8;
    margin-top: 4px;
}
.qa-score-green  { color: #16a34a; font-weight: 700; }
.qa-score-yellow { color: #d97706; font-weight: 700; }
.qa-score-red    { color: #dc2626; font-weight: 700; }
.qa-section-header {
    font-size: 15px;
    font-weight: 700;
    color: #0f172a;
    margin-bottom: 2px;
}
.qa-section-sub {
    font-size: 12px;
    color: #64748b;
    margin-bottom: 12px;
}
</style>
"""


def _inject_css(base: bool = True) -> None:
    if base:
        st.markdown(_BASE_CSS, unsafe_allow_html=True)
    st.markdown(_QA_CSS, unsafe_allow_html=True)


# =====================
# Rendering helpers
# =====================

def _divider() -> None:
    st.markdown(
        "<div style='height:1px;background:linear-gradient(90deg,#6366f1,transparent);margin:20px 0'></div>",
        unsafe_allow_html=True,
    )


def _section(title: str, subtitle: str = "") -> None:
    st.markdown(f"<div class='qa-section-header'>{title}</div>", unsafe_allow_html=True)
    if subtitle:
        st.markdown(f"<div class='qa-section-sub'>{subtitle}</div>", unsafe_allow_html=True)


def _kpi_card(label: str, value: str, sub: str = "") -> str:
    return (
        f"<div class='qa-kpi-card'>"
        f"<div class='qa-kpi-label'>{label}</div>"
        f"<div class='qa-kpi-value'>{value}</div>"
        f"{'<div class=qa-kpi-sub>' + sub + '</div>' if sub else ''}"
        f"</div>"
    )


def _fmt_hours(h: float | None) -> str:
    if h is None:
        return "—"
    if h < 1:
        return f"{h * 60:.0f}m"
    if h < 24:
        return f"{h:.1f}h"
    return f"{h / 24:.1f}d"


def _score_badge(score: float | None) -> str:
    if score is None:
        return "—"
    cls = "qa-score-green" if score >= 70 else ("qa-score-yellow" if score >= 50 else "qa-score-red")
    return f"<span class='{cls}'>{score:.0f}</span>"


def _render_kpi_row(curr: dict, prev: dict | None = None) -> None:
    metrics_cfg = [
        ("Active QEs",        "",  False, "#8b5cf6", "QE"),
        ("Avg Passes / QE",   "",  False, "#475569", "AVG"),
        ("Completed Passes",  "",  False, "#6366f1", "QA"),
        ("QA Productivity %", "%", False, "#06b6d4", "%"),
        ("Rework",            "",  True,  "#f43f5e", "RW"),
        ("QA Quality Score",  "",  False, "#10b981", "QS"),
    ]
    cols = st.columns(len(metrics_cfg))
    for col, (key, unit, invert, accent, icon) in zip(cols, metrics_cfg):
        val = curr.get(key, 0) or 0
        prev_val = prev.get(key, 0) if prev else None

        if prev_val is not None:
            dstr, is_pos = _fmt_delta(val - prev_val, unit, invert)
        else:
            dstr, is_pos = "—", None

        delta_color = "#16a34a" if is_pos is True else ("#dc2626" if is_pos is False else "#64748b")
        delta_bg    = "#dcfce7" if is_pos is True else ("#fee2e2" if is_pos is False else "#f1f5f9")

        if "%" in key:
            display_val = f"{float(val):.1f}%"
        elif key == "Avg Passes / QE":
            display_val = f"{float(val):.1f}"
        else:
            display_val = str(int(round(float(val))))

        bottom_html = (
            f'<div style="display:inline-block;background:{delta_bg};border-radius:20px;'
            f'padding:3px 10px;font-size:11px;color:{delta_color};font-weight:600">'
            f'{dstr} vs prev period</div>'
        )

        with col:
            st.markdown(f"""
<div style="background:#ffffff;border-radius:12px;padding:18px 16px;
            border:1px solid #e2e8f0;
            box-shadow:0 2px 8px rgba(0,0,0,.07);
            display:flex;align-items:flex-start;gap:14px">
  <div style="width:44px;height:44px;background:{accent}18;border-radius:10px;
              display:flex;align-items:center;justify-content:center;
              font-size:13px;font-weight:800;color:{accent};flex-shrink:0">{icon}</div>
  <div style="flex:1;min-width:0">
    <div style="font-size:11px;color:#64748b;letter-spacing:.07em;
                text-transform:uppercase;font-weight:600;margin-bottom:4px">{key}</div>
    <div style="font-size:26px;font-weight:700;color:#0f172a;
                line-height:1;margin-bottom:8px;letter-spacing:-.02em">{display_val}</div>
    {bottom_html}
  </div>
</div>""", unsafe_allow_html=True)


def _render_qe_table(qe_df: pd.DataFrame) -> None:
    if qe_df.empty:
        st.info("No QE data for the selected period.")
        return

    filtered = qe_df.sort_values("QA Productivity %", ascending=False).reset_index(drop=True)

    def _badge(p):
        if p >= PROD_GREEN_THRESHOLD:
            return '<span style="background:#dcfce7;color:#16a34a;padding:3px 10px;border-radius:20px;font-size:11px;font-weight:600;white-space:nowrap">● High</span>'
        elif p >= PROD_YELLOW_THRESHOLD:
            return '<span style="background:#fef3c7;color:#d97706;padding:3px 10px;border-radius:20px;font-size:11px;font-weight:600;white-space:nowrap">● Mid</span>'
        return '<span style="background:#fee2e2;color:#dc2626;padding:3px 10px;border-radius:20px;font-size:11px;font-weight:600;white-space:nowrap">● Low</span>'

    def _progress(p):
        w = min(float(p), 100)
        clr = "#16a34a" if p >= PROD_GREEN_THRESHOLD else ("#d97706" if p >= PROD_YELLOW_THRESHOLD else "#dc2626")
        return (
            f'<div style="display:flex;align-items:center;gap:8px;min-width:130px">'
            f'<div style="flex:1;background:#e2e8f0;border-radius:4px;height:5px">'
            f'<div style="width:{w:.0f}%;background:{clr};height:5px;border-radius:4px"></div>'
            f'</div>'
            f'<span style="font-size:12px;color:#0f172a;min-width:40px;text-align:right;font-weight:600">{p:.1f}%</span>'
            f'</div>'
        )

    th = "padding:10px 16px;font-size:11px;font-weight:600;color:#64748b;text-transform:uppercase;letter-spacing:.07em;text-align:left;border-bottom:1px solid #e2e8f0;white-space:nowrap"
    td = "padding:12px 16px;font-size:13px;color:#334155;border-bottom:1px solid #f1f5f9;vertical-align:middle"

    rows = ""
    for i, (_, r) in enumerate(filtered.iterrows()):
        bg = "#f8fafc" if i % 2 == 0 else "#ffffff"
        rw_clr = "#dc2626" if r["Rework"] > 0 else "#94a3b8"
        q = r["QA Quality Score"]
        q_clr = "#16a34a" if q >= QUALITY_GREEN_THRESHOLD else ("#d97706" if q >= QUALITY_YELLOW_THRESHOLD else "#dc2626")
        rows += (
            f'<tr style="background:{bg}" '
            f'onmouseover="this.style.background=\'#eff6ff\'" '
            f'onmouseout="this.style.background=\'{bg}\'">'
            f'<td style="{td}">{_badge(r["QA Productivity %"])}</td>'
            f'<td style="{td};font-weight:600;color:#0f172a">{_html.escape(str(r["QE"]))}</td>'
            f'<td style="{td};text-align:center;font-size:13px;color:#64748b">{float(r["Avg Passes"]):.1f}</td>'
            f'<td style="{td};text-align:center;font-weight:700;font-size:15px;color:#4f46e5">{int(r["Completed Passes"])}</td>'
            f'<td style="{td}">{_progress(r["QA Productivity %"])}</td>'
            f'<td style="{td};text-align:center;font-weight:700;color:{rw_clr}">{int(r["Rework"])}</td>'
            f'<td style="{td};text-align:center">'
            f'<span style="font-weight:700;font-size:14px;color:{q_clr}">{q}</span>'
            f'<span style="font-size:11px;color:#94a3b8">/100</span>'
            f'</td>'
            f'</tr>'
        )

    st.markdown(f"""
<div style="background:#ffffff;border-radius:12px;border:1px solid #e2e8f0;
            overflow:hidden;overflow-x:auto;box-shadow:0 1px 4px rgba(0,0,0,.06)">
  <table style="width:100%;border-collapse:collapse">
    <thead>
      <tr style="background:#f8fafc">
        <th style="{th}">Status</th>
        <th style="{th}">QE</th>
        <th style="{th};text-align:center">Avg Passes</th>
        <th style="{th};text-align:center">Completed Passes</th>
        <th style="{th}">QA Productivity</th>
        <th style="{th};text-align:center">Rework</th>
        <th style="{th};text-align:center">QA Quality Score</th>
      </tr>
    </thead>
    <tbody>{rows}</tbody>
  </table>
</div>""", unsafe_allow_html=True)


def _render_charts(qe_df: pd.DataFrame, all_passes: list[dict], issues_df: pd.DataFrame) -> None:
    if qe_df.empty:
        st.info("No pass data available for charts.")
        return

    col1, col2 = st.columns(2)

    with col1:
        _section("Completed QA Passes by Week")
        if all_passes:
            passes_df = pd.DataFrame(all_passes)
            passes_df["started_at"] = pd.to_datetime(passes_df["started_at"], utc=True, errors="coerce")
            key_to_qe = issues_df.set_index("Key")["QE"].to_dict()
            passes_df["QE"] = passes_df["issue_key"].map(key_to_qe).fillna("(Unassigned)")
            completed = passes_df[passes_df["outcome"] == "completed"].copy()
            if not completed.empty:
                completed["week"] = completed["started_at"].dt.to_period("W").dt.start_time.dt.tz_localize(None)
                weekly = completed.groupby(["week", "QE"]).size().reset_index(name="Passes")
                chart = alt.Chart(weekly).mark_bar().encode(
                    x=alt.X("week:T", title="Week", axis=alt.Axis(format="%d %b")),
                    y=alt.Y("Passes:Q", title="QA Passes"),
                    color=alt.Color("QE:N"),
                    tooltip=["week:T", "QE:N", "Passes:Q"],
                ).properties(height=220)
                st.altair_chart(chart, use_container_width=True)
            else:
                st.caption("No completed passes in period.")
        else:
            st.caption("No pass data available.")

    with col2:
        _section("QA Quality Score by QE")
        qs_data = qe_df[["QE", "QA Quality Score"]].copy()
        if not qs_data.empty:
            qs_data = qs_data.sort_values("QA Quality Score", ascending=False)
            qs_data["color"] = qs_data["QA Quality Score"].apply(
                lambda x: "#97C459" if x >= QUALITY_GREEN_THRESHOLD else (
                    "#EF9F27" if x >= QUALITY_YELLOW_THRESHOLD else "#A32D2D"
                )
            )
            chart = alt.Chart(qs_data).mark_bar().encode(
                x=alt.X("QA Quality Score:Q", scale=alt.Scale(domain=[0, 100]), title="Quality Score"),
                y=alt.Y("QE:N", sort="-x", title=""),
                color=alt.Color("color:N", scale=None),
                tooltip=["QE:N", "QA Quality Score:Q"],
            ).properties(height=max(180, len(qs_data) * 30))
            st.altair_chart(chart, use_container_width=True)
        else:
            st.caption("No quality data available.")


def _render_drilldown(
    selected_qe: str,
    issues_df: pd.DataFrame,
    all_passes: list[dict],
    changelogs: dict,
    now_utc: datetime,
    curr_qe_df: pd.DataFrame,
    prev_qe_df: pd.DataFrame | None = None,
) -> None:
    qi = issues_df[issues_df["QE"] == selected_qe]
    key_to_qe = issues_df.set_index("Key")["QE"].to_dict()

    if all_passes:
        passes_df = pd.DataFrame(all_passes)
        passes_df["started_at"] = pd.to_datetime(passes_df["started_at"], utc=True, errors="coerce")
        passes_df["ended_at"] = pd.to_datetime(passes_df["ended_at"], utc=True, errors="coerce")
        passes_df["QE"] = passes_df["issue_key"].map(key_to_qe).fillna("(Unassigned)")
        qp = passes_df[passes_df["QE"] == selected_qe].copy()
    else:
        passes_df = pd.DataFrame()
        qp = pd.DataFrame()

    curr_row = curr_qe_df[curr_qe_df["QE"] == selected_qe] if not curr_qe_df.empty else pd.DataFrame()
    prev_row = (
        prev_qe_df[prev_qe_df["QE"] == selected_qe]
        if (prev_qe_df is not None and not prev_qe_df.empty)
        else pd.DataFrame()
    )

    # ─── 4 KPI cards with deltas ───────────────────────────────────────
    metrics_cfg = [
        ("Completed Passes",  "",  False, "#6366f1", "QA"),
        ("QA Productivity %", "%", False, "#06b6d4", "%"),
        ("QA Quality Score",  "",  False, "#10b981", "Q"),
        ("Rework",            "",  True,  "#f43f5e", "RW"),
    ]
    cols = st.columns(4)
    for col, (key, unit, invert, accent, icon) in zip(cols, metrics_cfg):
        val = float(curr_row.iloc[0][key]) if not curr_row.empty else 0.0
        prev_val = float(prev_row.iloc[0][key]) if not prev_row.empty else None

        if prev_val is not None:
            dstr, is_pos = _fmt_delta(val - prev_val, unit, invert)
        else:
            dstr, is_pos = "—", None

        delta_color = "#16a34a" if is_pos is True else ("#dc2626" if is_pos is False else "#64748b")
        delta_bg = "#dcfce7" if is_pos is True else ("#fee2e2" if is_pos is False else "#f1f5f9")
        display_val = f"{val:.1f}%" if "%" in key else str(int(round(val)))

        with col:
            st.markdown(f"""
<div style="background:#ffffff;border-radius:12px;padding:16px 14px;
            border:1px solid #e2e8f0;border-top:3px solid {accent};position:relative;
            box-shadow:0 1px 4px rgba(0,0,0,.06)">
  <div style="position:absolute;top:12px;right:12px;width:26px;height:26px;
              background:{accent}18;border-radius:6px;display:flex;align-items:center;
              justify-content:center;font-size:11px;font-weight:700;color:{accent}">{icon}</div>
  <div style="font-size:11px;color:#64748b;letter-spacing:.07em;text-transform:uppercase;
              font-weight:600;margin-bottom:8px">{key}</div>
  <div style="font-size:26px;font-weight:700;color:#0f172a;line-height:1;
              margin-bottom:8px;letter-spacing:-.01em">{display_val}</div>
  <div style="display:inline-block;background:{delta_bg};border-radius:20px;
              padding:2px 8px;font-size:11px;color:{delta_color};font-weight:600">
    {dstr} vs prev
  </div>
</div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ─── Granularity + 3-line trend chart ─────────────────────────────
    gran = st.selectbox("Granularity", ["Week", "Month", "Quarter"], index=1, key="_qa_drilldown_gran")

    def _period_start(ts: pd.Timestamp, g: str) -> pd.Timestamp:
        if g == "Week":
            return ts.to_period("W").start_time
        elif g == "Month":
            return ts.to_period("M").start_time
        return ts.to_period("Q").start_time

    def _fmt_period(ts: pd.Timestamp, g: str) -> str:
        try:
            ts = pd.Timestamp(ts)
            if g == "Week":
                return ts.strftime("W%V-%Y")
            elif g == "Month":
                return ts.strftime("%b-%Y").upper()
            return f"Q{((ts.month-1)//3)+1}-{ts.year}"
        except Exception:
            return str(ts)

    if not qp.empty:
        completed_qp = qp[qp["outcome"] == "completed"].copy()
        rework_qp = qp[qp["outcome"] == "rework"].copy()

        if not completed_qp.empty:
            completed_qp["period"] = completed_qp["started_at"].apply(lambda ts: _period_start(ts, gran))
            cp_by_period = completed_qp.groupby("period").size().reset_index(name="Completed QA Passes")
        else:
            cp_by_period = pd.DataFrame(columns=["period", "Completed QA Passes"])

        if not rework_qp.empty:
            rework_qp["period"] = rework_qp["started_at"].apply(lambda ts: _period_start(ts, gran))
            rw_by_period = rework_qp.groupby("period").size().reset_index(name="Rework Events")
        else:
            rw_by_period = pd.DataFrame(columns=["period", "Rework Events"])

        if not passes_df.empty:
            all_comp = passes_df[passes_df["outcome"] == "completed"].copy()
            if not all_comp.empty:
                all_comp["period"] = all_comp["started_at"].apply(lambda ts: _period_start(ts, gran))
                n_qes = max(passes_df["QE"].nunique(), 1)
                team_by_period = all_comp.groupby("period").size().reset_index(name="_total")
                team_by_period["Team Avg Passes"] = team_by_period["_total"] / n_qes
                team_by_period = team_by_period.drop(columns=["_total"])
            else:
                team_by_period = pd.DataFrame(columns=["period", "Team Avg Passes"])
        else:
            team_by_period = pd.DataFrame(columns=["period", "Team Avg Passes"])

        combined = (
            cp_by_period
            .merge(team_by_period, on="period", how="outer")
            .merge(rw_by_period, on="period", how="outer")
            .fillna(0)
            .sort_values("period")
        )
        combined["Period Label"] = combined["period"].apply(lambda ts: _fmt_period(ts, gran))
        hist_order = combined["Period Label"].tolist()

        base = alt.Chart(combined).encode(x=alt.X("Period Label:N", title=gran, sort=hist_order))
        line_cp = base.mark_line(point=True, color="#6366f1", strokeWidth=2).encode(
            y=alt.Y("Completed QA Passes:Q", title="Completed Passes",
                    axis=alt.Axis(titleColor="#6366f1")),
            tooltip=["Period Label:N", "Completed QA Passes:Q"],
        )
        line_avg = base.mark_line(color="#334155", strokeWidth=2, strokeDash=[6, 4]).encode(
            y=alt.Y("Team Avg Passes:Q", title="Team Avg"),
            tooltip=["Period Label:N", "Team Avg Passes:Q"],
        )
        line_rw = base.mark_line(point=True, color="#f97316", strokeWidth=2, strokeDash=[4, 2]).encode(
            y=alt.Y("Rework Events:Q", title="Rework Events",
                    axis=alt.Axis(titleColor="#f97316")),
            tooltip=["Period Label:N", "Rework Events:Q"],
        )
        trend = (
            alt.layer(alt.layer(line_cp, line_avg), line_rw)
            .resolve_scale(y="independent")
            .properties(height=240)
        )
        st.altair_chart(trend, use_container_width=True)
    else:
        st.info(f"No pass data found for {selected_qe}.")

    # ─── Two-column layout: pass audit + open in-test ──────────────────
    left_col, right_col = st.columns(2)

    with left_col:
        _section("QA Pass Audit", "Every Test entry — use to investigate cycle-time outliers")
        if not qp.empty and "outcome" in qp.columns:
            audit = qp[qp["outcome"].isin(["completed", "rework", "in_progress"])].copy()
            if not audit.empty:
                audit["Entered Test"] = audit["started_at"].dt.tz_convert(IST).dt.strftime("%Y-%m-%d %H:%M")
                audit["Exited Test"] = audit["ended_at"].dt.tz_convert(IST).dt.strftime("%Y-%m-%d %H:%M")
                audit["Duration"] = audit["duration_hours"].apply(_fmt_hours)
                audit["SLA"] = audit.apply(
                    lambda r: "✓" if r.get("within_sla") is True
                    else ("✗" if r.get("within_sla") is False else "—"),
                    axis=1,
                )
                show = audit[["issue_key", "priority", "Entered Test", "Exited Test", "Duration", "SLA", "outcome"]].copy()
                show.columns = ["Issue", "Priority", "Entered Test", "Exited Test", "Duration", "SLA", "Outcome"]
                st.dataframe(show, use_container_width=True, hide_index=True)
            else:
                st.caption("No passes found.")
        else:
            st.caption("No pass data available.")

    with right_col:
        _section("Open In-Test Issues")
        in_test_issues = qi[qi["Status"].str.upper() == QA_TEST_STATUS].copy()
        if not in_test_issues.empty:
            key_to_age: dict[str, float] = {}
            for p in all_passes:
                if p.get("outcome") == "in_progress" and p["issue_key"] in set(in_test_issues["Key"]):
                    key_to_age[p["issue_key"]] = round(p["duration_hours"], 1)
            in_test_issues["Age in Test"] = in_test_issues["Key"].map(key_to_age)
            in_test_issues["SLA Budget (h)"] = in_test_issues["Priority"].map(QA_SLA_HOURS)
            in_test_issues["SLA Status"] = in_test_issues.apply(
                lambda r: "Breached"
                if pd.notna(r["Age in Test"]) and r["Age in Test"] > (r["SLA Budget (h)"] or 999)
                else "On Track",
                axis=1,
            )
            show = in_test_issues[["Key", "Summary", "Priority", "Age in Test", "SLA Status"]].copy()
            st.dataframe(show, use_container_width=True, hide_index=True)
        else:
            st.caption("No issues currently in Test for this QE.")

    with st.expander("Raw Jira Issues for this QE"):
        show_raw = qi[["Key", "Summary", "Status", "Priority", "Resolution", "Created", "QE", "Developer"]].copy()
        show_raw["Created"] = show_raw["Created"].dt.tz_convert(IST).dt.strftime("%Y-%m-%d %H:%M")
        st.dataframe(show_raw, use_container_width=True, hide_index=True)


# =====================
# Main entry point
# =====================

def render_qa_productivity(inject_base_css: bool = True) -> None:
    _inject_css(base=inject_base_css)

    qa_filter = _get_secret(QA_FILTER_SECRET, "20403")
    if not qa_filter:
        st.warning(
            "QA Productivity is not configured. "
            "Add `JIRA_QE_FILTER_ID` to your Streamlit secrets to enable this view."
        )
        st.code("JIRA_QE_FILTER_ID = '20403'", language="toml")
        return

    domain = _get_secret("JIRA_DOMAIN")
    email = _get_secret("JIRA_EMAIL")
    token = _get_secret("JIRA_API_TOKEN")

    if not all([domain, email, token]):
        st.error("Missing JIRA_DOMAIN, JIRA_EMAIL, or JIRA_API_TOKEN in secrets.")
        return

    # ── Header ─────────────────────────────────────────────────────────
    hcol, rcol = st.columns([6, 1])
    with hcol:
        st.markdown(
            "<div style='font-size:22px;font-weight:700;color:#0f172a;margin-bottom:2px'>"
            "🧪 QA Productivity</div>"
            "<div style='font-size:12px;color:#64748b;margin-bottom:16px'>"
            "QE throughput, cycle time, and quality metrics from Jira changelog</div>",
            unsafe_allow_html=True,
        )
    with rcol:
        if st.button("↺ Refresh", key="_qa_refresh"):
            st.cache_data.clear()
            st.rerun()

    # ── Period & filters ────────────────────────────────────────────────
    today = date.today()
    period_types = [
        "Current Week", "Current Month", "Current Quarter",
        "Current Year", "Last Month", "Last Quarter", "Custom",
    ]

    f1, f2 = st.columns([3, 7])
    with f1:
        selected_period = st.selectbox(
            "Period", period_types, index=4, key="_qa_period", label_visibility="collapsed"
        )
    if selected_period == "Custom":
        c1, c2, _ = st.columns([2, 2, 4])
        with c1:
            custom_start = st.date_input("Start", value=today - timedelta(weeks=4), key="_qa_start")
        with c2:
            custom_end = st.date_input("End", value=today, key="_qa_end")
        start_date, end_date = custom_start, min(custom_end, today)
        delta = end_date - start_date
        prev_end = start_date - timedelta(days=1)
        prev_start = prev_end - delta
    else:
        start_date, end_date, prev_start, prev_end = _get_period_bounds(selected_period, today)

    # ── Load data ───────────────────────────────────────────────────────
    with st.spinner("Loading QE data…"):
        try:
            issues_df, changelogs = load_qe_data(domain, email, token, qa_filter)
        except Exception as e:
            st.error(f"Failed to load QA data: {e}")
            return

    if issues_df.empty:
        st.warning("No issues found for this filter. Check `JIRA_QE_FILTER_ID` and Jira access.")
        return

    changelog_available = bool(changelogs)
    if not changelog_available:
        st.warning(
            "Jira changelog is unavailable. QA cycle-time and SLA metrics are hidden. "
            "Showing count and resolution metrics only."
        )

    # ── Secondary filters ───────────────────────────────────────────────
    all_qes = sorted(q for q in issues_df["QE"].unique() if q != "(Unassigned)")
    all_priorities = [p for p in PRIORITY_ORDER if p in issues_df["Priority"].unique()]
    all_components = sorted(c for c in issues_df["Components"].unique() if c)
    all_resolutions = sorted(r for r in issues_df["Resolution"].unique() if r)
    all_statuses = sorted(issues_df["Status"].unique())

    fc1, fc2, fc3 = st.columns([3, 2, 2])
    with fc1:
        sel_qes = st.multiselect("QE", all_qes, key="_qa_sel_qe", placeholder="All QEs")
    with fc2:
        sel_priorities = st.multiselect("Priority", all_priorities, key="_qa_sel_pri", placeholder="All priorities")
    with fc3:
        sel_statuses = st.multiselect("Status", all_statuses, key="_qa_sel_status", placeholder="All statuses")

    fc4, fc5, _ = st.columns([3, 3, 3])
    with fc4:
        sel_components = st.multiselect("Component", all_components, key="_qa_sel_comp", placeholder="All components")
    with fc5:
        sel_resolutions = st.multiselect("Resolution", all_resolutions, key="_qa_sel_res", placeholder="All resolutions")

    # Apply filters
    fdf = issues_df.copy()
    if sel_qes:
        fdf = fdf[fdf["QE"].isin(sel_qes)]
    if sel_priorities:
        fdf = fdf[fdf["Priority"].isin(sel_priorities)]
    if sel_statuses:
        fdf = fdf[fdf["Status"].isin(sel_statuses)]
    if sel_components:
        fdf = fdf[fdf["Components"].isin(sel_components)]
    if sel_resolutions:
        fdf = fdf[fdf["Resolution"].isin(sel_resolutions)]

    # ── Build QA passes from changelogs ─────────────────────────────────
    now_utc = datetime.now(tz=pytz.UTC)
    all_passes: list[dict] = []
    for _, row in fdf.iterrows():
        iid = row["IssueId"]
        cl_items = changelogs.get(iid, [])
        windows = extract_status_windows(iid, cl_items, row["Created"], now_utc)
        passes = build_qa_passes(windows, row["Key"], row["Priority"])
        all_passes.extend(passes)

    # ── Compute metrics ─────────────────────────────────────────────────
    team, qe_df = compute_qe_metrics(fdf, all_passes, start_date, end_date, now_utc)
    prev_team, prev_qe_df = compute_qe_metrics(fdf, all_passes, prev_start, prev_end, now_utc)

    if not team:
        st.info("No QE activity found for the selected period and filters.")
        return

    # ── KPI cards ───────────────────────────────────────────────────────
    _render_kpi_row(team, prev_team if prev_team else None)
    _divider()

    # ── QE Breakdown Table ──────────────────────────────────────────────
    _section("QE Breakdown", f"{start_date.strftime('%d %b')} – {end_date.strftime('%d %b %Y')}")
    _render_qe_table(qe_df)
    _divider()

    # ── Charts ──────────────────────────────────────────────────────────
    _section("Trend Charts")
    _render_charts(qe_df, all_passes, fdf)
    _divider()

    # ── QE Drill-Down ───────────────────────────────────────────────────
    _section("QE Drill-Down", "Select a QE to see their individual issue timeline and aging breakdown")
    qe_options = sorted(fdf["QE"].unique())
    if qe_options:
        selected_qe = st.selectbox("Select QE", qe_options, key="_qa_drilldown_sel", label_visibility="collapsed")
        _render_drilldown(selected_qe, fdf, all_passes, changelogs, now_utc,
                          qe_df, prev_qe_df if prev_team else None)
    _divider()

    # ── Raw Data / Export ───────────────────────────────────────────────
    _section("Raw Data", "All issues in the current filter view")
    show_raw = fdf[["Key", "Summary", "Status", "Priority", "Resolution", "Created", "QE", "Developer", "Components"]].copy()
    show_raw["Created"] = show_raw["Created"].dt.tz_convert(IST).dt.strftime("%Y-%m-%d %H:%M")
    st.dataframe(show_raw, use_container_width=True, hide_index=True)
    csv_bytes = show_raw.to_csv(index=False).encode()
    st.download_button(
        "⬇ Download CSV",
        data=csv_bytes,
        file_name=f"qa_issues_{start_date}_{end_date}.csv",
        mime="text/csv",
        key="_qa_csv_export",
    )
