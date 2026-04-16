import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import requests
from datetime import datetime, timedelta, date
import altair as alt
import pytz
import unicodedata
import re
from requests.auth import HTTPBasicAuth

# =====================
# Constants
# =====================
COMPLETED_STATUSES = ["ACCEPTED IN QA", "CLOSED"]
SP_BASELINE_PER_DAY = 1          # 5 SP/week baseline
PROD_GREEN_THRESHOLD = 80        # productivity % → green
PROD_YELLOW_THRESHOLD = 60       # productivity % → yellow, else red

# =====================
# Data loading
# =====================

@st.cache_data(ttl=14400, show_spinner=False)
def _jira_search_all(jira_domain: str, email: str, token: str, jql: str, fields: str):
    url = f"https://{jira_domain}/rest/api/3/search/jql"
    auth = HTTPBasicAuth(email, token)
    headers = {"Accept": "application/json"}

    all_issues, seen_ids, seen_tokens = [], set(), set()
    page_token, page_num = None, 0
    MAX_PAGES = 200

    while True:
        page_num += 1
        if page_num > MAX_PAGES:
            break

        params = {"jql": jql, "fields": fields, "maxResults": 100}
        if page_token:
            params["nextPageToken"] = page_token
            params["pageToken"] = page_token

        resp = requests.get(url, headers=headers, auth=auth, params=params)
        resp.raise_for_status()
        payload = resp.json()

        issues = payload.get("issues", []) or []
        added = 0
        for it in issues:
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


def _normalize_str(v):
    if pd.isna(v):
        return ""
    s = unicodedata.normalize("NFKC", str(v))
    return re.sub(r"\s+", " ", s).strip()


def _normalize_developer(dev_field):
    if isinstance(dev_field, dict):
        return dev_field.get("displayName") or dev_field.get("name") or dev_field.get("accountId") or ""
    elif isinstance(dev_field, list) and len(dev_field) > 0:
        return ", ".join(
            [d.get("displayName", "") if isinstance(d, dict) else str(d) for d in dev_field]
        )
    elif dev_field is not None:
        return str(dev_field)
    return ""


@st.cache_data(ttl=14400, show_spinner=False)
def load_jira_data():
    jira_domain = st.secrets["JIRA_DOMAIN"]
    email = st.secrets["JIRA_EMAIL"]
    token = st.secrets["JIRA_API_TOKEN"]
    jql = f"filter={st.secrets['JIRA_FILTER_ID']}"
    fields = "key,summary,status,customfield_10988,customfield_10010,customfield_11012,created"

    try:
        all_issues = _jira_search_all(jira_domain, email, token, jql, fields)
        data = []
        for issue in all_issues:
            f = issue.get("fields", {}) or {}
            data.append({
                "Key": issue.get("key"),
                "Summary": f.get("summary", "") or "",
                "Status": (f.get("status") or {}).get("name", "") or "",
                "Due Date": f.get("customfield_10988"),
                "Story Points": f.get("customfield_10010", 0),
                "Developer": _normalize_developer(f.get("customfield_11012")),
                "Created": f.get("created"),
            })

        df = pd.DataFrame(data)
        for col in ["Key", "Summary", "Status", "Developer"]:
            df[col] = df[col].apply(_normalize_str)

        df["Due Date"] = pd.to_datetime(df["Due Date"], errors="coerce")
        df["Created"] = pd.to_datetime(df["Created"], errors="coerce")
        df["Story Points"] = pd.to_numeric(df["Story Points"], errors="coerce").fillna(0)
        df["Is Completed"] = df["Status"].str.upper().isin([s.upper() for s in COMPLETED_STATUSES])
        df["Developer"] = df["Developer"].replace("", "(Unassigned)")

        # Period columns for trend charts
        df["Week"] = df["Due Date"].dt.to_period("W").dt.start_time
        df["Month"] = df["Due Date"].dt.to_period("M").dt.start_time
        df["Quarter"] = df["Due Date"].dt.to_period("Q").dt.start_time

        return df
    except Exception as e:
        st.error(f"Failed to fetch Jira task data: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=14400, show_spinner=False)
def load_bug_data():
    jira_domain = st.secrets["JIRA_DOMAIN"]
    email = st.secrets["JIRA_EMAIL"]
    token = st.secrets["JIRA_API_TOKEN"]
    bug_filter_id = st.secrets.get("JIRA_BUG_FILTER_ID", "18484")
    jql = f"filter={bug_filter_id}"
    fields = "key,summary,created,customfield_11012"

    try:
        all_issues = _jira_search_all(jira_domain, email, token, jql, fields)
        rows = []
        for issue in all_issues:
            f = issue.get("fields", {}) or {}
            rows.append({
                "Key": issue.get("key"),
                "Summary": f.get("summary", "") or "",
                "Created": f.get("created"),
                "Developer": _normalize_developer(f.get("customfield_11012")),
            })

        df = pd.DataFrame(rows)
        df["Created"] = pd.to_datetime(df["Created"], errors="coerce")
        df["Developer"] = df["Developer"].apply(_normalize_str).replace("", "(Unassigned)")
        df["Week"] = df["Created"].dt.to_period("W").dt.start_time
        df["Month"] = df["Created"].dt.to_period("M").dt.start_time
        df["Quarter"] = df["Created"].dt.to_period("Q").dt.start_time
        return df
    except Exception as e:
        st.error(f"Failed to fetch bug data: {e}")
        return pd.DataFrame()


# =====================
# Period utilities
# =====================

def count_working_days(start_dt, end_dt):
    return sum(1 for d in pd.date_range(start=start_dt, end=end_dt) if d.weekday() < 5)


def get_period_bounds(period_type: str, today: date):
    """Returns (start, end, prev_start, prev_end) as date objects."""
    if period_type == "This Week":
        start = today - timedelta(days=today.weekday())
        end = start + timedelta(days=6)
        prev_start = start - timedelta(weeks=1)
        prev_end = prev_start + timedelta(days=6)
    elif period_type == "This Month":
        start = today.replace(day=1)
        if today.month == 12:
            end = date(today.year + 1, 1, 1) - timedelta(days=1)
        else:
            end = date(today.year, today.month + 1, 1) - timedelta(days=1)
        prev_end = start - timedelta(days=1)
        prev_start = prev_end.replace(day=1)
    elif period_type == "Last Month":
        end = today.replace(day=1) - timedelta(days=1)
        start = end.replace(day=1)
        prev_end = start - timedelta(days=1)
        prev_start = prev_end.replace(day=1)
    elif period_type == "This Quarter":
        q_month = 3 * ((today.month - 1) // 3) + 1
        start = date(today.year, q_month, 1)
        next_q_month = q_month + 3
        if next_q_month > 12:
            end = date(today.year + 1, 1, 1) - timedelta(days=1)
        else:
            end = date(today.year, next_q_month, 1) - timedelta(days=1)
        prev_end = start - timedelta(days=1)
        prev_q_month = 3 * ((prev_end.month - 1) // 3) + 1
        prev_start = date(prev_end.year, prev_q_month, 1)
    elif period_type == "Last Quarter":
        cur_q_month = 3 * ((today.month - 1) // 3) + 1
        end = date(today.year, cur_q_month, 1) - timedelta(days=1)
        lq_month = 3 * ((end.month - 1) // 3) + 1
        start = date(end.year, lq_month, 1)
        prev_end = start - timedelta(days=1)
        prev_q_month = 3 * ((prev_end.month - 1) // 3) + 1
        prev_start = date(prev_end.year, prev_q_month, 1)
    else:
        start = today - timedelta(weeks=4)
        end = today
        prev_end = start - timedelta(days=1)
        prev_start = prev_end - timedelta(weeks=4)
    return start, min(end, today), prev_start, min(prev_end, today)


# =====================
# Metrics computation
# =====================

def compute_metrics(df: pd.DataFrame, bugs_df: pd.DataFrame, start_date: date, end_date: date):
    """
    Returns (team_metrics dict, dev_metrics DataFrame) for the given date range.

    Productivity % = Completed SP / (working_days × SP_BASELINE_PER_DAY × n_devs)
    Quality Score  = 100 − (bug_density × 200), clipped to [0, 100]
    """
    in_period = df[
        (df["Due Date"].dt.date >= start_date) &
        (df["Due Date"].dt.date <= end_date)
    ]
    completed = in_period[in_period["Is Completed"]]

    empty_bugs = pd.DataFrame(columns=["Developer"])
    if bugs_df is not None and not bugs_df.empty:
        bugs_in_period = bugs_df[
            (bugs_df["Created"].dt.date >= start_date) &
            (bugs_df["Created"].dt.date <= end_date)
        ]
    else:
        bugs_in_period = empty_bugs

    all_devs = sorted(
        set(in_period["Developer"].dropna().unique()) |
        set(bugs_in_period["Developer"].dropna().unique())
    )

    working_days = count_working_days(start_date, end_date)
    expected_sp_per_dev = max(working_days * SP_BASELINE_PER_DAY, 1)

    dev_rows = []
    for dev in all_devs:
        dev_completed = completed[completed["Developer"] == dev]
        dev_bugs = bugs_in_period[bugs_in_period["Developer"] == dev] if not bugs_in_period.empty else empty_bugs

        completed_sp = dev_completed["Story Points"].sum()
        bug_count = len(dev_bugs)

        productivity_pct = round(completed_sp / expected_sp_per_dev * 100, 1)
        if completed_sp > 0:
            bug_density = bug_count / completed_sp
            quality_score = int(max(0, min(100, round(100 - bug_density * 200))))
        elif bug_count > 0:
            quality_score = 0   # bugs with no output → worst score
        else:
            quality_score = 100  # no bugs, no output → neutral

        dev_rows.append({
            "Developer": dev,
            "Completed SP": int(completed_sp),
            "Productivity %": productivity_pct,
            "Bugs": bug_count,
            "Quality Score": quality_score,
        })

    dev_df = pd.DataFrame(dev_rows) if dev_rows else pd.DataFrame(
        columns=["Developer", "Completed SP", "Productivity %", "Bugs", "Quality Score"]
    )

    n_devs = max(len(all_devs), 1)
    team_completed_sp = int(completed["Story Points"].sum())
    team_expected_sp = expected_sp_per_dev * n_devs
    team_productivity = round(team_completed_sp / team_expected_sp * 100, 1)
    total_bugs = len(bugs_in_period)
    if team_completed_sp > 0:
        team_bug_density = total_bugs / team_completed_sp
        team_quality = int(max(0, min(100, round(100 - team_bug_density * 200))))
    elif total_bugs > 0:
        team_quality = 0
    else:
        team_quality = 100

    team_metrics = {
        "Completed SP": team_completed_sp,
        "Productivity %": team_productivity,
        "Bugs": total_bugs,
        "Quality Score": team_quality,
    }

    return team_metrics, dev_df


# =====================
# Global CSS + UI helpers
# =====================

def _inject_css():
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }

/* App background */
.stApp { background-color: #f8fafc !important; }
.main .block-container { padding: 1.5rem 2rem 3rem !important; max-width: 100% !important; }

/* Hide Streamlit chrome — use display:none so they take no space */
header[data-testid="stHeader"],
#MainMenu, footer,
[data-testid="stToolbar"],
[data-testid="stStatusWidget"] { display: none !important; }
/* Remove the top padding Streamlit adds to clear its header */
.stMainBlockContainer { padding-top: 0 !important; }

/* Headings */
h1 { font-size: 22px !important; font-weight: 700 !important; color: #0f172a !important; }
h2, h3 { color: #0f172a !important; }

/* Primary buttons */
.stButton > button {
    background: #4f46e5 !important; color: #fff !important;
    border: none !important; border-radius: 6px !important;
    font-weight: 600 !important; font-size: 13px !important;
    padding: 0.4rem 1.1rem !important; transition: background .2s !important;
}
.stButton > button:hover { background: #4338ca !important; }

/* Selectbox */
[data-testid="stSelectbox"] > div > div {
    background: #ffffff !important; border: 1px solid #e2e8f0 !important;
    border-radius: 6px !important; color: #0f172a !important;
}

/* Expanders */
[data-testid="stExpander"] {
    background: #ffffff !important; border: 1px solid #e2e8f0 !important;
    border-radius: 10px !important; overflow: hidden !important;
    box-shadow: 0 1px 3px rgba(0,0,0,.06) !important;
}
[data-testid="stExpander"] summary {
    color: #0f172a !important; font-weight: 600 !important; font-size: 14px !important;
    padding: 12px 16px !important;
}
[data-testid="stExpander"] summary:hover { background: #f1f5f9 !important; }

/* Dividers */
hr { border-color: #e2e8f0 !important; margin: 1.5rem 0 !important; }

/* Captions */
[data-testid="stCaptionContainer"], .stCaption { color: #64748b !important; }

/* Info / warning boxes */
[data-testid="stAlert"] {
    background: #ffffff !important; border-radius: 8px !important;
    border-left-color: #6366f1 !important;
}

/* Multiselect */
[data-testid="stMultiSelect"] > div {
    background: #ffffff !important; border: 1px solid #e2e8f0 !important;
    border-radius: 6px !important;
}

/* Date input */
[data-testid="stDateInput"] input {
    background: #ffffff !important; border: 1px solid #e2e8f0 !important;
    color: #0f172a !important; border-radius: 6px !important;
}

/* Spinner */
.stSpinner > div { border-top-color: #6366f1 !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #f1f5f9; }
::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #94a3b8; }

</style>
""", unsafe_allow_html=True)


def _section_header(title: str, subtitle: str = ""):
    sub = f'<p style="font-size:13px;color:#64748b;margin:4px 0 0 15px;font-weight:400">{subtitle}</p>' if subtitle else ""
    st.markdown(f"""
<div style="margin:8px 0 16px">
  <div style="display:flex;align-items:center;gap:10px">
    <div style="width:3px;height:22px;background:#6366f1;border-radius:2px;flex-shrink:0"></div>
    <span style="font-size:17px;font-weight:700;color:#0f172a;letter-spacing:-.01em">{title}</span>
  </div>
  {sub}
</div>""", unsafe_allow_html=True)


# =====================
# KPI cards
# =====================

def _fmt_delta(diff, unit, invert):
    """Format a numeric delta with arrow. Returns (display_str, is_positive_visually)."""
    if abs(diff) < 0.05:
        return "no change", None
    is_pos_visually = (diff > 0) if not invert else (diff < 0)
    arrow = "↑" if diff > 0 else "↓"
    sign = "+" if diff > 0 else ""
    if isinstance(diff, float) and not diff.is_integer():
        body = f"{sign}{diff:.1f}{unit}"
    else:
        body = f"{sign}{int(diff)}{unit}"
    return f"{body} {arrow}", is_pos_visually


def render_kpi_cards(curr: dict, prev: dict):
    metrics_cfg = [
        ("Completed SP", "",  False, "#6366f1", "SP"),
        ("Productivity %",  "%", False, "#06b6d4", "%"),
        ("Bugs",            "",  True,  "#f43f5e", "🐛"),
        ("Quality Score",   "",  False, "#10b981", "Q"),
    ]

    cols = st.columns(4)
    for col, (key, unit, invert, accent, icon) in zip(cols, metrics_cfg):
        val = curr.get(key, 0)
        prev_val = prev.get(key, 0) if prev else None

        if prev_val is not None:
            dstr, is_pos = _fmt_delta(val - prev_val, unit, invert)
        else:
            dstr, is_pos = "—", None

        delta_color = "#16a34a" if is_pos is True else ("#dc2626" if is_pos is False else "#64748b")
        delta_bg    = "#dcfce7" if is_pos is True else ("#fee2e2" if is_pos is False else "#f1f5f9")
        display_val = f"{val:.1f}%" if "%" in key else str(val)

        with col:
            st.markdown(f"""
<div style="background:#ffffff;border-radius:12px;padding:20px 18px;
            border:1px solid #e2e8f0;border-top:3px solid {accent};
            position:relative;overflow:hidden;
            box-shadow:0 1px 4px rgba(0,0,0,.06)">
  <div style="position:absolute;top:14px;right:14px;width:32px;height:32px;
              background:{accent}18;border-radius:8px;display:flex;
              align-items:center;justify-content:center;
              font-size:13px;font-weight:700;color:{accent}">{icon}</div>
  <div style="font-size:11px;color:#64748b;letter-spacing:.07em;
              text-transform:uppercase;font-weight:600;margin-bottom:10px">{key}</div>
  <div style="font-size:32px;font-weight:700;color:#0f172a;
              line-height:1;margin-bottom:10px;letter-spacing:-.02em">{display_val}</div>
  <div style="display:inline-block;background:{delta_bg};border-radius:20px;
              padding:3px 10px;font-size:11px;color:{delta_color};font-weight:600">
    {dstr} vs prev period
  </div>
</div>""", unsafe_allow_html=True)


# =====================
# Developer breakdown table
# =====================

def render_dev_table(curr_dev: pd.DataFrame, prev_dev: pd.DataFrame):
    if curr_dev.empty:
        st.info("No data for the selected period.")
        return

    df = curr_dev.copy().sort_values("Productivity %", ascending=False).reset_index(drop=True)

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
    for i, (_, r) in enumerate(df.iterrows()):
        bg = "#f8fafc" if i % 2 == 0 else "#ffffff"
        bug_clr = "#dc2626" if r["Bugs"] > 0 else "#94a3b8"
        q = r["Quality Score"]
        q_clr = "#16a34a" if q >= 80 else ("#d97706" if q >= 60 else "#dc2626")
        rows += (
            f'<tr style="background:{bg}" '
            f'onmouseover="this.style.background=\'#eff6ff\'" '
            f'onmouseout="this.style.background=\'{bg}\'">'
            f'<td style="{td}">{_badge(r["Productivity %"])}</td>'
            f'<td style="{td};font-weight:600;color:#0f172a">{r["Developer"]}</td>'
            f'<td style="{td};text-align:center;font-weight:700;font-size:15px;color:#4f46e5">{int(r["Completed SP"])}</td>'
            f'<td style="{td}">{_progress(r["Productivity %"])}</td>'
            f'<td style="{td};text-align:center;font-weight:700;color:{bug_clr}">{r["Bugs"]}</td>'
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
        <th style="{th}">Developer</th>
        <th style="{th};text-align:center">Completed SP</th>
        <th style="{th}">Productivity</th>
        <th style="{th};text-align:center">Bugs</th>
        <th style="{th};text-align:center">Quality Score</th>
      </tr>
    </thead>
    <tbody>{rows}</tbody>
  </table>
</div>""", unsafe_allow_html=True)


# =====================
# Developer drill-down
# =====================

def render_dev_drilldown(
    df: pd.DataFrame,
    bugs_df: pd.DataFrame,
    curr_dev: pd.DataFrame,
    prev_dev: pd.DataFrame,
    start_date: date,
    end_date: date,
):
    """Single-developer deep dive: KPI cards, historical SP chart, task + bug tables."""
    devs = sorted(df["Developer"].dropna().unique())
    if not devs:
        return

    selected = st.selectbox("Select developer", devs, key="drilldown_dev")

    row_curr = curr_dev[curr_dev["Developer"] == selected]
    row_prev = (
        prev_dev[prev_dev["Developer"] == selected]
        if (prev_dev is not None and not prev_dev.empty)
        else pd.DataFrame()
    )

    # --- KPI cards for this developer ---
    metrics_cfg = [
        ("Completed SP",   "",  False, "#6366f1", "SP"),
        ("Productivity %", "%", False, "#06b6d4", "%"),
        ("Bugs",           "",  True,  "#f43f5e", "🐛"),
        ("Quality Score",  "",  False, "#10b981", "Q"),
    ]
    cols = st.columns(4)
    for col, (key, unit, invert, accent, icon) in zip(cols, metrics_cfg):
        val = row_curr.iloc[0][key] if not row_curr.empty else 0
        prev_val = row_prev.iloc[0][key] if not row_prev.empty else None

        if prev_val is not None:
            dstr, is_pos = _fmt_delta(val - prev_val, unit, invert)
        else:
            dstr, is_pos = "—", None

        delta_color = "#16a34a" if is_pos is True else ("#dc2626" if is_pos is False else "#64748b")
        delta_bg    = "#dcfce7" if is_pos is True else ("#fee2e2" if is_pos is False else "#f1f5f9")
        display_val = f"{val:.1f}%" if "%" in key else str(val)

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

    # --- Historical SP + Bug chart ---
    gran = st.selectbox("Granularity", ["Week", "Month", "Quarter"], index=1, key="drilldown_gran")
    dev_completed = df[(df["Developer"] == selected) & (df["Is Completed"])].copy()

    st.markdown(f"**{selected} — Completed SP & Bugs over time**")

    if not dev_completed.empty:
        sp_hist = dev_completed.groupby(gran)["Story Points"].sum().reset_index().sort_values(gran)
        sp_hist["Period Label"] = sp_hist[gran].apply(lambda x: _fmt_period_label(x, gran))
        hist_order = sp_hist["Period Label"].tolist()

        # Merge in bug counts for this developer
        if bugs_df is not None and not bugs_df.empty:
            bug_hist = (
                bugs_df[bugs_df["Developer"] == selected]
                .groupby(gran).size().reset_index(name="Bugs")
            )
            combined = sp_hist.merge(bug_hist, on=gran, how="left").fillna({"Bugs": 0})
        else:
            combined = sp_hist.copy()
            combined["Bugs"] = 0
        combined["Bugs"] = combined["Bugs"].astype(int)

        base = alt.Chart(combined).encode(
            x=alt.X("Period Label:N", title=gran, sort=hist_order)
        )
        sp_line = base.mark_line(point=True, color="#6366f1", strokeWidth=2).encode(
            y=alt.Y("Story Points:Q", title="Completed SP",
                    axis=alt.Axis(titleColor="#6366f1")),
            tooltip=["Period Label", "Story Points"],
        )
        bug_line = base.mark_line(point=True, color="#ef4444", strokeWidth=2, strokeDash=[4, 2]).encode(
            y=alt.Y("Bugs:Q", title="Bugs",
                    axis=alt.Axis(titleColor="#ef4444")),
            tooltip=["Period Label", "Bugs"],
        )
        hist_chart = (
            alt.layer(sp_line, bug_line)
            .resolve_scale(y="independent")
            .properties(height=240)
        )
        st.altair_chart(hist_chart, use_container_width=True)
    else:
        st.info(f"No completed tasks found for {selected}.")

    # --- Task and bug tables for the selected period ---
    period_str = f"{start_date.strftime('%d %b')} – {end_date.strftime('%d %b %Y')}"
    dev_tasks = df[
        (df["Developer"] == selected) &
        (df["Due Date"].dt.date >= start_date) &
        (df["Due Date"].dt.date <= end_date)
    ].copy()

    t_col, b_col = st.columns(2)

    with t_col:
        st.markdown(f"**Tasks ({period_str})**")
        if not dev_tasks.empty:
            disp = dev_tasks[["Key", "Summary", "Status", "Story Points", "Due Date"]].copy()
            disp["Due Date"] = disp["Due Date"].dt.strftime("%d-%b-%Y").str.upper()
            disp["Story Points"] = disp["Story Points"].astype(int)
            st.dataframe(disp, use_container_width=True, hide_index=True)
        else:
            st.info("No tasks in this period.")

    with b_col:
        st.markdown(f"**Bugs ({period_str})**")
        if bugs_df is not None and not bugs_df.empty:
            dev_bugs = bugs_df[
                (bugs_df["Developer"] == selected) &
                (bugs_df["Created"].dt.date >= start_date) &
                (bugs_df["Created"].dt.date <= end_date)
            ].copy()
            if not dev_bugs.empty:
                disp_b = dev_bugs[["Key", "Summary", "Created"]].copy()
                disp_b["Created"] = disp_b["Created"].dt.strftime("%d-%b-%Y").str.upper()
                st.dataframe(disp_b, use_container_width=True, hide_index=True)
            else:
                st.info("No bugs in this period.")
        else:
            st.info("No bug data available.")


# =====================
# Trend charts
# =====================

def _fmt_period_label(ts, gran: str) -> str:
    try:
        if gran == "Week":
            return ts.strftime("W%V-%Y")
        elif gran == "Month":
            return ts.strftime("%b-%Y").upper()
        else:
            return f"Q{((ts.month - 1)//3)+1}-{ts.year}"
    except Exception:
        return str(ts)


# =====================
# Export / Share
# =====================

def render_share_section(period_label: str, team: dict, dev_df: pd.DataFrame):
    """
    One-click copy button (copies formatted HTML to clipboard) + inline preview.
    Pasting into Outlook preserves table borders, colours, and layout.
    """
    import json

    # Outlook uses Word's engine: bgcolor attribute beats CSS background,
    # and <span> with explicit color beats inherited styles on <th>.
    # font-weight:normal on every td prevents Outlook bold inheritance.
    HDR_BG = "#1e3a5f"
    td_base = "font-size:13px;border:1px solid #c8d0d8;padding:7px 12px;font-weight:normal;color:#111111"
    td      = td_base
    td_alt  = td_base + ";background:#f4f6f8"

    def hdr(text):
        """Header cell: <td> + bgcolor attr + white span — reliable in Outlook."""
        return (
            f'<td bgcolor="{HDR_BG}" style="background:{HDR_BG};padding:8px 12px;'
            f'font-size:13px;border:1px solid #c8d0d8;text-align:left">'
            f'<span style="color:#ffffff;font-weight:bold">{text}</span></td>'
        )

    kpi_cells = "".join(
        f'<td style="{td}">{v}</td>'
        for v in [
            team["Completed SP"],
            f"{team['Productivity %']}%",
            team["Bugs"],
            team["Quality Score"],
        ]
    )

    dev_rows = ""
    for i, (_, r) in enumerate(dev_df.sort_values("Productivity %", ascending=False).iterrows()):
        cell = td_alt if i % 2 else td
        dev_rows += (
            f'<tr>'
            f'<td style="{cell}">{r["Developer"]}</td>'
            f'<td style="{cell}">{int(r["Completed SP"])}</td>'
            f'<td style="{cell}">{r["Productivity %"]:.1f}%</td>'
            f'<td style="{cell}">{r["Bugs"]}</td>'
            f'<td style="{cell}">{r["Quality Score"]}</td>'
            f'</tr>'
        )

    report_html = (
        f'<div style="font-family:Calibri,Arial,sans-serif;color:#111111;font-weight:normal">'
        f'<p style="font-size:18px;font-weight:bold;margin-bottom:2px;color:#111111">Team Productivity Report</p>'
        f'<p style="font-size:13px;color:#555555;font-weight:normal;margin-top:0;margin-bottom:14px">{period_label}</p>'
        f'<p style="font-size:13px;font-weight:bold;margin-bottom:6px;color:#111111">Team Overview</p>'
        f'<table style="border-collapse:collapse;margin-bottom:18px" cellpadding="0" cellspacing="0">'
        f'<tr>{hdr("Completed SP")}{hdr("Productivity")}{hdr("Bugs")}{hdr("Quality Score")}</tr>'
        f'<tr>{kpi_cells}</tr>'
        f'</table>'
        f'<p style="font-size:13px;font-weight:bold;margin-bottom:6px;color:#111111">Developer Breakdown</p>'
        f'<table style="border-collapse:collapse;width:100%" cellpadding="0" cellspacing="0">'
        f'<tr>{hdr("Developer")}{hdr("Completed SP")}{hdr("Productivity")}{hdr("Bugs")}{hdr("Quality Score")}</tr>'
        f'{dev_rows}'
        f'</table>'
        f'<p style="font-size:11px;color:#999999;font-weight:normal;margin-top:10px">'
        f'Generated {datetime.now().strftime("%d %b %Y %H:%M")}</p>'
        f'</div>'
    )

    # Embed HTML as a JSON string so JS handles all escaping safely
    html_json = json.dumps(report_html)

    components.html(f"""
<button onclick="copyReport(this)"
  style="background:#4f46e5;color:#fff;border:none;padding:10px 22px;
         border-radius:6px;font-size:14px;cursor:pointer;font-family:sans-serif;
         font-weight:600;letter-spacing:.02em">
  📋 Copy Report
</button>
<span id="msg" style="margin-left:14px;font-size:13px;font-family:sans-serif"></span>
<script>
function copyReport(btn) {{
  const html = {html_json};
  navigator.clipboard.write([
    new ClipboardItem({{
      'text/html': new Blob([html], {{type: 'text/html'}})
    }})
  ]).then(() => {{
    const msg = document.getElementById('msg');
    msg.style.color = '#22c55e';
    msg.textContent = '✓ Copied — paste into Outlook';
    setTimeout(() => msg.textContent = '', 3000);
  }}).catch(() => {{
    const msg = document.getElementById('msg');
    msg.style.color = '#ef4444';
    msg.textContent = 'Copy failed — select the table below and use Ctrl+C';
  }});
}}
</script>
""", height=52)



# =====================
# Raw tasks expander
# =====================

def render_raw_data(df: pd.DataFrame, bugs_df: pd.DataFrame, default_start: date, default_end: date):
    with st.expander("Raw Tasks & Bugs", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            raw_start = st.date_input("From", value=default_start, key="raw_start")
        with col2:
            raw_end = st.date_input("To", value=default_end, key="raw_end")

        all_devs = sorted(df["Developer"].dropna().unique())
        sel_devs = st.multiselect("Filter by developer (optional)", all_devs)

        tasks_f = df[
            (df["Due Date"].dt.date >= raw_start) &
            (df["Due Date"].dt.date <= raw_end)
        ].copy()
        tasks_f["Due Date"] = tasks_f["Due Date"].dt.strftime("%d-%b-%Y").str.upper()

        if bugs_df is not None and not bugs_df.empty:
            bugs_f = bugs_df[
                (bugs_df["Created"].dt.date >= raw_start) &
                (bugs_df["Created"].dt.date <= raw_end)
            ].copy()
            bugs_f["Created"] = bugs_f["Created"].dt.strftime("%d-%b-%Y").str.upper()
        else:
            bugs_f = pd.DataFrame()

        if sel_devs:
            tasks_f = tasks_f[tasks_f["Developer"].isin(sel_devs)]
            if not bugs_f.empty:
                bugs_f = bugs_f[bugs_f["Developer"].isin(sel_devs)]

        st.markdown("**Tasks**")
        if not tasks_f.empty:
            st.dataframe(
                tasks_f[["Key", "Summary", "Developer", "Status", "Due Date", "Story Points"]],
                use_container_width=True, hide_index=True,
            )
        else:
            st.info("No tasks in this range.")

        st.markdown("**Bugs**")
        if not bugs_f.empty:
            st.dataframe(
                bugs_f[["Key", "Summary", "Developer", "Created"]],
                use_container_width=True, hide_index=True,
            )
        else:
            st.info("No bugs in this range.")


# =====================
# Main
# =====================

def main():
    st.set_page_config(
        page_title="Team Productivity Dashboard",
        page_icon="📊",
        layout="wide",
    )

    _inject_css()

    ist = pytz.timezone("Asia/Kolkata")
    today = datetime.now(ist).date()

    # ── Header ──────────────────────────────────────────────────────────
    st.markdown("""
<div style="display:flex;align-items:center;gap:14px;padding:8px 0 4px">
  <div style="width:42px;height:42px;background:linear-gradient(135deg,#6366f1,#8b5cf6);
              border-radius:10px;display:flex;align-items:center;justify-content:center;
              font-size:20px;flex-shrink:0;">📊</div>
  <div>
    <div style="font-size:22px;font-weight:700;color:#0f172a;letter-spacing:-0.3px;
                line-height:1.2">Team Productivity Dashboard</div>
    <div style="font-size:12px;color:#64748b;margin-top:2px">
      Engineering team metrics &amp; quality tracking
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── Controls ────────────────────────────────────────────────────────
    with st.container():
        period_types = [
            "This Week", "This Month", "Last Month",
            "This Quarter", "Last Quarter", "Custom",
        ]
        def _period_label(pt):
            if pt == "Custom":
                return "Custom"
            s, e, _, _ = get_period_bounds(pt, today)
            return f"{pt}  ({s.strftime('%d %b')} – {e.strftime('%d %b %Y')})"

        period_labels = [_period_label(pt) for pt in period_types]

        ctrl_col, btn_col, time_col = st.columns([3, 1, 2])
        with ctrl_col:
            selected_label = st.selectbox(
                "Period",
                period_labels,
                index=2,          # default: Last Month
                label_visibility="collapsed",
            )
            period_type = period_types[period_labels.index(selected_label)]
        with btn_col:
            if st.button("🔄 Refresh"):
                st.cache_data.clear()
                st.rerun()
        with time_col:
            now_str = datetime.now(ist).strftime("%d %b %Y · %H:%M %Z")
            st.markdown(
                f'<div style="display:flex;justify-content:flex-end;align-items:center;height:100%">'
                f'<span style="background:#f1f5f9;border:1px solid #e2e8f0;border-radius:20px;'
                f'padding:4px 12px;font-size:12px;color:#64748b;font-weight:500;white-space:nowrap">'
                f'🕐 {now_str}</span></div>',
                unsafe_allow_html=True,
            )

        if period_type == "Custom":
            c1, c2, _ = st.columns([2, 2, 2])
            with c1:
                custom_start = st.date_input("Start date", value=today - timedelta(weeks=4))
            with c2:
                custom_end = st.date_input("End date", value=today)

    # ── Resolve date bounds ─────────────────────────────────────────────
    if period_type == "Custom":
        start_date, end_date = custom_start, min(custom_end, today)
        delta = end_date - start_date
        prev_end = start_date - timedelta(days=1)
        prev_start = prev_end - delta
        period_label = f"{start_date.strftime('%d %b %Y')} – {end_date.strftime('%d %b %Y')}"
    else:
        start_date, end_date, prev_start, prev_end = get_period_bounds(period_type, today)
        period_label = f"{period_type} ({start_date.strftime('%d %b')} – {end_date.strftime('%d %b %Y')})"

    st.markdown("<div style='height:1px;background:linear-gradient(90deg,#6366f1,transparent);margin:12px 0 20px'></div>", unsafe_allow_html=True)

    # ── Load data ───────────────────────────────────────────────────────
    with st.spinner("Loading data from Jira…"):
        df = load_jira_data()
        bugs_df = load_bug_data()

    if df.empty:
        st.warning("No task data loaded. Check your Jira credentials and filter ID.")
        st.stop()

    # ── Compute metrics ─────────────────────────────────────────────────
    curr_team, curr_dev = compute_metrics(df, bugs_df, start_date, end_date)
    prev_team, prev_dev = compute_metrics(df, bugs_df, prev_start, prev_end)

    DIVIDER = "<div style='height:1px;background:linear-gradient(90deg,#6366f1,transparent);margin:20px 0'></div>"

    # ── Section 1: Team KPI cards ────────────────────────────────────────
    _section_header("Team Overview", period_label)
    render_kpi_cards(curr_team, prev_team)

    st.markdown(DIVIDER, unsafe_allow_html=True)

    # ── Section 2: Developer breakdown ───────────────────────────────────
    _section_header("Developer Breakdown", "Individual performance metrics")
    render_dev_table(curr_dev, prev_dev)

    st.markdown(DIVIDER, unsafe_allow_html=True)

    # ── Section 2b: Developer drill-down ─────────────────────────────────
    with st.expander("Developer Drill-Down", expanded=False):
        render_dev_drilldown(df, bugs_df, curr_dev, prev_dev, start_date, end_date)

    st.markdown(DIVIDER, unsafe_allow_html=True)

    # ── Raw tasks (collapsed) ────────────────────────────────────────────
    render_raw_data(df, bugs_df, start_date, end_date)

    # ── Share Report ─────────────────────────────────────────────────────
    st.markdown(DIVIDER, unsafe_allow_html=True)
    _section_header("Share Report", "Copy Outlook-ready summary to clipboard")
    st.markdown("<p style='font-size:12px;color:#64748b;margin-top:-8px;margin-bottom:12px'>Click the button below to copy the formatted table — paste directly into Outlook or any email client.</p>", unsafe_allow_html=True)

    render_share_section(period_label, curr_team, curr_dev)


if __name__ == "__main__":
    main()
