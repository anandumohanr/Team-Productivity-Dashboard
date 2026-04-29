import streamlit as st
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, date
import altair as alt
import pytz
import unicodedata
import re
import html as _html
from requests.auth import HTTPBasicAuth
from streamlit_autorefresh import st_autorefresh

# =====================
# Constants
# =====================
COMPLETED_STATUSES = ["ACCEPTED IN QA", "CLOSED"]
SP_BASELINE_PER_DAY = 1          # 5 SP/week baseline
PROD_GREEN_THRESHOLD = 80        # productivity % → green
PROD_YELLOW_THRESHOLD = 60       # productivity % → yellow, else red
QUALITY_GREEN_THRESHOLD = 80
QUALITY_YELLOW_THRESHOLD = 60
HIGH_BUG_COUNT_THRESHOLD = 3

REQUIRED_JIRA_SECRETS = [
    "JIRA_DOMAIN",
    "JIRA_EMAIL",
    "JIRA_API_TOKEN",
    "JIRA_FILTER_ID",
]
OPTIONAL_BUG_FILTER_SECRET = "JIRA_BUG_FILTER_ID"
OPTIONAL_BOARD_SECRET = "JIRA_BOARD_ID"

# Sprint structure — 2-week sprint has 10 calendar working days but only 7 dev days
# (3 days are sprint ceremonies: planning, review, retro)
SPRINT_WORKING_DAYS = 10
SPRINT_DEV_DAYS = 7
DEV_DAY_RATIO = SPRINT_DEV_DAYS / SPRINT_WORKING_DAYS   # 0.7

# Sprint status grouping — case-insensitive comparison (Jira returns title-case)
SPRINT_DONE_STATUSES = {"ACCEPTED IN QA", "CLOSED"}
SPRINT_IN_PROGRESS_STATUSES = {"IN PROGRESS", "TEST", "DESIGN", "REVIEW", "IN REVIEW", "STAGING", "QA", "REOPENED"}
SPRINT_TEST_MODE_STATUSES = {"TEST"}

COMMITMENT_FRESH = "Fresh Commitment"
COMMITMENT_CF_DEV = "Carry-forward Dev"
COMMITMENT_CF_TEST = "Carry-forward Test"
ACTIONABLE_COMMITMENT_BUCKETS = {COMMITMENT_FRESH, COMMITMENT_CF_DEV}

# Sprint planning capacity thresholds (per developer, per 2-week sprint)
SPRINT_MIN_SP_PER_DEV = 7
SPRINT_MAX_SP_PER_DEV = 10
SPRINT_EXPECTED_SP_PER_DEV = 7

# =====================
# Data loading
# =====================

def _get_secret_value(key: str, default: str = "") -> str:
    try:
        value = st.secrets.get(key, default)
    except Exception:
        return default
    if value is None:
        return default
    return str(value).strip()


def validate_jira_config():
    config = {key: _get_secret_value(key) for key in REQUIRED_JIRA_SECRETS}
    missing = [key for key, value in config.items() if not value]
    config[OPTIONAL_BUG_FILTER_SECRET] = _get_secret_value(OPTIONAL_BUG_FILTER_SECRET)
    return missing, config


def render_missing_config(missing_keys):
    st.error("Jira configuration is incomplete.")
    st.markdown(
        """
<div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:8px;padding:16px 18px;margin-top:8px">
  <div style="font-size:14px;font-weight:700;color:#0f172a;margin-bottom:8px">Missing required Streamlit secrets</div>
  <div style="font-size:13px;color:#475569;margin-bottom:12px">
    Add these values to local Streamlit secrets or the deployment configuration before loading Jira data.
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.code("\n".join(missing_keys), language="text")


def _empty_task_df():
    return pd.DataFrame(
        columns=[
            "Key",
            "Summary",
            "Status",
            "Due Date",
            "Story Points",
            "Developer",
            "Created",
            "Is Completed",
            "Week",
            "Month",
            "Quarter",
        ]
    )


def _empty_bug_df():
    return pd.DataFrame(columns=["Key", "Summary", "Created", "Developer", "Week", "Month", "Quarter"])

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

        resp = requests.get(url, headers=headers, auth=auth, params=params, timeout=30)
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
def load_jira_data(jira_domain: str, email: str, token: str, filter_id: str):
    jql = f"filter={filter_id}"
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

        df = pd.DataFrame(
            data,
            columns=["Key", "Summary", "Status", "Due Date", "Story Points", "Developer", "Created"],
        )
        if df.empty:
            return _empty_task_df()

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
        return _empty_task_df()


@st.cache_data(ttl=14400, show_spinner=False)
def load_bug_data(jira_domain: str, email: str, token: str, bug_filter_id: str):
    if not bug_filter_id:
        return _empty_bug_df()

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

        df = pd.DataFrame(rows, columns=["Key", "Summary", "Created", "Developer"])
        if df.empty:
            return _empty_bug_df()

        df["Created"] = pd.to_datetime(df["Created"], errors="coerce")
        df["Developer"] = df["Developer"].apply(_normalize_str).replace("", "(Unassigned)")
        df["Week"] = df["Created"].dt.to_period("W").dt.start_time
        df["Month"] = df["Created"].dt.to_period("M").dt.start_time
        df["Quarter"] = df["Created"].dt.to_period("Q").dt.start_time
        return df
    except Exception as e:
        st.error(f"Failed to fetch bug data: {e}")
        return _empty_bug_df()


# =====================
# Sprint data loading
# =====================

@st.cache_data(ttl=3600, show_spinner=False)
def load_sprint_list(jira_domain: str, email: str, token: str, board_id: str) -> list:
    url = f"https://{jira_domain}/rest/agile/1.0/board/{board_id}/sprint"
    auth = HTTPBasicAuth(email, token)
    headers = {"Accept": "application/json"}
    all_sprints, start_at = [], 0
    while True:
        params = {"state": "active,closed,future", "maxResults": 50, "startAt": start_at}
        try:
            resp = requests.get(url, headers=headers, auth=auth, params=params)
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            break
        values = data.get("values", [])
        all_sprints.extend(values)
        if data.get("isLast", True) or not values:
            break
        start_at += len(values)
    all_sprints.sort(key=lambda s: s.get("startDate") or "", reverse=True)
    return all_sprints


@st.cache_data(ttl=3600, show_spinner=False)
def load_sprint_issues(jira_domain: str, email: str, token: str, sprint_id: int) -> pd.DataFrame:
    jql = f"sprint = {sprint_id} ORDER BY status ASC"
    fields = "key,summary,status,customfield_10010,customfield_11012"
    empty = pd.DataFrame(columns=["Key", "Summary", "Status", "Status Group", "Story Points", "Developer", "Is Completed"])
    try:
        all_issues = _jira_search_all(jira_domain, email, token, jql, fields)
        rows = []
        for issue in all_issues:
            f = issue.get("fields", {}) or {}
            status_name = (f.get("status") or {}).get("name", "") or ""
            status_upper = status_name.upper()
            if status_upper in SPRINT_DONE_STATUSES:
                status_group = "Done"
            elif status_upper in SPRINT_IN_PROGRESS_STATUSES:
                status_group = "In Progress"
            else:
                status_group = "Not Started"
            dev_raw = f.get("customfield_11012")
            developer = _normalize_str(_normalize_developer(dev_raw)) if dev_raw else ""
            developer = developer or "(Unassigned)"
            rows.append({
                "Key": issue.get("key", ""),
                "Summary": f.get("summary", "") or "",
                "Status": status_name,
                "Status Group": status_group,
                "Story Points": f.get("customfield_10010"),
                "Developer": developer,
                "Is Completed": status_upper in SPRINT_DONE_STATUSES,
            })
        if not rows:
            return empty
        df = pd.DataFrame(rows)
        df["Story Points"] = pd.to_numeric(df["Story Points"], errors="coerce").fillna(0)
        return df
    except Exception as e:
        st.error(f"Failed to fetch sprint issues: {e}")
        return empty


# =====================
# Period utilities
# =====================

def count_working_days(start_dt, end_dt):
    return sum(1 for d in pd.date_range(start=start_dt, end=end_dt) if d.weekday() < 5)


def count_dev_days(start_dt, end_dt):
    """Working days adjusted for sprint ceremony overhead (7 dev days per 10-day sprint)."""
    return count_working_days(start_dt, end_dt) * DEV_DAY_RATIO


def get_period_bounds(period_type: str, today: date):
    """Returns (start, end, prev_start, prev_end) as date objects."""
    if period_type == "Current Week":
        start = today - timedelta(days=today.weekday())
        end = start + timedelta(days=6)
        prev_start = start - timedelta(weeks=1)
        prev_end = prev_start + timedelta(days=6)
    elif period_type == "Current Month":
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
    elif period_type == "Current Quarter":
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
    elif period_type == "Current Year":
        start = date(today.year, 1, 1)
        end = date(today.year, 12, 31)
        prev_start = date(today.year - 1, 1, 1)
        prev_end = date(today.year - 1, 12, 31)
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

    Productivity % = Completed SP / (dev_days × SP_BASELINE_PER_DAY × n_devs)
                   dev_days = working_days × DEV_DAY_RATIO (7 dev days per 10-day sprint)
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

    all_devs = sorted(in_period["Developer"].dropna().unique())

    working_days = count_dev_days(start_date, end_date)
    expected_sp_per_dev = max(working_days * SP_BASELINE_PER_DAY, 1)
    target_sp_per_dev = int(round(expected_sp_per_dev))

    dev_rows = []
    for dev in all_devs:
        dev_completed = completed[completed["Developer"] == dev]
        dev_bugs = bugs_in_period[bugs_in_period["Developer"] == dev] if not bugs_in_period.empty else empty_bugs

        completed_sp = dev_completed["Story Points"].sum()
        bug_count = len(dev_bugs)

        productivity_pct = round(completed_sp / target_sp_per_dev * 100, 1) if target_sp_per_dev > 0 else 0.0
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
            "Target SP": target_sp_per_dev,
            "Productivity %": productivity_pct,
            "Bugs": bug_count,
            "Quality Score": quality_score,
        })

    dev_df = pd.DataFrame(dev_rows) if dev_rows else pd.DataFrame(
        columns=["Developer", "Completed SP", "Target SP", "Productivity %", "Bugs", "Quality Score"]
    )

    n_real_devs = sum(1 for d in all_devs if d != "(Unassigned)")
    n_devs = max(len(all_devs), 1)
    team_completed_sp = int(completed["Story Points"].sum())
    team_expected_sp = max(target_sp_per_dev * n_real_devs, 1)
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
        "Active Devs": n_real_devs,
        "Capacity SP": target_sp_per_dev * n_real_devs,
        "Completed SP": team_completed_sp,
        "Productivity %": team_productivity,
        "Bugs": total_bugs,
        "Quality Score": team_quality,
    }

    return team_metrics, dev_df


def _commitment_bucket_for_issue(key: str, status: str, carry_forward_keys: set) -> str:
    if key not in carry_forward_keys:
        return COMMITMENT_FRESH
    if str(status or "").strip().upper() in SPRINT_TEST_MODE_STATUSES:
        return COMMITMENT_CF_TEST
    return COMMITMENT_CF_DEV


def add_commitment_classification(df: pd.DataFrame, carry_forward_keys: set) -> pd.DataFrame:
    """Classify sprint issues for planning capacity without mutating raw Jira rows."""
    result = df.copy()
    carry_forward_keys = set(carry_forward_keys or [])
    if result.empty:
        result["Origin"] = pd.Series(dtype="object")
        result["Commitment Bucket"] = pd.Series(dtype="object")
        result["Is Actionable Commitment"] = pd.Series(dtype=bool)
        return result

    result["Origin"] = result["Key"].apply(
        lambda k: "Carry-forward" if k in carry_forward_keys else "New"
    )
    result["Commitment Bucket"] = result.apply(
        lambda r: _commitment_bucket_for_issue(
            r.get("Key", ""), r.get("Status", ""), carry_forward_keys
        ),
        axis=1,
    )
    result["Is Actionable Commitment"] = result["Commitment Bucket"].isin(
        ACTIONABLE_COMMITMENT_BUCKETS
    )
    return result


def _commitment_summary(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "fresh_count": 0, "fresh_sp": 0.0,
            "cf_dev_count": 0, "cf_dev_sp": 0.0,
            "cf_test_count": 0, "cf_test_sp": 0.0,
            "cf_count": 0, "cf_sp": 0.0,
            "actionable_count": 0, "actionable_sp": 0.0,
        }

    if "Commitment Bucket" in df.columns:
        bucket = df["Commitment Bucket"]
    elif "Origin" in df.columns:
        bucket = df["Origin"].map({
            "Carry-forward": COMMITMENT_CF_DEV,
            "New": COMMITMENT_FRESH,
        }).fillna(COMMITMENT_FRESH)
    else:
        bucket = pd.Series(COMMITMENT_FRESH, index=df.index)

    fresh_mask = bucket == COMMITMENT_FRESH
    cf_dev_mask = bucket == COMMITMENT_CF_DEV
    cf_test_mask = bucket == COMMITMENT_CF_TEST
    actionable_mask = bucket.isin(ACTIONABLE_COMMITMENT_BUCKETS)

    fresh_sp = round(float(df.loc[fresh_mask, "Story Points"].sum()), 1)
    cf_dev_sp = round(float(df.loc[cf_dev_mask, "Story Points"].sum()), 1)
    cf_test_sp = round(float(df.loc[cf_test_mask, "Story Points"].sum()), 1)
    cf_count = int(cf_dev_mask.sum() + cf_test_mask.sum())
    cf_sp = round(cf_dev_sp + cf_test_sp, 1)

    return {
        "fresh_count": int(fresh_mask.sum()),
        "fresh_sp": fresh_sp,
        "cf_dev_count": int(cf_dev_mask.sum()),
        "cf_dev_sp": cf_dev_sp,
        "cf_test_count": int(cf_test_mask.sum()),
        "cf_test_sp": cf_test_sp,
        "cf_count": cf_count,
        "cf_sp": cf_sp,
        "actionable_count": int(actionable_mask.sum()),
        "actionable_sp": round(fresh_sp + cf_dev_sp, 1),
    }


def compute_sprint_metrics(df: pd.DataFrame):
    """Returns (team_dict, dev_df) for a sprint issues DataFrame."""
    has_origin = "Origin" in df.columns
    empty_cols = ["Developer", "Committed SP", "Delivered SP", "Completion %", "Total Issues", "Done Issues"]
    if has_origin:
        empty_cols += ["Fresh SP", "Carry-forward SP"]
    empty_dev = pd.DataFrame(columns=empty_cols)
    if df.empty:
        return {"committed_sp": 0, "delivered_sp": 0, "completion_pct": 0.0, "total_issues": 0,
                "done_issues": 0, "carryover_issues": 0, "active_devs": 0,
                "status_breakdown": {"Done": 0, "In Progress": 0, "Not Started": 0}}, empty_dev

    committed_sp = df["Story Points"].sum()
    delivered_sp = df[df["Is Completed"]]["Story Points"].sum()
    carryover_sp = committed_sp - delivered_sp
    completion_pct = round(float(delivered_sp) / float(committed_sp) * 100, 1) if committed_sp > 0 else 0.0
    total_issues = len(df)
    done_issues = int(df["Is Completed"].sum())
    carryover_issues = total_issues - done_issues
    devs = sorted(df["Developer"].dropna().unique())
    active_devs = sum(1 for d in devs if d != "(Unassigned)")
    status_breakdown = {
        grp: round(float(df[df["Status Group"] == grp]["Story Points"].sum()), 1)
        for grp in df["Status Group"].unique()
    }
    dev_rows = []
    for dev in devs:
        ddf = df[df["Developer"] == dev]
        committed = float(ddf["Story Points"].sum())
        delivered = float(ddf[ddf["Is Completed"]]["Story Points"].sum())
        cp = round(delivered / committed * 100, 1) if committed > 0 else 0.0
        row = {
            "Developer": dev,
            "Committed SP": round(committed, 1),
            "Delivered SP": round(delivered, 1),
            "Completion %": cp,
            "Total Issues": len(ddf),
            "Done Issues": int(ddf["Is Completed"].sum()),
        }
        if has_origin:
            fresh = float(ddf[ddf["Origin"] == "New"]["Story Points"].sum())
            cf = float(ddf[ddf["Origin"] == "Carry-forward"]["Story Points"].sum())
            row["Fresh SP"] = round(fresh, 1)
            row["Carry-forward SP"] = round(cf, 1)
        dev_rows.append(row)
    dev_df = pd.DataFrame(dev_rows).sort_values("Committed SP", ascending=False).reset_index(drop=True)
    team = {
        "committed_sp": round(float(committed_sp), 1),
        "delivered_sp": round(float(delivered_sp), 1),
        "carryover_sp": round(float(carryover_sp), 1),
        "completion_pct": completion_pct,
        "total_issues": total_issues,
        "done_issues": done_issues,
        "carryover_issues": carryover_issues,
        "active_devs": active_devs,
        "status_breakdown": status_breakdown,
    }
    return team, dev_df


def compute_planning_metrics(df: pd.DataFrame):
    """Returns (team_dict, dev_df) for the Planning lens.

    Planning commitment excludes carry-forward already in TEST, because that work
    is waiting on QE rather than consuming developer capacity.
    """
    base_cols = [
        "Developer", "Committed SP", "Balance SP", "Total Issues",
        "Fresh SP", "CF Dev SP", "CF Test SP", "Utilization",
    ]
    empty_dev = pd.DataFrame(columns=base_cols)
    empty_team = {
        "committed_sp": 0.0, "committed_count": 0,
        "total_issues": 0, "active_devs": 0, "avg_sp_per_dev": 0.0,
        "fresh_count": 0, "fresh_sp": 0.0,
        "cf_dev_count": 0, "cf_dev_sp": 0.0,
        "cf_test_count": 0, "cf_test_sp": 0.0,
        "unestimated_count": 0, "unestimated_sp": 0.0,
        "unassigned_count": 0, "unassigned_sp": 0.0,
    }
    if df.empty:
        return empty_team, empty_dev

    summary = _commitment_summary(df)
    committed_sp = summary["actionable_sp"]
    total_issues = len(df)

    if "Is Actionable Commitment" in df.columns:
        actionable_mask = df["Is Actionable Commitment"].fillna(False)
    elif "Commitment Bucket" in df.columns:
        actionable_mask = df["Commitment Bucket"].isin(ACTIONABLE_COMMITMENT_BUCKETS)
    else:
        actionable_mask = pd.Series(True, index=df.index)

    actionable_df = df[actionable_mask]
    active_devs = sum(
        1 for d in actionable_df["Developer"].dropna().unique() if d != "(Unassigned)"
    )
    avg_sp = round(committed_sp / active_devs, 1) if active_devs > 0 else 0.0

    unestimated_mask = actionable_mask & (df["Story Points"] == 0)
    unassigned_mask = actionable_mask & (df["Developer"] == "(Unassigned)")
    unestimated_count = int(unestimated_mask.sum())
    unestimated_sp = round(float(df.loc[unestimated_mask, "Story Points"].sum()), 1)
    unassigned_count = int(unassigned_mask.sum())
    unassigned_sp = round(float(df.loc[unassigned_mask, "Story Points"].sum()), 1)

    team = {
        "committed_sp": committed_sp,
        "committed_count": summary["actionable_count"],
        "total_issues": total_issues,
        "active_devs": active_devs,
        "avg_sp_per_dev": avg_sp,
        "fresh_count": summary["fresh_count"],
        "fresh_sp": summary["fresh_sp"],
        "cf_dev_count": summary["cf_dev_count"],
        "cf_dev_sp": summary["cf_dev_sp"],
        "cf_test_count": summary["cf_test_count"],
        "cf_test_sp": summary["cf_test_sp"],
        "unestimated_count": unestimated_count,
        "unestimated_sp": unestimated_sp,
        "unassigned_count": unassigned_count,
        "unassigned_sp": unassigned_sp,
    }

    devs = sorted(df["Developer"].dropna().unique())
    dev_rows = []
    for dev in devs:
        if dev == "(Unassigned)":
            continue

        ddf = df[df["Developer"] == dev]
        dev_summary = _commitment_summary(ddf)
        committed = dev_summary["actionable_sp"]
        balance_sp = round(max(SPRINT_EXPECTED_SP_PER_DEV - committed, 0.0), 1)
        dev_rows.append({
            "Developer": dev,
            "Committed SP": committed,
            "Balance SP": balance_sp,
            "Total Issues": len(ddf),
            "Fresh SP": dev_summary["fresh_sp"],
            "CF Dev SP": dev_summary["cf_dev_sp"],
            "CF Test SP": dev_summary["cf_test_sp"],
            "Utilization": _utilization_label(committed),
        })

    if not dev_rows:
        return team, empty_dev

    planning_dev = pd.DataFrame(dev_rows).sort_values(
        ["Committed SP", "Fresh SP", "CF Dev SP", "CF Test SP"],
        ascending=False,
    ).reset_index(drop=True)
    return team, planning_dev


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

/* Hide the invisible iframe that streamlit-autorefresh injects */
div[data-testid="stElementContainer"]:has(iframe[title*="autorefresh"]),
div[data-testid="element-container"]:has(iframe[title*="autorefresh"]) {
    display: none !important;
}
iframe[title*="autorefresh"] { display: none !important; }

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

/* ── Inner Analytics sub-tabs (st.tabs) ─────────────────────────── */
[data-testid="stTabsTabList"] {
    gap: 0 !important;
    border-bottom: 1px solid #e2e8f0 !important;
    background: transparent !important;
    padding: 0 !important;
    margin-bottom: 4px !important;
}
button[data-testid="stTab"] {
    font-size: 13px !important;
    font-weight: 500 !important;
    color: #94a3b8 !important;
    padding: 10px 18px !important;
    border-radius: 0 !important;
    background: transparent !important;
    border-bottom: 2px solid transparent !important;
}
button[data-testid="stTab"]:hover {
    color: #6366f1 !important;
    background: #f8fafc !important;
}
button[data-testid="stTab"][aria-selected="true"] {
    color: #6366f1 !important;
    font-weight: 600 !important;
    border-bottom: 2px solid #6366f1 !important;
    background: transparent !important;
}

/* ── Selectbox polish ────────────────────────────────────────────── */
[data-testid="stSelectbox"] > div > div {
    box-shadow: 0 1px 3px rgba(0,0,0,.05) !important;
}

/* ── Generic card hover (applied where cards are used) ───────────── */
.kpi-card:hover { box-shadow: 0 4px 16px rgba(99,102,241,.12) !important; }

/* ── Clickable developer-row overlay (table rows in render_dev_table) ── */
[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"][class*="st-key-btn_dev_row_"]),
[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"][class*="st-key-plan_drill_"]) {
    position: relative !important;
    gap: 0 !important;
    margin-top: -1rem !important;
}
[data-testid="stElementContainer"][class*="st-key-btn_dev_row_"],
[data-testid="stElementContainer"][class*="st-key-plan_drill_"] {
    position: absolute !important; inset: 0 !important;
    z-index: 10 !important; margin: 0 !important; padding: 0 !important;
    pointer-events: none !important;
}
[class*="st-key-btn_dev_row_"] > div,
[class*="st-key-plan_drill_"] > div { height: 100% !important; margin: 0 !important; }
[class*="st-key-btn_dev_row_"] > div > button,
[class*="st-key-plan_drill_"] > div > button {
    height: 100% !important; width: 100% !important;
    min-height: 0 !important;
    background: transparent !important; border: none !important;
    box-shadow: none !important; outline: none !important;
    padding: 0 !important; cursor: pointer !important;
    border-radius: 0 !important; pointer-events: auto !important;
}
[class*="st-key-btn_dev_row_"] > div > button:hover,
[class*="st-key-btn_dev_row_"] > div > button:focus,
[class*="st-key-btn_dev_row_"] > div > button:active,
[class*="st-key-plan_drill_"] > div > button:hover,
[class*="st-key-plan_drill_"] > div > button:focus,
[class*="st-key-plan_drill_"] > div > button:active {
    background: transparent !important; box-shadow: none !important; outline: none !important;
}

/* ── Sprint Hero Card (header above the lens tabs) ──────────────────── */
.sprint-hero-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-left: 4px solid #94a3b8;
    box-shadow: 0 2px 8px rgba(0,0,0,.06);
    padding: 14px 22px;
}
.sprint-hero-card-standalone {
    border-radius: 14px;
    margin-bottom: 16px;
}
.sprint-hero-card-with-tabs {
    border-radius: 14px 14px 0 0;
    border-bottom: 1px solid #e2e8f0;
    margin-bottom: 0;
    /* Clip the bottom of the shadow so it doesn't bleed into the tab strip below */
    clip-path: inset(-20px -20px 0 -20px);
}
.sprint-hero-card .hero-row {
    display: flex;
    align-items: center;
    gap: 24px;
    flex-wrap: wrap;
    row-gap: 6px;
}
.sprint-hero-card .hero-name-group {
    display: flex;
    align-items: center;
    gap: 10px;
}
.sprint-hero-card .hero-name {
    font-size: 17px;
    font-weight: 700;
    color: #0f172a;
    letter-spacing: -.01em;
    line-height: 1.2;
}
.sprint-hero-card .hero-meta {
    font-size: 13px;
    color: #64748b;
    line-height: 1.4;
}
.sprint-hero-card .hero-stats {
    font-size: 13px;
    color: #475569;
    font-weight: 500;
    line-height: 1.4;
}
.sprint-hero-card .hero-sep {
    color: #cbd5e1;
    user-select: none;
    margin: 0 2px;
}

/* ── Attached underline tabs (Planning ↔ Execution) ─────────────────── */
[data-testid="stHorizontalBlock"]:has([class*="st-key-lens_plan_"]) {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-top: none !important;
    border-radius: 0 0 14px 14px !important;
    padding: 0 22px !important;
    gap: 4px !important;
    width: auto !important;
    margin: 0 0 16px 0 !important;
    box-shadow: 0 2px 8px rgba(0,0,0,.06) !important;
    /* Clip the top of the shadow so it doesn't bleed onto the card above */
    clip-path: inset(0 -20px -20px -20px) !important;
}
[data-testid="stHorizontalBlock"]:has([class*="st-key-lens_plan_"]) [data-testid="stColumn"] {
    width: auto !important;
    flex: 0 0 auto !important;
    min-width: 0 !important;
    padding: 0 !important;
}
[data-testid="stHorizontalBlock"]:has([class*="st-key-lens_plan_"]) [data-testid="stColumn"]:nth-child(3) {
    display: none !important;
}
[class*="st-key-lens_plan_on_"] > div > button,
[class*="st-key-lens_exec_on_"] > div > button {
    background: transparent !important;
    color: #4f46e5 !important;
    border: none !important;
    border-bottom: 3px solid #6366f1 !important;
    border-radius: 0 !important;
    box-shadow: none !important;
    font-weight: 700 !important;
    font-size: 13px !important;
    letter-spacing: .015em !important;
    padding: 16px 26px 13px !important;
    height: auto !important;
    transition: color .25s ease, border-color .25s ease, background .2s ease !important;
}
[class*="st-key-lens_plan_off_"] > div > button,
[class*="st-key-lens_exec_off_"] > div > button {
    background: transparent !important;
    color: #64748b !important;
    border: none !important;
    border-bottom: 3px solid transparent !important;
    border-radius: 0 !important;
    box-shadow: none !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    letter-spacing: .01em !important;
    padding: 16px 26px 13px !important;
    height: auto !important;
    transition: color .25s ease, border-color .25s ease, background .2s ease !important;
}
[class*="st-key-lens_plan_off_"] > div > button:hover,
[class*="st-key-lens_exec_off_"] > div > button:hover {
    background: rgba(99,102,241,.04) !important;
    color: #334155 !important;
    border-bottom-color: #cbd5e1 !important;
    box-shadow: none !important;
}
[class*="st-key-lens_plan_on_"] > div > button:hover,
[class*="st-key-lens_exec_on_"] > div > button:hover {
    background: rgba(99,102,241,.06) !important;
    color: #4338ca !important;
    border-bottom-color: #4f46e5 !important;
    box-shadow: none !important;
}
[class*="st-key-lens_plan_"] > div > button:focus,
[class*="st-key-lens_exec_"] > div > button:focus {
    outline: none !important;
}

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


def _group_bg(g):
    return {"Done": "#f0fdf4", "In Progress": "#fffbeb", "Not Delivered": "#fff1f2"}.get(g, "#ffffff")


def _group_badge(g):
    if g == "Done":
        return '<span style="background:#dcfce7;color:#16a34a;padding:2px 8px;border-radius:12px;font-size:11px;font-weight:600">Done</span>'
    if g == "In Progress":
        return '<span style="background:#fef3c7;color:#d97706;padding:2px 8px;border-radius:12px;font-size:11px;font-weight:600">In Progress</span>'
    if g == "Not Delivered":
        return '<span style="background:#fee2e2;color:#dc2626;padding:2px 8px;border-radius:12px;font-size:11px;font-weight:600">Not Delivered</span>'
    return '<span style="background:#f1f5f9;color:#64748b;padding:2px 8px;border-radius:12px;font-size:11px;font-weight:600">Not Started</span>'


def _origin_badge(o):
    if o == "Carry-forward":
        return '<span style="background:#fff7ed;color:#c2410c;padding:2px 8px;border-radius:12px;font-size:11px;font-weight:600">↩ Carry-forward</span>'
    return '<span style="background:#eff6ff;color:#1d4ed8;padding:2px 8px;border-radius:12px;font-size:11px;font-weight:600">New</span>'


def _commitment_badge(bucket):
    if bucket == COMMITMENT_CF_TEST:
        return '<span style="background:#ecfeff;color:#0891b2;padding:2px 8px;border-radius:12px;font-size:11px;font-weight:600">CF Test</span>'
    if bucket == COMMITMENT_CF_DEV:
        return '<span style="background:#fff7ed;color:#c2410c;padding:2px 8px;border-radius:12px;font-size:11px;font-weight:600">CF Dev</span>'
    return '<span style="background:#eff6ff;color:#1d4ed8;padding:2px 8px;border-radius:12px;font-size:11px;font-weight:600">Fresh</span>'


def _issue_count_text(count: int) -> str:
    return f"{count} issue{'s' if count != 1 else ''}"


def _utilization_badge(util_label, sp):
    palette = {
        "Under": ("#fee2e2", "#dc2626"),
        "Healthy": ("#dcfce7", "#16a34a"),
        "Over": ("#fed7aa", "#ea580c"),
    }
    bg, fg = palette.get(util_label, ("#f1f5f9", "#64748b"))
    if util_label == "Under":
        body = f"{sp:.1f} / {SPRINT_MIN_SP_PER_DEV} SP"
    elif util_label == "Over":
        body = f"{sp:.1f} / {SPRINT_MAX_SP_PER_DEV} SP"
    else:
        body = f"{sp:.1f} SP"
    return (
        f'<span style="background:{bg};color:{fg};padding:5px 12px;border-radius:12px;'
        f'font-size:12px;font-weight:700">{util_label.upper()} · {body}</span>'
    )


def _utilization_label(committed_sp):
    if committed_sp < SPRINT_MIN_SP_PER_DEV:
        return "Under"
    if committed_sp > SPRINT_MAX_SP_PER_DEV:
        return "Over"
    return "Healthy"


def _sprint_state_accent(sprint_state: str) -> str:
    return {"active": "#22c55e", "future": "#6366f1"}.get(sprint_state, "#94a3b8")


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
        ("Active Devs",    "",  False, "#8b5cf6", "DEV"),
        ("Capacity SP",    "",  False, "#475569", "CAP"),
        ("Completed SP",   "",  False, "#6366f1", "SP"),
        ("Productivity %", "%", False, "#06b6d4", "%"),
        ("Bugs",           "",  True,  "#f43f5e", "BUG"),
        ("Quality Score",  "",  False, "#10b981", "QS"),
    ]

    cols = st.columns(len(metrics_cfg))
    for col, (key, unit, invert, accent, icon) in zip(cols, metrics_cfg):
        val = curr.get(key, 0)
        prev_val = prev.get(key, 0) if prev else None

        if prev_val is not None:
            dstr, is_pos = _fmt_delta(val - prev_val, unit, invert)
        else:
            dstr, is_pos = "—", None

        delta_color = "#16a34a" if is_pos is True else ("#dc2626" if is_pos is False else "#94a3b8")
        display_val = f"{val:.1f}%" if "%" in key else str(int(round(val)))

        if key == "Active Devs":
            sublabel_html = (
                f'<div style="font-size:11px;color:#94a3b8;font-weight:500;'
                f'white-space:nowrap;overflow:hidden;text-overflow:ellipsis">with assigned work</div>'
            )
        elif key == "Capacity SP":
            sublabel_html = (
                f'<div style="font-size:11px;color:#94a3b8;font-weight:500;'
                f'white-space:nowrap;overflow:hidden;text-overflow:ellipsis">dev-day baseline</div>'
            )
        elif key == "Completed SP":
            cap = int(round(curr.get("Capacity SP", 0)))
            display_val = str(int(round(val)))
            sublabel_html = (
                f'<div style="font-size:11px;color:#94a3b8;font-weight:500;'
                f'white-space:nowrap;overflow:hidden;text-overflow:ellipsis">'
                f'{int(round(val))} / {cap} SP capacity</div>'
            )
        elif key == "Quality Score":
            display_val = f'{int(round(val))}'
            sublabel_html = (
                f'<div style="font-size:11px;color:{delta_color};font-weight:600;'
                f'white-space:nowrap;overflow:hidden;text-overflow:ellipsis">{dstr} vs prev</div>'
            )
        else:
            sublabel_html = (
                f'<div style="font-size:11px;color:{delta_color};font-weight:600;'
                f'white-space:nowrap;overflow:hidden;text-overflow:ellipsis">{dstr} vs prev</div>'
            )

        with col:
            st.markdown(f"""
<div style="background:#ffffff;border-radius:12px;padding:18px 16px;
            border:1px solid #e2e8f0;border-top:3px solid {accent};
            position:relative;box-shadow:0 1px 4px rgba(0,0,0,.06);
            min-height:118px">
  <div style="position:absolute;top:12px;right:12px;width:28px;height:28px;
              background:{accent}18;border-radius:7px;display:flex;align-items:center;
              justify-content:center;font-size:11px;font-weight:700;color:{accent}">{icon}</div>
  <div style="font-size:10px;color:#64748b;letter-spacing:.07em;text-transform:uppercase;
              font-weight:600;margin-bottom:8px;padding-right:36px;
              white-space:nowrap;overflow:hidden;text-overflow:ellipsis">{key}</div>
  <div style="font-size:26px;font-weight:700;color:#0f172a;line-height:1;
              margin-bottom:8px;letter-spacing:-.02em">{display_val}</div>
  {sublabel_html}
</div>""", unsafe_allow_html=True)


# =====================
# Developer breakdown table
# =====================

def render_dev_table(
    curr_dev: pd.DataFrame,
    prev_dev: pd.DataFrame,
    df: pd.DataFrame,
    bugs_df: pd.DataFrame,
    start_date: date,
    end_date: date,
):
    if curr_dev.empty:
        st.info("No data for the selected period.")
        return

    filtered = curr_dev.sort_values("Productivity %", ascending=False).reset_index(drop=True)
    n = len(filtered)

    def _badge(p):
        if p >= PROD_GREEN_THRESHOLD:
            return '<span style="display:inline-block;background:#dcfce7;color:#16a34a;padding:5px 12px;border-radius:20px;font-size:11px;font-weight:600;white-space:nowrap">● High</span>'
        elif p >= PROD_YELLOW_THRESHOLD:
            return '<span style="display:inline-block;background:#fef3c7;color:#d97706;padding:5px 12px;border-radius:20px;font-size:11px;font-weight:600;white-space:nowrap">● Mid</span>'
        return '<span style="display:inline-block;background:#fee2e2;color:#dc2626;padding:5px 12px;border-radius:20px;font-size:11px;font-weight:600;white-space:nowrap">● Low</span>'

    def _progress(p):
        w = min(float(p), 100)
        clr = "#16a34a" if p >= PROD_GREEN_THRESHOLD else ("#d97706" if p >= PROD_YELLOW_THRESHOLD else "#dc2626")
        return (
            f'<div style="display:flex;align-items:center;gap:10px;width:100%;min-width:130px">'
            f'<div style="flex:1;background:#e2e8f0;border-radius:4px;height:6px">'
            f'<div style="width:{w:.0f}%;background:{clr};height:6px;border-radius:4px"></div>'
            f'</div>'
            f'<span style="font-size:12px;color:#0f172a;min-width:48px;text-align:right;font-weight:600">{p:.1f}%</span>'
            f'</div>'
        )

    # Shared column widths (sum to 100). table-layout:fixed + identical colgroup
    # across every <table> guarantees columns line up across header + every row.
    COL_WIDTHS = [10, 18, 11, 13, 22, 10, 16]
    colgroup = "<colgroup>" + "".join(f'<col style="width:{w}%">' for w in COL_WIDTHS) + "</colgroup>"

    th = (
        "padding:16px 20px;font-size:11px;font-weight:600;color:#64748b;"
        "text-transform:uppercase;letter-spacing:.07em;text-align:left;"
        "white-space:nowrap;vertical-align:middle;background:#f8fafc;"
        "border:1px solid #e2e8f0"
    )
    td = (
        "padding:18px 20px;font-size:13px;color:#334155;vertical-align:middle;"
        "border:1px solid #e2e8f0;border-top:0"
    )

    # Header table — top corners rounded, full borders so the seam below is one line.
    st.markdown(
        f"""
<div class="dev-table-wrap" style="border-radius:12px 12px 0 0;overflow:hidden;
            box-shadow:0 1px 4px rgba(0,0,0,.06)">
  <table style="width:100%;border-collapse:collapse;table-layout:fixed">
    {colgroup}
    <thead>
      <tr>
        <th style="{th}">Status</th>
        <th style="{th}">Developer</th>
        <th style="{th};text-align:center">Target SP</th>
        <th style="{th};text-align:center">Completed SP</th>
        <th style="{th}">Productivity</th>
        <th style="{th};text-align:center">Bugs</th>
        <th style="{th};text-align:center">Quality Score</th>
      </tr>
    </thead>
  </table>
</div>""",
        unsafe_allow_html=True,
    )

    for i, (_, r) in enumerate(filtered.iterrows()):
        bg = "#f8fafc" if i % 2 == 0 else "#ffffff"
        bug_clr = "#dc2626" if r["Bugs"] > 0 else "#94a3b8"
        q = r["Quality Score"]
        q_clr = "#16a34a" if q >= QUALITY_GREEN_THRESHOLD else ("#d97706" if q >= QUALITY_YELLOW_THRESHOLD else "#dc2626")

        is_last = (i == n - 1)
        radius = "0 0 12px 12px" if is_last else "0"

        with st.container():
            st.markdown(
                f"""
<div class="dev-table-wrap" style="border-radius:{radius};overflow:hidden;
            box-shadow:0 1px 4px rgba(0,0,0,.06)">
  <table style="width:100%;border-collapse:collapse;table-layout:fixed;background:{bg}">
    {colgroup}
    <tbody>
      <tr>
        <td style="{td}">{_badge(r["Productivity %"])}</td>
        <td style="{td};font-weight:600;color:#0f172a">{_html.escape(str(r["Developer"]))}</td>
        <td style="{td};text-align:center;color:#64748b">{int(r["Target SP"])}</td>
        <td style="{td};text-align:center;font-weight:700;font-size:15px;color:#4f46e5">{int(r["Completed SP"])}</td>
        <td style="{td}">{_progress(r["Productivity %"])}</td>
        <td style="{td};text-align:center;font-weight:700;color:{bug_clr}">{r["Bugs"]}</td>
        <td style="{td};text-align:center"><span style="font-weight:700;font-size:14px;color:{q_clr}">{q}</span><span style="font-size:11px;color:#94a3b8;margin-left:2px">/100</span></td>
      </tr>
    </tbody>
  </table>
</div>""",
                unsafe_allow_html=True,
            )
            if st.button(" ", key=f"btn_dev_row_{i}", use_container_width=True):
                _show_dev_drilldown_dialog(
                    r["Developer"], df, bugs_df, curr_dev, prev_dev, start_date, end_date,
                )

# =====================
# Developer drill-down
# =====================

@st.dialog("Developer Drill-Down", width="large")
def _show_dev_drilldown_dialog(
    selected: str,
    df: pd.DataFrame,
    bugs_df: pd.DataFrame,
    curr_dev: pd.DataFrame,
    prev_dev: pd.DataFrame,
    start_date: date,
    end_date: date,
):
    """Single-developer deep dive: KPI cards, historical SP chart, task + bug tables."""
    st.markdown(
        f"<div style='font-size:13px;color:#64748b;margin:-6px 0 14px 0'>"
        f"<span style='font-weight:600;color:#0f172a'>{_html.escape(str(selected))}</span>"
        f" · {start_date.strftime('%d %b')} – {end_date.strftime('%d %b %Y')}"
        f"</div>",
        unsafe_allow_html=True,
    )

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
    gran = st.selectbox(
        "Granularity",
        ["Week", "Month", "Quarter"],
        index=1,
        key=f"drilldown_gran_{selected}",
    )
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
        combined["Capacity SP"] = combined[gran].apply(lambda ts: _period_capacity_sp(ts, gran))

        base = alt.Chart(combined).encode(
            x=alt.X("Period Label:N", title=gran, sort=hist_order)
        )
        sp_line = base.mark_line(point=True, color="#6366f1", strokeWidth=2).encode(
            y=alt.Y("Story Points:Q", title="Completed SP",
                    axis=alt.Axis(titleColor="#6366f1")),
            tooltip=["Period Label", "Story Points"],
        )
        capacity_line = base.mark_line(color="#334155", strokeWidth=2, strokeDash=[6, 4]).encode(
            y=alt.Y("Capacity SP:Q", title="Capacity SP",
                    axis=alt.Axis(titleColor="#334155")),
            tooltip=["Period Label", "Capacity SP"],
        )
        bug_line = base.mark_line(point=True, color="#ef4444", strokeWidth=2, strokeDash=[4, 2]).encode(
            y=alt.Y("Bugs:Q", title="Bugs",
                    axis=alt.Axis(titleColor="#ef4444")),
            tooltip=["Period Label", "Bugs"],
        )
        hist_chart = (
            alt.layer(alt.layer(sp_line, capacity_line), bug_line)
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


@st.dialog("Sprint Task Drill-Down", width="large")
def _show_sprint_dev_dialog(dev: str, sprint_name: str, dev_issues_df: pd.DataFrame,
                             jira_domain: str):
    """Single-developer sprint task list — no historical chart, just commitment."""
    st.markdown(
        f"<div style='font-size:13px;color:#64748b;margin:-6px 0 14px 0'>"
        f"<span style='font-weight:600;color:#0f172a'>{_html.escape(dev)}</span> · "
        f"{_html.escape(sprint_name)}"
        f"</div>",
        unsafe_allow_html=True,
    )

    if dev_issues_df.empty:
        st.info("No issues assigned to this developer in this sprint.")
        return

    dev_summary = _commitment_summary(dev_issues_df)
    committed_sp = dev_summary["actionable_sp"]
    n_issues = len(dev_issues_df)
    util_label = _utilization_label(committed_sp)
    issue_sublabel = "Assigned to dev"
    if dev_summary["cf_test_count"] > 0:
        issue_sublabel = (
            f"{_issue_count_text(dev_summary['cf_test_count'])} in TEST carry-forward"
        )

    kpi_cols = st.columns(3)
    kpi_cfg = [
        ("Committed SP", f"{committed_sp:.1f}", "Fresh + dev CF",      "#6366f1"),
        ("Issues",       str(n_issues),         issue_sublabel,        "#8b5cf6"),
        ("Utilization",  util_label,            f"Target {SPRINT_MIN_SP_PER_DEV}–{SPRINT_MAX_SP_PER_DEV} SP", "#06b6d4"),
    ]
    for col, (label, value, sublabel, accent) in zip(kpi_cols, kpi_cfg):
        with col:
            st.markdown(f"""
<div style="background:#ffffff;border-radius:10px;padding:14px;
            border:1px solid #e2e8f0;border-left:4px solid {accent};
            box-shadow:0 1px 3px rgba(0,0,0,.05)">
  <div style="font-size:10px;color:#64748b;letter-spacing:.07em;text-transform:uppercase;
              font-weight:600;margin-bottom:6px">{label}</div>
  <div style="font-size:22px;font-weight:700;color:#0f172a;line-height:1;
              margin-bottom:4px;letter-spacing:-.02em">{value}</div>
  <div style="font-size:11px;color:#94a3b8;font-weight:500">{sublabel}</div>
</div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    show_commitment = "Commitment Bucket" in dev_issues_df.columns
    jira_base = f"https://{jira_domain}/browse/"
    th = "padding:8px 12px;font-size:11px;font-weight:600;color:#64748b;text-transform:uppercase;letter-spacing:.07em;text-align:left;border-bottom:1px solid #e2e8f0;white-space:nowrap"
    td = "padding:10px 12px;font-size:13px;color:#334155;border-bottom:1px solid #f1f5f9;vertical-align:middle"

    rows_html = ""
    for _, r in dev_issues_df.iterrows():
        sp_str = f"{r['Story Points']:.1f}" if r["Story Points"] > 0 else "—"
        link = (f'<a href="{jira_base}{r["Key"]}" target="_blank" '
                f'style="color:#6366f1;font-weight:600;text-decoration:none">'
                f'{_html.escape(r["Key"])}</a>')
        summ_full = str(r["Summary"])
        summ = _html.escape(summ_full[:90]) + ("…" if len(summ_full) > 90 else "")
        commitment_cell = (
            f'<td style="{td}">{_commitment_badge(r["Commitment Bucket"])}</td>'
            if show_commitment else ""
        )
        rows_html += (
            f'<tr>'
            f'<td style="{td};white-space:nowrap">{link}</td>'
            f'<td style="{td};max-width:340px">{summ}</td>'
            f'<td style="{td};white-space:nowrap">{_html.escape(r["Status"])}</td>'
            f'<td style="{td}">{_group_badge(r["Status Group"])}</td>'
            f'<td style="{td};text-align:center;font-weight:600;color:#4f46e5">{sp_str}</td>'
            f'{commitment_cell}'
            f'</tr>'
        )

    commitment_header = f'<th style="{th}">Commitment</th>' if show_commitment else ""
    st.markdown(f"""
<div style="background:#ffffff;border-radius:12px;border:1px solid #e2e8f0;
            overflow:hidden;overflow-x:auto;box-shadow:0 1px 4px rgba(0,0,0,.06)">
  <table style="width:100%;border-collapse:collapse">
    <thead>
      <tr style="background:#f8fafc">
        <th style="{th}">Key</th>
        <th style="{th}">Summary</th>
        <th style="{th}">Status</th>
        <th style="{th}">Group</th>
        <th style="{th};text-align:center">SP</th>
        {commitment_header}
      </tr>
    </thead>
    <tbody>{rows_html}</tbody>
  </table>
</div>
""", unsafe_allow_html=True)


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


def _period_capacity_sp(ts, gran: str) -> int:
    if pd.isna(ts):
        return 0

    start = pd.Timestamp(ts).date()
    if gran == "Week":
        end = start + timedelta(days=6)
    elif gran == "Month":
        if start.month == 12:
            end = date(start.year + 1, 1, 1) - timedelta(days=1)
        else:
            end = date(start.year, start.month + 1, 1) - timedelta(days=1)
    else:
        q_month = 3 * ((start.month - 1) // 3) + 1
        next_q_month = q_month + 3
        if next_q_month > 12:
            end = date(start.year + 1, 1, 1) - timedelta(days=1)
        else:
            end = date(start.year, next_q_month, 1) - timedelta(days=1)

    return int(round(max(count_dev_days(start, end) * SP_BASELINE_PER_DAY, 1)))


# =====================
# Sprint view
# =====================

def _default_sprint_tab(sprint_state: str, start_dt, today) -> str:
    """Pick the default lens. Planning for future sprints and the first 3 days
    of an active sprint; Execution otherwise."""
    if sprint_state == "future":
        return "Planning"
    if sprint_state == "active" and start_dt is not None:
        days_in = (today - start_dt).days
        if 0 <= days_in < 3:
            return "Planning"
    return "Execution"


def _render_sprint_lens_tabs(sprint_id, default_lens: str) -> str:
    """Segmented-control style tabs for switching between Planning and Execution.
    Active state is encoded into the button key so CSS can style each segment
    differently. State persists in st.session_state per sprint."""
    state_key = f"sprint_lens_{sprint_id}"
    if state_key not in st.session_state:
        st.session_state[state_key] = default_lens
    active = st.session_state[state_key]

    plan_suffix = "on" if active == "Planning" else "off"
    exec_suffix = "on" if active == "Execution" else "off"

    cols = st.columns([1, 1, 8])
    with cols[0]:
        if st.button("📋  Planning", key=f"lens_plan_{plan_suffix}_{sprint_id}",
                     use_container_width=True):
            if active != "Planning":
                st.session_state[state_key] = "Planning"
                st.rerun()
    with cols[1]:
        if st.button("▶  Execution", key=f"lens_exec_{exec_suffix}_{sprint_id}",
                     use_container_width=True):
            if active != "Execution":
                st.session_state[state_key] = "Execution"
                st.rerun()

    return active


def render_sprint_planning_section(issues_df: pd.DataFrame, sprint_name: str,
                                    prev_sprint, jira_domain: str,
                                    divider_html: str):
    team, dev_df = compute_planning_metrics(issues_df)

    _section_header(
        "Sprint Planning",
        f"{team['committed_count']} developer-work issues · {team['active_devs']} capacity developers",
    )

    commitment_split = (
        f"Fresh: {team['fresh_sp']:.1f} SP · "
        f"Dev CF: {team['cf_dev_sp']:.1f} SP · "
        f"Test CF: {team['cf_test_sp']:.1f} SP"
    )

    kpi_cfg = [
        ("Committed SP",  f"{team['committed_sp']:.1f}",
            commitment_split,                                                         "#6366f1", "SP"),
        ("Capacity Devs", str(team['active_devs']),
            "With actionable work",                                                    "#8b5cf6", "DEV"),
        ("Avg SP / Dev",  f"{team['avg_sp_per_dev']:.1f}",
            f"Target {SPRINT_MIN_SP_PER_DEV}–{SPRINT_MAX_SP_PER_DEV}",                 "#06b6d4", "μ"),
        ("Unestimated",   str(team['unestimated_count']),
            f"{_issue_count_text(team['unestimated_count'])} missing SP",
            "#f59e0b", "?"),
        ("Unassigned",    str(team['unassigned_count']),
            f"{team['unassigned_sp']:.1f} SP without owner",                           "#ef4444", "⚠"),
    ]

    kpi_cols = st.columns(len(kpi_cfg))
    for col, (label, value, sublabel, accent, icon) in zip(kpi_cols, kpi_cfg):
        with col:
            st.markdown(f"""
<div style="background:#ffffff;border-radius:12px;padding:18px 16px;
            border:1px solid #e2e8f0;border-top:3px solid {accent};
            position:relative;box-shadow:0 1px 4px rgba(0,0,0,.06)">
  <div style="position:absolute;top:12px;right:12px;width:28px;height:28px;
              background:{accent}18;border-radius:7px;display:flex;align-items:center;
              justify-content:center;font-size:11px;font-weight:700;color:{accent}">{icon}</div>
  <div style="font-size:10px;color:#64748b;letter-spacing:.07em;text-transform:uppercase;
              font-weight:600;margin-bottom:8px">{label}</div>
  <div style="font-size:26px;font-weight:700;color:#0f172a;line-height:1;
              margin-bottom:8px;letter-spacing:-.02em">{value}</div>
  <div style="font-size:11px;color:#94a3b8;font-weight:500">{sublabel}</div>
</div>""", unsafe_allow_html=True)

    st.markdown(divider_html, unsafe_allow_html=True)

    show_commitment_split = "Fresh SP" in dev_df.columns

    _section_header("Developer Capacity", "Fresh + dev carry-forward count toward utilization")

    if dev_df.empty:
        st.markdown("""
<div style="background:#fff7ed;border:1px solid #fdba74;border-radius:10px;padding:18px;
            text-align:center;color:#9a3412;font-weight:500;margin-bottom:20px">
  No developer-owned actionable commitments found. Assign developers in Jira to see capacity utilization.
</div>""", unsafe_allow_html=True)
        return

    th2 = ("padding:14px 16px;font-size:12px;font-weight:600;color:#64748b;"
           "text-transform:uppercase;letter-spacing:.07em;text-align:left;"
           "white-space:nowrap;background:#f8fafc;border:1px solid #e2e8f0")
    td2 = ("padding:16px 18px;font-size:15px;color:#334155;vertical-align:middle;"
           "border:1px solid #e2e8f0;border-top:0")

    col_widths = [19, 11, 11, 9]
    if show_commitment_split:
        col_widths += [11, 11, 11]
    col_widths += [15]
    total = sum(col_widths)
    col_widths = [round(w / total * 100, 1) for w in col_widths]
    colgroup = "<colgroup>" + "".join(f'<col style="width:{w}%">' for w in col_widths) + "</colgroup>"

    headers_html = f'<th style="{th2}">Developer</th>'
    headers_html += f'<th style="{th2};text-align:center">Committed SP</th>'
    headers_html += f'<th style="{th2};text-align:center">Balance SP</th>'
    headers_html += f'<th style="{th2};text-align:center"># Issues</th>'
    if show_commitment_split:
        headers_html += f'<th style="{th2};text-align:center">Fresh SP</th>'
        headers_html += f'<th style="{th2};text-align:center">CF Dev</th>'
        headers_html += f'<th style="{th2};text-align:center">CF Test</th>'
    headers_html += f'<th style="{th2};text-align:center">Utilization</th>'

    st.markdown(f"""
<div class="dev-table-wrap" style="border-radius:12px 12px 0 0;overflow:hidden;
            box-shadow:0 1px 4px rgba(0,0,0,.06)">
  <table style="width:100%;border-collapse:collapse;table-layout:fixed">
    {colgroup}
    <thead>
      <tr>
        {headers_html}
      </tr>
    </thead>
  </table>
</div>""", unsafe_allow_html=True)

    n = len(dev_df)
    issues_by_dev = {dev: ddf for dev, ddf in issues_df.groupby("Developer")}

    for i, (_, r) in enumerate(dev_df.iterrows()):
        bg = "#f8fafc" if i % 2 == 0 else "#ffffff"
        is_last = (i == n - 1)
        radius = "0 0 12px 12px" if is_last else "0"

        committed = float(r["Committed SP"])
        balance = float(r.get("Balance SP", 0))
        util_cell = _utilization_badge(r["Utilization"], committed)

        commitment_cells = ""
        if show_commitment_split:
            fresh = float(r.get("Fresh SP", 0))
            cf_dev = float(r.get("CF Dev SP", 0))
            cf_test = float(r.get("CF Test SP", 0))
            commitment_cells = (
                f'<td style="{td2};text-align:center;font-weight:600;color:#6366f1">{fresh:.1f}</td>'
                f'<td style="{td2};text-align:center;font-weight:600;color:#f59e0b">{cf_dev:.1f}</td>'
                f'<td style="{td2};text-align:center;font-weight:600;color:#0891b2">{cf_test:.1f}</td>'
            )

        with st.container():
            st.markdown(f"""
<div class="dev-table-wrap" style="border-radius:{radius};overflow:hidden;
            box-shadow:0 1px 4px rgba(0,0,0,.06)">
  <table style="width:100%;border-collapse:collapse;table-layout:fixed;background:{bg}">
    {colgroup}
    <tbody>
      <tr>
        <td style="{td2};font-weight:600;color:#0f172a">{_html.escape(str(r["Developer"]))}</td>
        <td style="{td2};text-align:center;font-weight:700;font-size:17px;color:#6366f1">{committed:.1f}</td>
        <td style="{td2};text-align:center;font-weight:700;font-size:17px;color:#16a34a">{balance:.1f}</td>
        <td style="{td2};text-align:center;color:#64748b">{int(r["Total Issues"])}</td>
        {commitment_cells}
        <td style="{td2};text-align:center">{util_cell}</td>
      </tr>
    </tbody>
  </table>
</div>""", unsafe_allow_html=True)
            if st.button(" ", key=f"plan_drill_{i}", use_container_width=True):
                dev_issues = issues_by_dev.get(r["Developer"], pd.DataFrame())
                _show_sprint_dev_dialog(
                    r["Developer"], sprint_name, dev_issues, jira_domain,
                )


def render_sprint_execution_section(issues_df: pd.DataFrame, prev_sprint,
                                     prev_sprint_name, cf_count: int, fresh_count: int,
                                     fresh_sp: float,
                                     jira_domain: str, divider_html: str):
    team, dev_df = compute_sprint_metrics(issues_df)
    commitment_summary = _commitment_summary(issues_df)

    _section_header("Sprint Metrics", f"{team['total_issues']} issues · {team['active_devs']} developers")
    kpi_cols = st.columns(5)
    kpi_cfg = [
        ("Committed SP",  f"{team['committed_sp']:.1f}", "Total SP in sprint",        "#6366f1", "SP"),
        ("Delivered SP",  f"{team['delivered_sp']:.1f}", f"{team['done_issues']} issues done", "#22c55e", "✓"),
        ("Completion %",  f"{team['completion_pct']:.1f}%", "Delivered / Committed",  "#06b6d4", "%"),
        ("Pending SP",    f"{team['carryover_sp']:.1f}", f"{team['carryover_issues']} issues not done", "#f59e0b", "→"),
        ("Active Devs",   str(team['active_devs']),      "Contributors",               "#8b5cf6", "DEV"),
    ]
    for col, (label, value, sublabel, accent, icon) in zip(kpi_cols, kpi_cfg):
        with col:
            st.markdown(f"""
<div style="background:#ffffff;border-radius:12px;padding:18px 16px;
            border:1px solid #e2e8f0;border-top:3px solid {accent};
            position:relative;box-shadow:0 1px 4px rgba(0,0,0,.06)">
  <div style="position:absolute;top:12px;right:12px;width:28px;height:28px;
              background:{accent}18;border-radius:7px;display:flex;align-items:center;
              justify-content:center;font-size:11px;font-weight:700;color:{accent}">{icon}</div>
  <div style="font-size:10px;color:#64748b;letter-spacing:.07em;text-transform:uppercase;
              font-weight:600;margin-bottom:8px">{label}</div>
  <div style="font-size:26px;font-weight:700;color:#0f172a;line-height:1;
              margin-bottom:8px;letter-spacing:-.02em">{value}</div>
  <div style="font-size:11px;color:#94a3b8;font-weight:500">{sublabel}</div>
</div>""", unsafe_allow_html=True)

    if prev_sprint and cf_count > 0:
        prev_label = prev_sprint_name or "previous sprint"
        _section_header("Sprint Composition", f"Fresh commitments vs carry-forward from {prev_label}")
        comp_cols = st.columns(3)
        comp_cfg = [
            ("Fresh Commitment", f"{fresh_sp:.1f} SP",
                f"{_issue_count_text(fresh_count)} new",                    "#6366f1"),
            ("Carry-forward Dev", f"{commitment_summary['cf_dev_sp']:.1f} SP",
                f"{_issue_count_text(commitment_summary['cf_dev_count'])} from {prev_label}", "#f59e0b"),
            ("Carry-forward Test", f"{commitment_summary['cf_test_sp']:.1f} SP",
                f"{_issue_count_text(commitment_summary['cf_test_count'])} in TEST", "#0891b2"),
        ]
        for col, (label, value, sublabel, accent) in zip(comp_cols, comp_cfg):
            with col:
                st.markdown(f"""
<div style="background:#ffffff;border-radius:12px;padding:18px 16px;
            border:1px solid #e2e8f0;border-left:4px solid {accent};
            box-shadow:0 1px 4px rgba(0,0,0,.06)">
  <div style="font-size:10px;color:#64748b;letter-spacing:.07em;text-transform:uppercase;
              font-weight:600;margin-bottom:8px">{label}</div>
  <div style="font-size:24px;font-weight:700;color:#0f172a;line-height:1;
              margin-bottom:6px;letter-spacing:-.02em">{value}</div>
  <div style="font-size:11px;color:#94a3b8;font-weight:500">{sublabel}</div>
</div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(divider_html, unsafe_allow_html=True)

    show_origin_split = bool(prev_sprint) and cf_count > 0 and "Fresh SP" in dev_df.columns
    breakdown_subtitle = (
        "Fresh vs Carry-forward · Planned vs Delivered per developer"
        if show_origin_split else "Planned vs Delivered per developer"
    )
    _section_header("Developer Breakdown", breakdown_subtitle)
    if not dev_df.empty:
        th2 = "padding:10px 14px;font-size:11px;font-weight:600;color:#64748b;text-transform:uppercase;letter-spacing:.07em;text-align:left;border-bottom:1px solid #e2e8f0"
        td2 = "padding:12px 14px;font-size:13px;color:#334155;border-bottom:1px solid #f1f5f9;vertical-align:middle"
        dev_rows = ""
        for i, (_, r) in enumerate(dev_df.iterrows()):
            bg = "#f8fafc" if i % 2 == 0 else "#ffffff"
            cp = r["Completion %"]
            bar_clr = "#16a34a" if cp >= 80 else ("#d97706" if cp >= 60 else "#dc2626")
            bar_w = min(float(cp), 100)
            progress = (
                f'<div style="display:flex;align-items:center;gap:6px;min-width:130px">'
                f'<div style="flex:1;background:#e2e8f0;border-radius:4px;height:5px">'
                f'<div style="width:{bar_w:.0f}%;background:{bar_clr};height:5px;border-radius:4px"></div>'
                f'</div>'
                f'<span style="font-size:12px;font-weight:600;color:#0f172a;min-width:40px;text-align:right">{cp:.1f}%</span>'
                f'</div>'
            )
            origin_cells = ""
            if show_origin_split:
                origin_cells = (
                    f'<td style="{td2};text-align:center;font-weight:600;color:#6366f1">{r["Fresh SP"]:.1f}</td>'
                    f'<td style="{td2};text-align:center;font-weight:600;color:#f59e0b">{r["Carry-forward SP"]:.1f}</td>'
                )
            dev_rows += (
                f'<tr style="background:{bg}">'
                f'<td style="{td2};font-weight:600;color:#0f172a">{_html.escape(str(r["Developer"]))}</td>'
                f'{origin_cells}'
                f'<td style="{td2};text-align:center;font-weight:700;font-size:15px;color:#6366f1">{r["Committed SP"]:.1f}</td>'
                f'<td style="{td2};text-align:center;font-weight:700;font-size:15px;color:#22c55e">{r["Delivered SP"]:.1f}</td>'
                f'<td style="{td2}">{progress}</td>'
                f'<td style="{td2};text-align:center;color:#64748b">{int(r["Done Issues"])}/{int(r["Total Issues"])}</td>'
                f'</tr>'
            )
        origin_headers = (
            f'<th style="{th2};text-align:center">Fresh SP</th>'
            f'<th style="{th2};text-align:center">Carry-forward SP</th>'
        ) if show_origin_split else ""
        st.markdown(f"""
<div style="background:#ffffff;border-radius:12px;border:1px solid #e2e8f0;
            overflow:hidden;overflow-x:auto;box-shadow:0 1px 4px rgba(0,0,0,.06);margin-bottom:20px">
  <table style="width:100%;border-collapse:collapse">
    <thead>
      <tr style="background:#f8fafc">
        <th style="{th2}">Developer</th>
        {origin_headers}
        <th style="{th2};text-align:center">Committed SP</th>
        <th style="{th2};text-align:center">Delivered SP</th>
        <th style="{th2}">Completion</th>
        <th style="{th2};text-align:center">Issues Done</th>
      </tr>
    </thead>
    <tbody>{dev_rows}</tbody>
  </table>
</div>
""", unsafe_allow_html=True)

    st.markdown(divider_html, unsafe_allow_html=True)

    with st.expander(f"Sprint Issues  ({len(issues_df)} total)", expanded=False):
        all_sprint_devs = sorted(issues_df["Developer"].dropna().unique())
        sel_sprint_devs = st.multiselect("Filter by developer", all_sprint_devs, key="sprint_dev_filter")
        display_df = issues_df.copy()
        if sel_sprint_devs:
            display_df = display_df[display_df["Developer"].isin(sel_sprint_devs)]

        jira_base = f"https://{jira_domain}/browse/"
        th3 = "padding:8px 12px;font-size:11px;font-weight:600;color:#64748b;text-transform:uppercase;letter-spacing:.07em;text-align:left;border-bottom:1px solid #e2e8f0;white-space:nowrap"
        td3 = "padding:10px 12px;font-size:13px;color:#334155;border-bottom:1px solid #f1f5f9;vertical-align:middle"

        show_commitment = (prev_sprint is not None) and ("Commitment Bucket" in issues_df.columns)
        issue_rows = ""
        for _, r in display_df.iterrows():
            bg = _group_bg(r["Status Group"])
            sp_str = f"{r['Story Points']:.1f}" if r["Story Points"] > 0 else "—"
            link = f'<a href="{jira_base}{r["Key"]}" target="_blank" style="color:#6366f1;font-weight:600;text-decoration:none">{_html.escape(r["Key"])}</a>'
            summ = _html.escape(str(r["Summary"])[:90]) + ("…" if len(str(r["Summary"])) > 90 else "")
            commitment_cell = (
                f'<td style="{td3}">{_commitment_badge(r["Commitment Bucket"])}</td>'
                if show_commitment else ""
            )
            issue_rows += (
                f'<tr style="background:{bg}">'
                f'<td style="{td3};white-space:nowrap">{link}</td>'
                f'<td style="{td3};max-width:340px">{summ}</td>'
                f'<td style="{td3};white-space:nowrap">{_html.escape(r["Developer"])}</td>'
                f'<td style="{td3};text-align:center;font-weight:600;color:#4f46e5">{sp_str}</td>'
                f'<td style="{td3};white-space:nowrap">{_html.escape(r["Status"])}</td>'
                f'<td style="{td3}">{_group_badge(r["Status Group"])}</td>'
                f'{commitment_cell}'
                f'</tr>'
            )

        commitment_header = f'<th style="{th3}">Commitment</th>' if show_commitment else ""
        st.markdown(f"""
<div style="background:#ffffff;border-radius:12px;border:1px solid #e2e8f0;
            overflow:hidden;overflow-x:auto;box-shadow:0 1px 4px rgba(0,0,0,.06)">
  <table style="width:100%;border-collapse:collapse">
    <thead>
      <tr style="background:#f8fafc">
        <th style="{th3}">Key</th>
        <th style="{th3}">Summary</th>
        <th style="{th3}">Developer</th>
        <th style="{th3};text-align:center">SP</th>
        <th style="{th3}">Status</th>
        <th style="{th3}">Group</th>
        {commitment_header}
      </tr>
    </thead>
    <tbody>{issue_rows}</tbody>
  </table>
</div>
""", unsafe_allow_html=True)


def render_sprint_view(jira_config: dict):
    board_id = _get_secret_value(OPTIONAL_BOARD_SECRET)

    if not board_id:
        st.markdown("""
<div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:12px;padding:28px;text-align:center;max-width:480px;margin:32px auto">
  <div style="font-size:36px;margin-bottom:12px">🏃</div>
  <div style="font-size:16px;font-weight:700;color:#0f172a;margin-bottom:8px">Sprint View not configured</div>
  <div style="font-size:13px;color:#64748b;margin-bottom:16px">Add your Jira board ID to <code>~/.streamlit/secrets.toml</code> to enable this view.</div>
  <code style="background:#f1f5f9;padding:6px 14px;border-radius:6px;font-size:13px;color:#4f46e5">JIRA_BOARD_ID = "473"</code>
</div>
""", unsafe_allow_html=True)
        return

    jira_domain = jira_config["JIRA_DOMAIN"]
    email = jira_config["JIRA_EMAIL"]
    token = jira_config["JIRA_API_TOKEN"]

    DIVIDER = "<div style='height:1px;background:linear-gradient(90deg,#6366f1,transparent);margin:20px 0'></div>"
    ist = pytz.timezone("Asia/Kolkata")
    today = datetime.now(ist).date()

    sprint_list = load_sprint_list(jira_domain, email, token, board_id)

    if not sprint_list:
        st.info("No sprints found for this board.")
        return

    def _sprint_label(s):
        name = s.get("name", "Sprint")
        state = s.get("state", "")
        try:
            end_dt = datetime.strptime(s["endDate"][:10], "%Y-%m-%d").date() if s.get("endDate") else None
            start_dt = datetime.strptime(s["startDate"][:10], "%Y-%m-%d").date() if s.get("startDate") else None
        except ValueError:
            start_dt = end_dt = None
        if state == "active":
            if end_dt:
                days_left = (end_dt - today).days
                if days_left < 0:
                    suffix = f"Active · {abs(days_left)}d overdue"
                elif days_left == 0:
                    suffix = "Active · last day"
                else:
                    suffix = f"Active · {days_left}d left"
            else:
                suffix = "Active"
            return f"▶ {name}  ({suffix})"
        if state == "future":
            if start_dt:
                days_to_start = (start_dt - today).days
                if days_to_start >= 0:
                    return f"◔ {name}  (Upcoming · starts in {days_to_start}d)"
            return f"◔ {name}  (Upcoming)"
        if start_dt and end_dt:
            return f"{name}  ({start_dt.strftime('%d %b')} – {end_dt.strftime('%d %b %Y')})"
        return name

    labels = [_sprint_label(s) for s in sprint_list]
    default_idx = next((i for i, s in enumerate(sprint_list) if s.get("state") == "active"), 0)

    selected_label = st.selectbox(
            "Sprint", labels, index=default_idx,
            label_visibility="collapsed", key="sprint_selector",
        )

    selected_idx = labels.index(selected_label)
    selected_sprint = sprint_list[selected_idx]
    sprint_id = selected_sprint["id"]
    sprint_name = selected_sprint["name"]
    sprint_state = selected_sprint.get("state", "closed")
    # Previous sprint is the next item in the list (list is sorted newest-first)
    prev_sprint = sprint_list[selected_idx + 1] if selected_idx + 1 < len(sprint_list) else None

    try:
        start_dt = datetime.strptime(selected_sprint["startDate"][:10], "%Y-%m-%d").date() if selected_sprint.get("startDate") else None
        end_dt = datetime.strptime(selected_sprint["endDate"][:10], "%Y-%m-%d").date() if selected_sprint.get("endDate") else None
    except ValueError:
        start_dt = end_dt = None

    date_range_str = f"{start_dt.strftime('%d %b')} – {end_dt.strftime('%d %b %Y')}" if (start_dt and end_dt) else ""

    # State badge — visual pill on the right of row 1
    if sprint_state == "active":
        state_badge = '<span style="background:#dcfce7;color:#16a34a;padding:3px 10px;border-radius:20px;font-size:12px;font-weight:700;white-space:nowrap">● ACTIVE</span>'
    elif sprint_state == "future":
        state_badge = '<span style="background:#eef2ff;color:#6366f1;padding:3px 10px;border-radius:20px;font-size:12px;font-weight:700;white-space:nowrap">◔ UPCOMING</span>'
    else:
        state_badge = '<span style="background:#f1f5f9;color:#475569;padding:3px 10px;border-radius:20px;font-size:12px;font-weight:700;white-space:nowrap">✓ CLOSED</span>'

    # Days info — plain text (with optional accent color) for row 2
    days_text = ""
    days_color = None
    if sprint_state == "active" and end_dt:
        days_left = (end_dt - today).days
        if days_left < 0:
            days_text = f'⚠ {abs(days_left)} day{"s" if abs(days_left) != 1 else ""} overdue'
            days_color = "#dc2626"
        elif days_left == 0:
            days_text = "Last day"
            days_color = "#f59e0b"
        else:
            days_text = f'{days_left} day{"s" if days_left != 1 else ""} remaining'
    elif sprint_state == "future" and start_dt:
        days_to_start = (start_dt - today).days
        if days_to_start > 0:
            days_text = f'starts in {days_to_start} day{"s" if days_to_start != 1 else ""}'
        elif days_to_start == 0:
            days_text = "starts today"
            days_color = "#f59e0b"

    # Load and classify issues before rendering the hero so active/future sprint
    # stats use developer-actionable commitment, not TEST carry-forward.
    with st.spinner(f"Loading issues for {sprint_name}…"):
        issues_df = load_sprint_issues(jira_domain, email, token, sprint_id)

    prev_sprint_name = None
    carry_forward_keys = set()
    if not issues_df.empty:
        # For a closed sprint, mid-sprint statuses (In Progress, Test, etc.) are
        # misleading — those tickets were dropped to backlog unfinished. Collapse
        # everything non-Done into a single "Not Delivered" bucket.
        if sprint_state == "closed":
            issues_df = issues_df.copy()
            issues_df["Status Group"] = issues_df["Status Group"].apply(
                lambda g: g if g == "Done" else "Not Delivered"
            )

        # Carry-forward detection: issues that also appeared in the previous sprint.
        if prev_sprint:
            prev_sprint_name = prev_sprint.get("name", "")
            prev_df = load_sprint_issues(jira_domain, email, token, prev_sprint["id"])
            if not prev_df.empty:
                carry_forward_keys = set(prev_df["Key"]) & set(issues_df["Key"])

        issues_df = add_commitment_classification(issues_df, carry_forward_keys)

    commitment_summary = _commitment_summary(issues_df)

    if issues_df.empty:
        total_issues, active_devs_count, committed_sp_total = 0, 0, 0.0
        hero_sp_label = "SP committed"
    else:
        total_issues = len(issues_df)
        if sprint_state == "closed":
            active_devs_count = sum(
                1 for d in issues_df["Developer"].dropna().unique() if d != "(Unassigned)"
            )
            committed_sp_total = float(issues_df["Story Points"].sum())
            hero_sp_label = "SP in sprint"
        else:
            actionable_df = issues_df[issues_df["Is Actionable Commitment"]]
            active_devs_count = sum(
                1 for d in actionable_df["Developer"].dropna().unique() if d != "(Unassigned)"
            )
            committed_sp_total = commitment_summary["actionable_sp"]
            hero_sp_label = "SP committed"

    # Hero card — three-row layout with a state-color left accent. Tabs follow
    # only when there's actual content to lens (not closed, not empty).
    has_tabs = sprint_state != "closed" and not issues_df.empty
    accent_color = _sprint_state_accent(sprint_state)
    card_modifier = "sprint-hero-card-with-tabs" if has_tabs else "sprint-hero-card-standalone"

    row2_parts = []
    if date_range_str:
        row2_parts.append(_html.escape(date_range_str))
    if days_text:
        if days_color:
            row2_parts.append(
                f'<span style="color:{days_color};font-weight:600">{_html.escape(days_text)}</span>'
            )
        else:
            row2_parts.append(_html.escape(days_text))
    row2_html = " · ".join(row2_parts) if row2_parts else "&nbsp;"

    issue_word = "issue" if total_issues == 1 else "issues"
    dev_word = "dev" if active_devs_count == 1 else "devs"
    row3_html = (
        f'{total_issues} {issue_word} · {active_devs_count} {dev_word} · '
        f'{committed_sp_total:.1f} {hero_sp_label}'
    )

    meta_html = f'<div class="hero-meta">{row2_html}</div>' if row2_parts else ""
    stats_html = f'<div class="hero-stats">{row3_html}</div>' if not issues_df.empty else ""

    st.markdown(f"""
<div class="sprint-hero-card {card_modifier}" style="border-left-color:{accent_color}">
  <div class="hero-row">
    <div class="hero-name-group">
      <span class="hero-name">{_html.escape(sprint_name)}</span>
      {state_badge}
    </div>
    {meta_html}
    {stats_html}
  </div>
</div>
""", unsafe_allow_html=True)

    if issues_df.empty:
        st.info("No issues found for this sprint.")
        return

    fresh_sp = commitment_summary["fresh_sp"]
    cf_count = commitment_summary["cf_count"]
    fresh_count = commitment_summary["fresh_count"]

    # Lens choice — Planning is meaningless for closed sprints, so the toggle
    # is hidden in that case and Execution renders directly. Active sprints
    # default to Planning during the first 3 days of the window; future
    # sprints always default to Planning.
    if sprint_state == "closed":
        active_lens = "Execution"
    else:
        default_lens = _default_sprint_tab(sprint_state, start_dt, today)
        active_lens = _render_sprint_lens_tabs(sprint_id, default_lens)

    if active_lens == "Planning":
        render_sprint_planning_section(
            issues_df, sprint_name, prev_sprint, jira_domain, DIVIDER,
        )
    else:
        render_sprint_execution_section(
            issues_df, prev_sprint, prev_sprint_name,
            cf_count, fresh_count, fresh_sp, jira_domain, DIVIDER,
        )


# =====================
# Main
# =====================

def main():
    st.set_page_config(
        page_title="MedLern Product Team Metrics",
        page_icon="📊",
        layout="wide",
    )

    _inject_css()

    ist = pytz.timezone("Asia/Kolkata")
    today = datetime.now(ist).date()

    # Global auto-refresh — fires every 30 min; cache TTLs handle actual re-fetch
    st_autorefresh(interval=30 * 60 * 1000, key="global_autorefresh")

    # ── Header ──────────────────────────────────────────────────────────
    _last_refreshed = st.session_state.get("_last_refreshed")
    _refresh_label = (
        _last_refreshed.strftime("%d %b · %H:%M IST")
        if _last_refreshed else "—"
    )
    _hdr_left, _hdr_btn = st.columns([6, 1])
    with _hdr_left:
        st.markdown("""
<div style="display:flex;align-items:center;gap:14px;padding:16px 0 14px">
  <div style="width:38px;height:38px;background:linear-gradient(135deg,#6366f1,#8b5cf6);
              border-radius:9px;display:flex;align-items:center;justify-content:center;
              font-size:18px;flex-shrink:0;box-shadow:0 2px 8px rgba(99,102,241,.3)">📊</div>
  <div>
    <div style="font-size:20px;font-weight:700;color:#0f172a;letter-spacing:-0.4px;
                line-height:1.2">MedLern Product Team Metrics</div>
    <div style="font-size:11px;color:#94a3b8;margin-top:2px;font-weight:500;
                letter-spacing:0.04em;text-transform:uppercase">
      Engineering metrics &amp; quality tracking
    </div>
  </div>
</div>
""", unsafe_allow_html=True)
    with _hdr_btn:
        st.markdown("""
<style>
.st-key-_global_refresh { margin-top: 16px !important; }
.st-key-_global_refresh button {
    white-space: nowrap !important;
    font-size: 12px !important;
    padding: 6px 14px !important;
    line-height: 1.4 !important;
}
</style>
""", unsafe_allow_html=True)
        if st.button(
            f"↻ Refresh all  ·  {_refresh_label}",
            key="_global_refresh",
            use_container_width=True,
        ):
            st.cache_data.clear()
            st.session_state.pop("_data_prewarmed", None)
            st.session_state.pop("_last_refreshed", None)
            st.rerun()
    st.markdown(
        "<div style='height:1px;background:#e2e8f0;margin:0 0 4px'></div>",
        unsafe_allow_html=True,
    )

    missing_keys, jira_config = validate_jira_config()
    if missing_keys:
        render_missing_config(missing_keys)
        st.stop()

    # ── Parallel data pre-warm ───────────────────────────────────────────
    # All three mode data sources are loaded concurrently so that switching
    # between modes is instant (subsequent calls are @st.cache_data hits).
    _board_id = _get_secret_value(OPTIONAL_BOARD_SECRET)

    def _prewarm_analytics():
        load_jira_data(jira_config["JIRA_DOMAIN"], jira_config["JIRA_EMAIL"],
                       jira_config["JIRA_API_TOKEN"], jira_config["JIRA_FILTER_ID"])
        load_bug_data(jira_config["JIRA_DOMAIN"], jira_config["JIRA_EMAIL"],
                      jira_config["JIRA_API_TOKEN"], jira_config[OPTIONAL_BUG_FILTER_SECRET])

    def _prewarm_defect():
        from defect_sla_dashboard import IST as _defect_ist
        from defect_sla_dashboard import build_derived as _build_derived
        from defect_sla_dashboard import load_defect_data as _load_defect_data

        raw_df, _ = _load_defect_data()
        if not raw_df.empty:
            _now_min = datetime.now(_defect_ist).replace(second=0, microsecond=0)
            _build_derived(raw_df, _now_min)

    def _prewarm_sprint():
        if _board_id:
            load_sprint_list(jira_config["JIRA_DOMAIN"], jira_config["JIRA_EMAIL"],
                             jira_config["JIRA_API_TOKEN"], _board_id)

    if not st.session_state.get("_data_prewarmed"):
        with st.spinner("Loading dashboard data…"):
            with ThreadPoolExecutor(max_workers=3) as _pool:
                _futures = {
                    _pool.submit(_prewarm_analytics): "Team Overview",
                    _pool.submit(_prewarm_defect): "Client Issues",
                    _pool.submit(_prewarm_sprint): "Sprint Tracker",
                }
                for _fut in as_completed(_futures):
                    try:
                        _fut.result()
                    except Exception as _e:
                        st.warning(f"{_futures[_fut]} data load warning: {_e}")
        st.session_state["_data_prewarmed"] = True
        st.session_state["_last_refreshed"] = datetime.now(ist)
        st.rerun()  # re-render header so the timestamp appears immediately

    # ── Mode toggle ─────────────────────────────────────────────────────
    VIEW_MODES = ["📊 Team Overview", "🏃 Sprint Tracker", "🚨 Client Issues"]
    if "view_mode" not in st.session_state:
        st.session_state["view_mode"] = "📊 Team Overview"

    _active_idx = VIEW_MODES.index(st.session_state["view_mode"])
    st.markdown(f"""
<style>
/* Underline tab nav — scoped to these three button keys */
.st-key-_nav_0 button, .st-key-_nav_1 button, .st-key-_nav_2 button {{
    background: transparent !important;
    border: none !important;
    border-bottom: 3px solid transparent !important;
    border-radius: 0 !important;
    color: #94a3b8 !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    padding: 10px 8px 11px !important;
    box-shadow: none !important;
    width: 100% !important;
    letter-spacing: 0.01em !important;
    transition: color 0.15s ease, border-color 0.15s ease !important;
}}
.st-key-_nav_0 button:hover, .st-key-_nav_1 button:hover, .st-key-_nav_2 button:hover {{
    background: transparent !important;
    color: #6366f1 !important;
    border-bottom: 3px solid #c7d2fe !important;
    box-shadow: none !important;
}}
.st-key-_nav_{_active_idx} button {{
    color: #6366f1 !important;
    border-bottom: 3px solid #6366f1 !important;
    font-weight: 600 !important;
}}
</style>
""", unsafe_allow_html=True)

    _nav_cols = st.columns(3)
    for _i, (_col, _mode) in enumerate(zip(_nav_cols, VIEW_MODES)):
        with _col:
            if st.button(_mode, key=f"_nav_{_i}", use_container_width=True):
                st.session_state["view_mode"] = _mode
                st.rerun()

    view_mode = st.session_state["view_mode"]
    st.markdown(
        "<div style='height:1px;background:#e2e8f0;margin:-4px 0 20px'></div>",
        unsafe_allow_html=True,
    )

    if view_mode == "🏃 Sprint Tracker":
        render_sprint_view(jira_config)
        return

    if view_mode == "🚨 Client Issues":
        from defect_sla_dashboard import render_defect_sla
        render_defect_sla(inject_base_css=False)
        return

    # ── Analytics mode ──────────────────────────────────────────────────
    with st.container():
        period_types = [
            "Current Week", "Current Month", "Current Quarter", "Current Year",
            "Last Month", "Last Quarter", "Custom",
        ]
        def _period_label(pt):
            if pt == "Custom":
                return "Custom"
            s, e, _, _ = get_period_bounds(pt, today)
            return f"{pt}  ({s.strftime('%d %b')} – {e.strftime('%d %b %Y')})"

        period_labels = [_period_label(pt) for pt in period_types]

        st.markdown(
            "<div style='margin-top:12px'>"
            "<p style='font-size:11px;font-weight:600;color:#64748b;"
            "letter-spacing:.06em;text-transform:uppercase;margin-bottom:6px'>"
            "Period</p></div>",
            unsafe_allow_html=True,
        )
        _period_col, _ = st.columns([2, 3])
        with _period_col:
            selected_label = st.selectbox(
                "Period",
                period_labels,
                index=4,          # default: Last Month
                label_visibility="collapsed",
            )
            period_type = period_types[period_labels.index(selected_label)]

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
    # Data is guaranteed cached from the prewarm step — no spinner needed.
    df = load_jira_data(
        jira_config["JIRA_DOMAIN"],
        jira_config["JIRA_EMAIL"],
        jira_config["JIRA_API_TOKEN"],
        jira_config["JIRA_FILTER_ID"],
    )
    bugs_df = load_bug_data(
        jira_config["JIRA_DOMAIN"],
        jira_config["JIRA_EMAIL"],
        jira_config["JIRA_API_TOKEN"],
        jira_config[OPTIONAL_BUG_FILTER_SECRET],
    )

    if df.empty:
        st.warning("No task data loaded. Check your Jira credentials and filter ID.")
        st.stop()

    # ── Compute metrics ─────────────────────────────────────────────────
    curr_team, curr_dev = compute_metrics(df, bugs_df, start_date, end_date)
    prev_team, prev_dev = compute_metrics(df, bugs_df, prev_start, prev_end)

    DIVIDER = "<div style='height:1px;background:linear-gradient(90deg,#6366f1,transparent);margin:20px 0'></div>"

    _section_header("Team Overview", period_label)
    render_kpi_cards(curr_team, prev_team)

    st.markdown(DIVIDER, unsafe_allow_html=True)
    _section_header("Developer Breakdown", "Click a row for individual drill-down")
    render_dev_table(curr_dev, prev_dev, df, bugs_df, start_date, end_date)


if __name__ == "__main__":
    main()
