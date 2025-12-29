import pandas as pd
import numpy as np
import os
import json
import subprocess
import prayer_analytics_lib as pal
from datetime import timedelta

# --- Helper Functions ---

def commit_and_push_changes(filepath, commit_message):
    """
    Commits and pushes a specific file to the GitHub repository.
    Requires 'GITHUB_TOKEN' to be set in environment variables.
    """
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        return False, "GITHUB_TOKEN not found in environment variables. Changes saved locally only."

    # Get current remote url
    try:
        remote_url_bytes = subprocess.check_output(["git", "remote", "get-url", "origin"])
        remote_url = remote_url_bytes.decode("utf-8").strip()
    except Exception as e:
        return False, f"Could not determine git remote URL: {str(e)}"

    # Inject token into URL for authentication
    if "https://" in remote_url:
        repo_url = remote_url.replace("https://", f"https://{token}@")
    else:
        return False, "Git remote URL format not supported (requires https)."

    try:
        # Configure git user (local to this run)
        subprocess.run(["git", "config", "user.email", "flask-bot@example.com"], check=False)
        subprocess.run(["git", "config", "user.name", "Flask Bot"], check=False)

        # Add file
        subprocess.run(["git", "add", filepath], check=True, capture_output=True, text=True)

        # Commit
        status = subprocess.run(["git", "status", "--porcelain", filepath], capture_output=True, text=True)
        if not status.stdout.strip():
            return True, "No changes to commit."

        subprocess.run(["git", "commit", "-m", commit_message], check=True, capture_output=True, text=True)

        # Pull (Rebase)
        subprocess.run(["git", "pull", "--rebase", repo_url, "main"], check=True, capture_output=True, text=True)

        # Push
        subprocess.run(["git", "push", repo_url, "HEAD:main"], check=True, capture_output=True, text=True)

        return True, "Changes committed and pushed to GitHub successfully!"
    except subprocess.CalledProcessError as e:
        return False, f"Git error: {e.stderr if e.stderr else str(e)}"
    except Exception as e:
        return False, f"Error: {str(e)}"

def format_date_custom(d):
    """Formats date as '24th May 25'."""
    day = d.day
    suffix = "th" if 11 <= day <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
    return f"{day}{suffix} {d.strftime('%B %y')}"

def load_pray_days_config(filepath="pray_days.json"):
    if os.path.exists(filepath):
        try:
            with open(filepath, "r") as f:
                return json.load(f)
        except:
            return []
    return []

def save_pray_days_config(data, filepath="pray_days.json"):
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)
    # Sync to Git
    return commit_and_push_changes(filepath, f"Update {filepath}: Modified Pray Days list")

# --- Core Analysis Logic ---

def run_full_analysis(input_files, outdir, start_date, end_date, goal_percentage):
    """
    Executes the analysis logic from prayer_analytics_lib.py.
    """

    # 1. LOAD, NORMALIZE, AND PREPARE UNFILTERED DATA
    df_unfiltered = pal.load_and_combine_csvs(input_files)
    df_unfiltered = pal.normalize_columns(df_unfiltered)

    today = pd.Timestamp.now().normalize()
    data_min_date = df_unfiltered["start_time"].min().normalize()

    # Load email duplicates
    email_dupes_path = "email_duplicates.csv"
    email_map = {}
    if os.path.exists(email_dupes_path):
        email_dupes_df = pd.read_csv(email_dupes_path)
        for _, row in email_dupes_df.iterrows():
            primary_email = str(row["primary"] or "").strip().lower()
            additional_email = str(row["additional"] or "").strip().lower()
            if primary_email and additional_email:
                email_map[additional_email] = primary_email

    # Load person merges
    merge_map = {}
    person_merges_path = "person_merges.csv"
    if os.path.exists(person_merges_path):
        try:
            pm_df = pd.read_csv(person_merges_path)
            for _, row in pm_df.iterrows():
                 src = str(row["source_key"] or "").strip()
                 tgt = str(row["target_key"] or "").strip()
                 if src and tgt:
                     merge_map[src] = tgt
        except Exception as e:
            print(f"Error loading person_merges.csv: {e}")

    # Apply keys and calculate hours on the UNFILTERED data
    df_unfiltered["person_key"] = df_unfiltered.apply(pal.make_person_key, axis=1, args=(email_map,))

    # Apply merges
    if merge_map:
        df_unfiltered["person_key"] = df_unfiltered["person_key"].replace(merge_map)

    # Re-apply keys logic if needed or just trust the merge
    # (Original code re-applied make_person_key then applied merges again, likely redundancy but we follow strict logic)
    # Actually original code:
    # 1. make_person_key
    # 2. replace with merge_map
    # 3. make_person_key (This looks redundant in original code, I will skip the second make_person_key unless it updates based on merged data which it doesn't really)
    # 4. replace with merge_map again.
    # I will stick to: Make Key -> Merge.

    df_unfiltered["person_name"] = df_unfiltered.apply(
        lambda r: "UNKNOWN"
        if r["person_key"] == "UNKNOWN"
        else f"{str(r['firstname'] or '').strip()} {str(r['lastname'] or '').strip()}".strip(),
        axis=1,
    )
    df_unfiltered = pal.calculate_hours(df_unfiltered)
    pal.check_short_sessions(df_unfiltered)

    # Calculate UNFILTERED recent stats
    h_7, u_7, c_7 = pal.calculate_time_period_stats(df_unfiltered, 7)
    h_28, u_28, c_28 = pal.calculate_time_period_stats(df_unfiltered, 28)
    recently_stats = {
        "H7": h_7, "U7": u_7, "C7": c_7,
        "H28": h_28, "U28": u_28, "C28": c_28,
    }

    # 2. APPLY FILTERING
    # Process date inputs
    # start_date and end_date are expected to be datetime.date objects or strings

    filter_start_date = pd.to_datetime(start_date).normalize() if start_date else data_min_date
    filter_end_date = pd.to_datetime(end_date).normalize() if end_date else today

    if filter_start_date > filter_end_date:
        filter_start_date, filter_end_date = filter_end_date, filter_start_date

    df = df_unfiltered[(df_unfiltered["start_time"].dt.normalize() >= filter_start_date) & (df_unfiltered["start_time"].dt.normalize() <= filter_end_date)].copy()

    if df.empty:
        # We can handle empty DF gracefully or raise
        # For web app, returning empty structs is better than crashing
        return None

    df = df[df["start_time"] <= today]
    df = df.dropna(subset=["start_time", "end_time"])
    df = df.dropna(subset=['start_time', 'end_time', 'person_key'])

    # 3. RUN CORE ANALYSIS
    df_monthly_total = df.groupby(pd.Grouper(key="start_time", freq="MS"))["duration"].sum()

    os.makedirs(outdir, exist_ok=True)

    metrics = pal.calculate_dashboard_metrics(df, df_monthly_total)
    user_summary = pal.export_user_summary_to_csv(df, outdir)
    regular_metrics = pal.calculate_regulars_metrics(df, user_summary, outdir)
    fortnightly_metrics = pal.calculate_fortnightly_regulars_metrics(df, user_summary, outdir)
    average_metrics = pal.calculate_average_metrics(df)

    all_metrics = metrics + regular_metrics + fortnightly_metrics + average_metrics
    metrics_df = pd.DataFrame(all_metrics)

    likelihood_df = pal.export_weekly_likelihood_metrics(df, outdir)
    total_sessions_count = len(df)

    return {
        "df": df,
        "df_monthly_total": df_monthly_total,
        "metrics_df": metrics_df,
        "user_summary": user_summary,
        "likelihood_df": likelihood_df,
        "recently_stats": recently_stats,
        "total_sessions_count": total_sessions_count,
        "df_unfiltered": df_unfiltered
    }
