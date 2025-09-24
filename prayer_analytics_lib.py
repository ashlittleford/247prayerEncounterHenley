import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import timedelta
import matplotlib.dates as mdates

# -----------------------
# Core Functions
# -----------------------

def load_and_combine_csvs(file_paths):
    """
    Loads and combines data from a list of CSV file paths.
    """
    dfs = []
    for path in file_paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            # Handle different date formats based on file
            if "initial.csv" in path:
                df["start_time"] = pd.to_datetime(df["start_time"], format="%d/%m/%Y %H:%M", errors="coerce")
                df["end_time"] = pd.to_datetime(df["end_time"], format="%d/%m/%Y %H:%M", errors="coerce")
            else:
                df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce")
                df["end_time"] = pd.to_datetime(df["end_time"], errors="coerce")
            dfs.append(df)
        else:
            print(f"Warning: File not found at {path}. Skipping.")
    if not dfs:
        raise FileNotFoundError("No CSV files found from the provided paths.")
    df = pd.concat(dfs, ignore_index=True)
    return df

def normalize_columns(df):
    """Normalizes column names and ensures required columns exist."""
    rename_map = {"phone": "mobile"}
    df = df.rename(columns=rename_map)
    for col in ["firstname", "lastname", "email", "mobile", "start_time", "end_time"]:
        if col not in df.columns:
            df[col] = np.nan
    return df

def make_person_key(row, email_map):
    """Creates a unique identifier for each person based on a new priority order: email, then full name, then mobile+first3."""
    first = str(row["firstname"] or "").strip().lower()
    last = str(row["lastname"] or "").strip().lower()
    email = str(row["email"] or "").replace(" ", "").strip().lower()
    mobile = str(row["mobile"] or "").replace(" ", "").strip()
    
    # Use the email map to standardize the email address
    if email in email_map:
        email = email_map[email]

    # New Priority 1: Use email + first 3 letters of first name
    if email and first:
        return f"{email}-{first[:3]}"
    # Fallback to Priority 2: Full name
    if first and last:
        return f"{first} {last}"
    # Fallback to Priority 3: Mobile number
    if mobile:
        return f"{mobile}"
    return "UNKNOWN"

def calculate_hours(df):
    """Calculates the duration in hours for each entry."""
    df["duration"] = (df["end_time"] - df["start_time"]).dt.total_seconds() / 3600.0
    return df

def check_short_sessions(df):
    """Prints a summary of all sessions with a duration of less than 1 hour."""
    short_sessions = df[df["duration"] < 1]
    if not short_sessions.empty:
        print("\n--- Sessions less than 1 hour found ---")
        print(short_sessions[["person_name", "start_time", "duration"]])
    else:
        print("\n--- No sessions less than 1 hour were found in the data. ---")

# -----------------------
# Dashboard Metrics
# -----------------------

def calculate_dashboard_metrics(df, df_monthly_total):
    """Calculates all dashboard metrics and returns them in a list of dictionaries."""
    metrics = []

    # --- OVERVIEW ---
    total_hours = df["duration"].sum()
    total_users = df["person_key"].nunique()
    
    df["weekday"] = df["start_time"].dt.day_name()
    df["hour"] = df["start_time"].dt.hour
    popular_slots = df.groupby(["weekday", "hour"]).size().sort_values(ascending=False).head(3)

    # Calculate time periods based on start of data to today
    start_date = df["start_time"].min().date()
    today = pd.Timestamp.now().normalize().date()
    total_days = (today - start_date).days + 1
    total_weeks = total_days / 7
    total_years = total_days / 365.25

    avg_hours_day = total_hours / total_days
    avg_hours_week = total_hours / total_weeks
    avg_hours_year = total_hours / total_years

    metrics.append({"Metric": "Total Hours of Prayer", "Value": f"{total_hours:.2f}"})
    metrics.append({"Metric": "Total Individual Users", "Value": f"{total_users}"})
    for i, ((weekday, hour), count) in enumerate(popular_slots.items(), 1):
        metrics.append({"Metric": f"Top {i} Timeslot", "Value": f"{weekday} at {hour:02d}:00 with {count} bookings"})
    metrics.append({"Metric": "Avg Hours per Day", "Value": f"{avg_hours_day:.2f}"})
    metrics.append({"Metric": "Avg Hours per Week", "Value": f"{avg_hours_week:.2f}"})
    metrics.append({"Metric": "Avg Hours per Year", "Value": f"{avg_hours_year:.2f}"})

    # --- RECENTLY ---
    today_ts = pd.Timestamp.now().normalize()
    last_7_days_start = today_ts - timedelta(days=7)
    prev_7_days_start = today_ts - timedelta(days=14)
    
    df_last_7 = df[(df["start_time"] >= last_7_days_start) & (df["start_time"] < today_ts)]
    df_prev_7 = df[(df["start_time"] >= prev_7_days_start) & (df["start_time"] < last_7_days_start)]
    
    hours_last_7 = df_last_7["duration"].sum()
    hours_prev_7 = df_prev_7["duration"].sum()
    
    total_possible_hours_7 = 7 * 24
    pct_last_7 = (hours_last_7 / total_possible_hours_7) * 100 if total_possible_hours_7 > 0 else 0
    pct_prev_7 = (hours_prev_7 / total_possible_hours_7) * 100 if total_possible_hours_7 > 0 else 0

    last_month_end = today_ts.to_period('M').to_timestamp()
    last_month_start = last_month_end - pd.offsets.MonthBegin(1)
    prev_month_end = last_month_start - pd.offsets.Day(1)
    prev_month_start = prev_month_end.to_period('M').to_timestamp()
    
    hours_last_month = df_monthly_total.iloc[-1] if not df_monthly_total.empty and len(df_monthly_total) >= 1 else 0
    hours_prev_month = df_monthly_total.iloc[-2] if len(df_monthly_total) >= 2 else 0

    total_possible_hours_last_month = (last_month_end - last_month_start).days * 24
    total_possible_hours_prev_month = (prev_month_end - prev_month_start).days * 24

    pct_last_month = (hours_last_month / total_possible_hours_last_month) * 100 if total_possible_hours_last_month > 0 else 0
    pct_prev_month = (hours_prev_month / total_possible_hours_prev_month) * 100 if total_possible_hours_prev_month > 0 else 0
    
    metrics.append({"Metric": "Total Hours Last 7 Days", "Value": f"{hours_last_7:.2f}"})
    metrics.append({"Metric": "Total Hours Previous 7 Days", "Value": f"{hours_prev_7:.2f}"})
    metrics.append({"Metric": "Last 7 Days Hours Filled (%)", "Value": f"{pct_last_7:.2f}%"})
    metrics.append({"Metric": "Previous 7 Days Hours Filled (%)", "Value": f"{pct_prev_7:.2f}%"})
    metrics.append({"Metric": "Total Hours Last Month", "Value": f"{hours_last_month:.2f}"})
    metrics.append({"Metric": "Total Hours Previous Month", "Value": f"{hours_prev_month:.2f}"})
    metrics.append({"Metric": "Last Month Hours Filled (%)", "Value": f"{pct_last_month:.2f}%"})
    metrics.append({"Metric": "Previous Month Hours Filled (%)", "Value": f"{pct_prev_month:.2f}%"})
    
    return metrics

def export_user_summary_to_csv(df, outdir):
    """Exports a summary of users with their weekly and fortnightly average hours,
    calculated by dividing their total hours by the number of weeks/fortnights
    in the total data period."""
    
    start_date = df["start_time"].min().date()
    today = pd.Timestamp.now().normalize().date()
    total_days = (today - start_date).days + 1
    total_weeks = total_days / 7
    total_fortnights = total_days / 14
    
    # Get the most common person_name for each person_key
    most_common_names = df.groupby('person_key')['person_name'].apply(lambda x: x.mode()[0] if not x.mode().empty else 'UNKNOWN').reset_index(name='person_name')
    
    user_summary = df.groupby("person_key").agg(
        total_hours=("duration", "sum"),
    ).reset_index()

    # Merge the most common names back into the summary
    user_summary = user_summary.merge(most_common_names, on='person_key', how='left')

    user_summary["weekly_avg_hours"] = user_summary["total_hours"] / total_weeks if total_weeks > 0 else 0
    user_summary["fortnightly_avg_hours"] = user_summary["total_hours"] / total_fortnights if total_fortnights > 0 else 0
    
    # Format to 2 decimal places before saving
    user_summary[["weekly_avg_hours", "fortnightly_avg_hours"]] = user_summary[["weekly_avg_hours", "fortnightly_avg_hours"]].round(2)
    
    user_summary[["person_key", "person_name", "weekly_avg_hours", "fortnightly_avg_hours"]].to_csv(os.path.join(outdir, "user.csv"), index=False)
    
    return user_summary

def calculate_regulars_metrics(df, user_summary, outdir):
    """Identifies regulars, exports their lists, and prints metrics."""
    # Define regulars based on new logic
    weekly_regulars = user_summary[user_summary["weekly_avg_hours"] >= 1]["person_key"].tolist()
    fortnightly_regulars = user_summary[user_summary["fortnightly_avg_hours"] >= 1]["person_key"].tolist()
    
    # Calculate percentages
    total_hours = df["duration"].sum()
    pct_weekly_hours = (df[df["person_key"].isin(weekly_regulars)]["duration"].sum() / total_hours) * 100 if total_hours > 0 else 0
    pct_fortnightly_hours = (df[df["person_key"].isin(fortnightly_regulars)]["duration"].sum() / total_hours) * 100 if total_hours > 0 else 0
    
    # Export CSVs with formatted values
    user_summary[user_summary["person_key"].isin(weekly_regulars)].round(2).to_csv(os.path.join(outdir, "weekly_regulars.csv"), index=False)
    user_summary[user_summary["person_key"].isin(fortnightly_regulars)].round(2).to_csv(os.path.join(outdir, "fortnightly_regulars.csv"), index=False)
    
    metrics = []
    metrics.append({"Metric": "Number of Weekly Regulars", "Value": len(weekly_regulars)})
    metrics.append({"Metric": "Total Hours by Weekly Regulars (%)", "Value": f"{pct_weekly_hours:.2f}%"})
    metrics.append({"Metric": "Number of Fortnightly Regulars", "Value": f"{len(fortnightly_regulars)}"})
    metrics.append({"Metric": "Total Hours by Fortnightly Regulars (%)", "Value": f"{pct_fortnightly_hours:.2f}%"})
    return metrics

def export_top_10_slots(df, outdir):
    """Exports the top 10 most booked prayer slots to a CSV file."""
    df["weekday"] = df["start_time"].dt.day_name()
    df["hour"] = df["start_time"].dt.hour
    
    top_10 = df.groupby(["weekday", "hour"]).size().reset_index(name="bookings").sort_values("bookings", ascending=False).head(10)
    top_10.to_csv(os.path.join(outdir, "top_10_slots.csv"), index=False)

def plot_hourly_distribution(df, outdir):
    """Plots the hourly distribution of total prayer hours."""
    df["hour"] = df["start_time"].dt.hour
    per_hour = df.groupby("hour")["duration"].sum()
    per_hour_pct = per_hour / per_hour.sum() * 100 if per_hour.sum() > 0 else per_hour

    plt.figure(figsize=(10, 5))
    plt.bar(per_hour_pct.index, per_hour_pct.values, color="skyblue")
    plt.title("Hourly Distribution (% of Total Hours)")
    plt.xlabel("Hour of Day")
    plt.ylabel("Percentage (%)")
    plt.xticks(range(24), [f"{h:02d}:00" for h in range(24)], rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "hourly_distribution.png"))
    plt.close()

def plot_binary_hourly_distribution(df, outdir):
    """Counts each user only once per hour block"""
    df["hour"] = df["start_time"].dt.hour
    df_unique_user_hour = df.drop_duplicates(subset=["person_key", "hour"])
    per_hour = df_unique_user_hour.groupby("hour")["person_key"].count()

    plt.figure(figsize=(10, 5))
    plt.bar(per_hour.index, per_hour.values, color="skyblue")
    plt.title("Hourly Distribution (Unique Users per Hour)")
    plt.xlabel("Hour of Day")
    plt.ylabel("Number of Unique Users")
    plt.xticks(range(24), [f"{h:02d}:00" for h in range(24)], rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "hourly_users.png"))
    plt.close()

def plot_weekly_total_hours(df, outdir):
    """Creates a line graph of the weekly total hours."""
    df_weekly_total = df.groupby(pd.Grouper(key="start_time", freq="W"))["duration"].sum().reset_index()
    
    plt.figure(figsize=(10, 5))
    plt.plot(df_weekly_total["start_time"], df_weekly_total["duration"], marker='o')
    plt.title("Total Hours Logged per Week")
    plt.xlabel("Week Starting")
    plt.ylabel("Total Hours")
    plt.grid(True)
    
    ax = plt.gca()
    # Set the x-axis to show a label for every second week to improve readability
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d, %Y'))
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "weekly_total_hours.png"))
    plt.close()

def plot_monthly_total_hours(df_monthly_total, outdir):
    """Creates a line graph of the monthly total hours."""
    df_monthly_total = df_monthly_total.reset_index()
    
    # Format the dates for the x-axis
    df_monthly_total["month"] = df_monthly_total["month"].dt.strftime('%b %Y')
    
    plt.figure(figsize=(10, 5))
    plt.plot(df_monthly_total["month"], df_monthly_total["duration"], marker='o')
    plt.title("Total Hours Logged per Month")
    plt.xlabel("Month")
    plt.ylabel("Total Hours")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "monthly_total_hours.png"))
    plt.close()

def plot_avg_hours_distribution(user_summary, outdir):
    """Plots the weekly average hours for each user on a bar chart."""
    
    # Sort users by weekly average for a meaningful plot
    weekly_sorted = user_summary.sort_values("weekly_avg_hours", ascending=False)
    
    fig, ax = plt.subplots(figsize=(15, 7))
    
    # Weekly Average Bar Chart
    weekly_sorted.plot(x="person_name", y="weekly_avg_hours", kind="bar", ax=ax, color='skyblue', legend=False)
    ax.set_title("Weekly Average Hours per User")
    ax.set_xlabel("User")
    ax.set_ylabel("Weekly Average Hours")
    ax.tick_params(axis='x', rotation=90)
    ax.grid(axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "avg_hours_distribution.png"))
    plt.close()

def run_analysis(input_files, outdir):
    """Orchestrates the entire analysis process."""
    os.makedirs(outdir, exist_ok=True)
    
    df = load_and_combine_csvs(input_files)
    df = normalize_columns(df)
    
    today = pd.Timestamp.now().normalize()
    df = df[df["start_time"] <= today]
    df = df.dropna(subset=["start_time", "end_time"])

    # Load email duplicates and create a lookup map
    email_dupes_path = "email_duplicates.csv"
    email_map = {}
    if os.path.exists(email_dupes_path):
        email_dupes_df = pd.read_csv(email_dupes_path)
        for _, row in email_dupes_df.iterrows():
            primary_email = str(row["primary"] or "").strip().lower()
            additional_email = str(row["additional"] or "").strip().lower()
            if primary_email and additional_email:
                email_map[additional_email] = primary_email

    df["person_key"] = df.apply(make_person_key, axis=1, args=(email_map,))
    df["person_name"] = df.apply(
        lambda r: "UNKNOWN"
        if r["person_key"] == "UNKNOWN"
        else f"{str(r['firstname'] or '').strip()} {str(r['lastname'] or '').strip()}".strip(),
        axis=1,
    )
    df = calculate_hours(df)
    
    check_short_sessions(df)
    
    df["month"] = df["start_time"].dt.to_period('M')
    df_monthly_total = df.groupby("month")["duration"].sum()

    metrics = calculate_dashboard_metrics(df, df_monthly_total)
    
    user_summary = export_user_summary_to_csv(df, outdir)
    regular_metrics = calculate_regulars_metrics(df, user_summary, outdir)
    
    all_metrics = metrics + regular_metrics
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(os.path.join(outdir, "dashboard_summary.csv"), index=False)
    
    print("Dashboard summary has been saved to dashboard_summary.csv.")
    print("User summary with weekly and fortnightly averages has been saved to user.csv.")
    print("Top 10 most booked slots have been exported to top_10_slots.csv.")

    export_top_10_slots(df, outdir)
    plot_hourly_distribution(df, outdir)
    plot_binary_hourly_distribution(df, outdir)
    plot_weekly_total_hours(df, outdir)
    plot_monthly_total_hours(df_monthly_total, outdir)
    plot_avg_hours_distribution(user_summary, outdir)
    
    print(f"All dashboard graphs have been saved to the '{outdir}' folder.")