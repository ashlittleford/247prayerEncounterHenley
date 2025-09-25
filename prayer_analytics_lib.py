import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import timedelta
import matplotlib.dates as mdates
import sys

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
                # Specific format for initial.csv (DD/MM/YYYY HH:MM)
                df["start_time"] = pd.to_datetime(df["start_time"], format="%d/%m/%Y %H:%M", errors="coerce")
                df["end_time"] = pd.to_datetime(df["end_time"], format="%d/%m/%Y %H:%M", errors="coerce")
            else:
                # Default inference for current.csv (YYYY-MM-DD HH:MM:SS)
                df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce")
                df["end_time"] = pd.to_datetime(df["end_time"], errors="coerce")
            dfs.append(df)
        # File not found warning is removed as requested
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
    """Removed all print statements for short session checks."""
    short_sessions = df[df["duration"] < 1]
    # No action/output if sessions are found or not found
    pass

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
    
    # 7-Day Periods
    last_7_days_start = today_ts - timedelta(days=7)
    prev_7_days_start = today_ts - timedelta(days=14)
    
    df_last_7 = df[(df["start_time"] >= last_7_days_start) & (df["start_time"] < today_ts)]
    df_prev_7 = df[(df["start_time"] >= prev_7_days_start) & (df["start_time"] < last_7_days_start)]
    
    hours_last_7 = df_last_7["duration"].sum()
    hours_prev_7 = df_prev_7["duration"].sum()
    users_last_7 = df_last_7["person_key"].nunique()
    users_prev_7 = df_prev_7["person_key"].nunique()
    
    total_possible_hours_7 = 7 * 24
    pct_last_7 = (hours_last_7 / total_possible_hours_7) * 100 if total_possible_hours_7 > 0 else 0
    pct_prev_7 = (hours_prev_7 / total_possible_hours_7) * 100 if total_possible_hours_7 > 0 else 0

    # 28-Day Periods (4 Weeks)
    last_28_days_start = today_ts - timedelta(days=28)
    prev_28_days_start = today_ts - timedelta(days=56)
    
    # Filter dataframes for the actual 28-day periods
    df_last_28 = df[(df["start_time"] >= last_28_days_start) & (df["start_time"] < today_ts)]
    df_prev_28 = df[(df["start_time"] >= prev_28_days_start) & (df["start_time"] < last_28_days_start)]
    
    hours_last_28 = df_last_28["duration"].sum()
    hours_prev_28 = df_prev_28["duration"].sum()
    users_last_28 = df_last_28["person_key"].nunique()
    users_prev_28 = df_prev_28["person_key"].nunique()

    # Calculate total possible hours for the 28-day period: 28 * 24 = 672
    total_possible_hours_28 = 28 * 24

    pct_last_28 = (hours_last_28 / total_possible_hours_28) * 100 if total_possible_hours_28 > 0 else 0
    pct_prev_28 = (hours_prev_28 / total_possible_hours_28) * 100 if total_possible_hours_28 > 0 else 0
    
    # --- Add Metrics to List ---
    
    metrics.append({"Metric": "Total Hours Last 7 Days", "Value": f"{hours_last_7:.2f}"})
    metrics.append({"Metric": "Unique Users Last 7 Days", "Value": f"{users_last_7}"})
    metrics.append({"Metric": "Total Hours Previous 7 Days", "Value": f"{hours_prev_7:.2f}"})
    metrics.append({"Metric": "Unique Users Previous 7 Days", "Value": f"{users_prev_7}"})
    metrics.append({"Metric": "Last 7 Days Hours Filled (%)", "Value": f"{pct_last_7:.2f}%"})
    metrics.append({"Metric": "Previous 7 Days Hours Filled (%)", "Value": f"{pct_prev_7:.2f}%"})
    
    # Updated to 28-day metrics
    metrics.append({"Metric": "Total Hours Last 28 Days", "Value": f"{hours_last_28:.2f}"})
    metrics.append({"Metric": "Unique Users Last 28 Days", "Value": f"{users_last_28}"})
    metrics.append({"Metric": "Total Hours Previous 28 Days", "Value": f"{hours_prev_28:.2f}"})
    metrics.append({"Metric": "Unique Users Previous 28 Days", "Value": f"{users_prev_28}"})
    metrics.append({"Metric": "Last 28 Days Hours Filled (%)", "Value": f"{pct_last_28:.2f}%"})
    metrics.append({"Metric": "Previous 28 Days Hours Filled (%)", "Value": f"{pct_prev_28:.2f}%"})
    
    return metrics

def export_user_summary_to_csv(df, outdir):
    """Exports a summary of users with their personal details and weekly/fortnightly average hours.
    (Keeps: firstname, lastname, mobile, email columns)."""
    
    start_date = df["start_time"].min().date()
    today = pd.Timestamp.now().normalize().date()
    total_days = (today - start_date).days + 1
    total_weeks = total_days / 7
    total_fortnights = total_days / 14
    
    # 1. Aggregate user details and total hours (using 'first' to grab one representative detail)
    user_agg = df.groupby("person_key").agg(
        firstname=("firstname", "first"),
        lastname=("lastname", "first"),
        email=("email", "first"),
        mobile=("mobile", "first"),
        total_hours=("duration", "sum"),
    ).reset_index()

    # 2. Get the most common person_name (for display name stability)
    most_common_names = df.groupby('person_key')['person_name'].apply(lambda x: x.mode()[0] if not x.mode().empty else 'UNKNOWN').reset_index(name='person_name')

    # 3. Merge display name and calculate averages
    user_summary = user_agg.merge(most_common_names, on='person_key', how='left')

    user_summary["weekly_avg_hours"] = user_summary["total_hours"] / total_weeks if total_weeks > 0 else 0
    user_summary["fortnightly_avg_hours"] = user_summary["total_hours"] / total_fortnights if total_fortnights > 0 else 0
    
    # Format to 2 decimal places before saving
    user_summary[["weekly_avg_hours", "fortnightly_avg_hours"]] = user_summary[["weekly_avg_hours", "fortnightly_avg_hours"]].round(2)
    
    # Export with new columns
    user_summary[[
        "firstname", "lastname", "mobile", "email", 
        "person_key", "person_name", 
        "weekly_avg_hours", "fortnightly_avg_hours"
    ]].to_csv(os.path.join(outdir, "user.csv"), index=False)
    
    return user_summary

def calculate_regulars_metrics(df, user_summary, outdir):
    """Identifies regulars, exports their lists, and returns metrics."""
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
    """Exports the top 10 most booked prayer slots based on total raw bookings/entries."""
    
    df["weekday"] = df["start_time"].dt.day_name()
    df["hour"] = df["start_time"].dt.hour
    
    # 1. Count the total number of bookings for each slot (using the full dataframe)
    top_slots_count = df.groupby(["weekday", "hour"]).size().reset_index(name="total_bookings")
    
    # 2. Sort by count and take top 10
    top_10 = top_slots_count.sort_values("total_bookings", ascending=False).head(10)

    # 3. Save the result
    top_10[["weekday", "hour", "total_bookings"]].to_csv(os.path.join(outdir, "top_10_slots.csv"), index=False)

def export_weekly_likelihood_metrics(df, outdir):
    """
    Calculates and exports the top 3 slots for two likelihood metrics, one being a probability distribution
    over all 168 slots (7 days * 24 hours).
    """
    
    # Calculate Total Weeks in the dataset (Denominator for Metric 1)
    start_date = df["start_time"].min().normalize()
    end_date = df["start_time"].max().normalize()
    total_weeks = max(1.0, (end_date - start_date).days / 7)
    
    df["weekday"] = df["start_time"].dt.day_name()
    df["hour"] = df["start_time"].dt.hour
    
    # -----------------------------------------------
    # Metric 1: Top 3 slots Avg Bookings per Week (Volume)
    # -----------------------------------------------
    
    # 1. Count total bookings per slot (Weekday + Hour)
    total_bookings_per_slot = df.groupby(["weekday", "hour"]).size().reset_index(name="total_bookings")
    
    # 2. Calculate average weekly bookings
    total_bookings_per_slot["weekly_avg_bookings"] = total_bookings_per_slot["total_bookings"] / total_weeks
    
    # 3. Rank and take top 3
    top_3_total = total_bookings_per_slot.sort_values("weekly_avg_bookings", ascending=False).head(3)
    
    top_3_total = top_3_total.rename(columns={"weekly_avg_bookings": "Likelihood Value"})
    top_3_total["Metric"] = "Top 3 - Avg Bookings per Week (Volume)"
    top_3_total = top_3_total[["Metric", "weekday", "hour", "Likelihood Value"]]


    # -----------------------------------------------
    # Metric 2: Slot Selection Probability (Unique User Choice) - All 168 slots sum to 1
    # -----------------------------------------------
    
    # 1. Filter: Get unique user selection for a slot (Person + Weekday + Hour)
    df_unique_choice = df.drop_duplicates(subset=["person_key", "weekday", "hour"])

    # 2. Denominator: Total number of unique slot choices made across the entire dataset (sum of unique entries)
    total_unique_choices = len(df_unique_choice)

    # 3. Numerator: Count how many unique users chose each slot
    slot_count = df_unique_choice.groupby(["weekday", "hour"]).size().reset_index(name="unique_user_count")

    # 4. Create all 168 possible slots (to ensure 0s are included)
    all_weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    all_hours = range(24)
    all_slots = pd.MultiIndex.from_product([all_weekdays, all_hours], names=['weekday', 'hour']).to_frame(index=False)

    # 5. Merge, fill 0s, and calculate probability
    slot_probability = all_slots.merge(slot_count, on=['weekday', 'hour'], how='left').fillna(0)
    slot_probability["unique_user_count"] = slot_probability["unique_user_count"].astype(int)
    slot_probability["Probability"] = slot_probability["unique_user_count"] / total_unique_choices
    
    # Export the full 168-slot probability table
    full_probability_df = slot_probability[["weekday", "hour", "Probability"]].sort_values(by=["Probability", "weekday", "hour"], ascending=[False, True, True])
    full_probability_df.to_csv(os.path.join(outdir, "unique_user_slot_probability.csv"), index=False)

    # Extract Top 3 from the full probability table for the comparison CSV
    top_3_probability = slot_probability.sort_values("Probability", ascending=False).head(3)
    
    top_3_probability = top_3_probability.rename(columns={"Probability": "Likelihood Value"})
    top_3_probability["Metric"] = "Top 3 - Slot Selection Probability (Unique User)"
    top_3_probability = top_3_probability[["Metric", "weekday", "hour", "Likelihood Value"]]
    
    
    # -----------------------------------------------
    # Combine Top 3s and Export (for comparison)
    # -----------------------------------------------
    
    combined_top_slots = pd.concat([top_3_total, top_3_probability], ignore_index=True)
    
    # Format value column to 4 decimal places for probability
    combined_top_slots["Likelihood Value"] = combined_top_slots["Likelihood Value"].round(4)
        
    combined_top_slots.to_csv(os.path.join(outdir, "weekly_slot_likelihood.csv"), index=False)
    
    return combined_top_slots


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

def plot_weekly_total_hours(df, outdir, goal_percentage):
    """Creates a line graph of the weekly total hours with a dynamic goal line (or no line if goal_percentage is None)."""
    df_weekly_total = df.groupby(pd.Grouper(key="start_time", freq="W"))["duration"].sum().reset_index()
    
    plt.figure(figsize=(10, 5))
    plt.plot(df_weekly_total["start_time"], df_weekly_total["duration"], marker='o')
    plt.title("Total Hours Logged per Week")
    plt.xlabel("Week Starting")
    plt.ylabel("Total Hours")
    plt.grid(True)
    
    ax = plt.gca()

    # --- Conditional Goal Line Logic ---
    if goal_percentage is not None and isinstance(goal_percentage, (int, float)):
        # Calculate goal line: 7 days * 24 hours = 168 total hours
        weekly_goal_hours = (7 * 24) * (goal_percentage / 100)
        # Add a red dashed horizontal line at the goal percentage mark
        ax.axhline(y=weekly_goal_hours, color='red', linestyle='--', label=f'{goal_percentage}% Goal ({weekly_goal_hours:.1f} hrs)')
        ax.legend()
    
    # Set the x-axis to show a label for every second week to improve readability
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d, %Y'))
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "weekly_total_hours.png"))
    plt.close()

def plot_monthly_total_hours(df_monthly_total, outdir, goal_percentage):
    """Creates a line graph of the monthly total hours with a dynamic goal line (or no line if goal_percentage is None)."""
    df_monthly_total = df_monthly_total.reset_index()
    
    # Corrected: Use 'start_time' column for formatting the month
    df_monthly_total["month"] = df_monthly_total["start_time"].dt.strftime('%b %Y')
    
    plt.figure(figsize=(10, 5))
    plt.plot(df_monthly_total["month"], df_monthly_total["duration"], marker='o')
    plt.title("Total Hours Logged per Month")
    plt.xlabel("Month")
    plt.ylabel("Total Hours")
    plt.xticks(rotation=45)
    plt.grid(True)
    
    ax = plt.gca()

    # --- Conditional Goal Line Logic ---
    if goal_percentage is not None and isinstance(goal_percentage, (int, float)):
        # Calculate goal line: Using standard 30 days * 24 hours = 720 total hours
        monthly_goal_hours = (30 * 24) * (goal_percentage / 100)
        # Add a red dashed horizontal line at the goal percentage mark
        ax.axhline(y=monthly_goal_hours, color='red', linestyle='--', label=f'{goal_percentage}% Goal ({monthly_goal_hours:.1f} hrs)')
        ax.legend()

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

def run_analysis(input_files, outdir, start_filter=None, end_filter=None, goal_percentage=10):
    """
    Orchestrates the entire analysis process.
    
    :param start_filter: Optional. A string representing the start date (e.g., "2024-01-01") or a year (e.g., "2024").
    :param end_filter: Optional. A string representing the end date (e.g., "2024-12-31") or a year (e.g., "2024").
    :param goal_percentage: The target percentage for weekly/monthly hours for goal lines on charts. Pass None to disable the line.
    """
    
    # ----------------------------------------------------
    # 1. LOAD, NORMALIZE, AND FILTER DATA
    # ----------------------------------------------------
    df = load_and_combine_csvs(input_files)
    df = normalize_columns(df)
    
    today = pd.Timestamp.now().normalize()
    
    # Determine overall min/max dates from raw data
    data_min_date = df["start_time"].min().normalize()
    
    # Set default filter dates: start from min data date, end today
    filter_start_date = data_min_date
    filter_end_date = today
    
    # Process start_filter input
    if start_filter:
        try:
            # Check if it's a year (4 digits)
            if str(start_filter).isdigit() and len(str(start_filter)) == 4:
                year = int(start_filter)
                filter_start_date = pd.to_datetime(f"{year}-01-01").normalize()
                # If only a start year is provided, set end to the end of that year
                if not end_filter:
                    filter_end_date = pd.to_datetime(f"{year}-12-31").normalize()
            # Otherwise, assume it's a date string
            else:
                filter_start_date = pd.to_datetime(start_filter, errors='coerce').normalize()
                # Fallback silently if parsing fails
                if pd.isna(filter_start_date):
                     filter_start_date = data_min_date
        except Exception:
            # Fallback silently if an unexpected error occurs
            filter_start_date = data_min_date
            
    # Process end_filter input
    if end_filter:
        try:
            # Check if it's a year (4 digits)
            if str(end_filter).isdigit() and len(str(end_filter)) == 4:
                year = int(end_filter)
                filter_end_date = pd.to_datetime(f"{year}-12-31").normalize()
            # Otherwise, assume it's a date string
            else:
                filter_end_date = pd.to_datetime(end_filter, errors='coerce').normalize()
                # Fallback silently if parsing fails
                if pd.isna(filter_end_date):
                    filter_end_date = today
        except Exception:
            # Fallback silently if an unexpected error occurs
            filter_end_date = today

    # Ensure start is before end and handle time boundary
    if filter_start_date > filter_end_date:
        # Warning print removed
        filter_start_date, filter_end_date = filter_end_date, filter_start_date
        
    # --- Update OUTDIR to include the analyzed date range ---
    date_range_str = f"_{filter_start_date.strftime('%Y%m%d')}_to_{filter_end_date.strftime('%Y%m%d')}"
    final_outdir = outdir.rstrip("/\\") + date_range_str
    
    # Apply filtering
    df = df[(df["start_time"].dt.normalize() >= filter_start_date) & (df["start_time"].dt.normalize() <= filter_end_date)]
    
    if df.empty:
        raise ValueError(f"No data found between {filter_start_date.strftime('%Y-%m-%d')} and {filter_end_date.strftime('%Y-%m-%d')}. Analysis aborted.")
        
    # Reassign outdir to the date-specific folder and create it
    outdir = final_outdir
    os.makedirs(outdir, exist_ok=True)
    
    # Drop NaNs and ensure data is before 'today' (which is the upper bound of the filter)
    df = df[df["start_time"] <= today]
    df = df.dropna(subset=["start_time", "end_time"])

    # ----------------------------------------------------
    # 2. RUN ANALYSIS
    # ----------------------------------------------------
    
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
    
    # Group by month start for plotting purposes (index is now a Timestamp)
    df_monthly_total = df.groupby(pd.Grouper(key="start_time", freq="MS"))["duration"].sum()

    metrics = calculate_dashboard_metrics(df, df_monthly_total)
    
    user_summary = export_user_summary_to_csv(df, outdir)
    regular_metrics = calculate_regulars_metrics(df, user_summary, outdir)
    
    all_metrics = metrics + regular_metrics
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(os.path.join(outdir, "dashboard_summary.csv"), index=False)
    
    # All analysis complete messages removed
    
    export_top_10_slots(df, outdir)
    export_weekly_likelihood_metrics(df, outdir)
    plot_hourly_distribution(df, outdir)
    plot_binary_hourly_distribution(df, outdir)
    
    # Call plotting functions with goal_percentage which can now be None
    plot_weekly_total_hours(df, outdir, goal_percentage)
    plot_monthly_total_hours(df_monthly_total, outdir, goal_percentage)
    plot_avg_hours_distribution(user_summary, outdir)
