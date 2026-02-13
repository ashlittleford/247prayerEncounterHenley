import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
from datetime import timedelta
import matplotlib.dates as mdates
import sys

# -----------------------
# Font Setup for Matplotlib
# -----------------------
# Register fonts
font_path_regular = os.path.join("fonts", "Montserrat-Regular.ttf")
font_path_bold = os.path.join("fonts", "Montserrat-Bold.ttf")

if os.path.exists(font_path_regular):
    fm.fontManager.addfont(font_path_regular)
if os.path.exists(font_path_bold):
    fm.fontManager.addfont(font_path_bold)

plt.rcParams['font.family'] = 'Montserrat'


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
            # Add source file column
            df["source_file"] = os.path.basename(path)

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
    for col in ["firstname", "lastname", "email", "mobile", "start_time", "end_time", "source_file"]:
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

    # 28-day metrics
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
    Runs the entire analysis process.

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

    # --- Update OUTDIR to include the analysed date range ---
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
    email_map = create_email_map()

    df["person_key"] = df.apply(make_person_key, axis=1, args=(email_map,))
    df["person_name"] = df.apply(
        lambda r: "UNKNOWN"
        if r["person_key"] == "UNKNOWN"
        else f"{str(r['firstname'] or '').strip()} {str(r['lastname'] or '').strip()}".strip(),
        axis=1,
    )
    df = calculate_hours(df)

    check_short_sessions(df)

    # Group by month start for plotting purposes
    df_monthly_total = df.groupby(pd.Grouper(key="start_time", freq="MS"))["duration"].sum()

    metrics = calculate_dashboard_metrics(df, df_monthly_total)

    user_summary = export_user_summary_to_csv(df, outdir)
    regular_metrics = calculate_regulars_metrics(df, user_summary, outdir)

    all_metrics = metrics + regular_metrics
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(os.path.join(outdir, "dashboard_summary.csv"), index=False)

    export_top_10_slots(df, outdir)
    export_weekly_likelihood_metrics(df, outdir)
    plot_hourly_distribution(df, outdir)
    plot_binary_hourly_distribution(df, outdir)

    # Call plotting functions with goal_percentage
    plot_weekly_total_hours(df, outdir, goal_percentage)
    plot_monthly_total_hours(df_monthly_total, outdir, goal_percentage)
    plot_avg_hours_distribution(user_summary, outdir)


# --- New Functions for Streamlit App (Add to prayer_analytics_lib.py) ---

def calculate_time_period_stats(df, days):
    """Calculates hours and unique users for a given time period (e.g., 7 days) and the preceding period."""
    today = pd.Timestamp.now().normalize()
    end_current = today + timedelta(days=1) # up to the start of today
    start_current = end_current - timedelta(days=days)

    start_previous = start_current - timedelta(days=days)
    end_previous = start_current

    # Filter for current period
    df_current = df[(df["start_time"] >= start_current) & (df["start_time"] < end_current)]
    current_hours = df_current["duration"].sum().round(2)
    current_unique_users = df_current["person_key"].nunique()

    # Filter for previous period
    df_previous = df[(df["start_time"] >= start_previous) & (df["start_time"] < end_previous)]
    previous_hours = df_previous["duration"].sum().round(2)

    # Calculate percentage change
    if previous_hours > 0:
        change_pct = ((current_hours - previous_hours) / previous_hours) * 100
    elif current_hours > 0:
        change_pct = 100.0 # From zero to a positive number
    else:
        change_pct = 0.0

    return current_hours, current_unique_users, change_pct

def calculate_fortnightly_regulars_metrics(df, user_summary, outdir):
    """
    Calculates the number of users who average > 1 hour per 14-day period.
    This metric is based on the entire filtered dataset.
    """
    total_days = (df["start_time"].max().normalize() - df["start_time"].min().normalize()).days
    if total_days < 14:
        # If the total duration is less than 14 days, base the calculation on 14 days.
        fortnight_count = 1
    else:
        # Calculate the number of full fortnight periods in the data
        fortnight_count = max(1, total_days / 14)

    # Calculate average duration per user per fortnight
    user_summary["avg_hours_per_fortnight"] = user_summary["total_hours"] / fortnight_count

    # UPDATED: Define a fortnight regular as someone averaging >= 1 hour per fortnight
    fortnight_regulars_df = user_summary[user_summary["avg_hours_per_fortnight"] >= 1]
    num_fortnight_regulars = len(fortnight_regulars_df)

    # Calculate hours contributed by fortnight regulars
    hours_by_fortnight_regulars = fortnight_regulars_df["total_hours"].sum()
    total_hours = user_summary["total_hours"].sum()

    hours_pct = (hours_by_fortnight_regulars / total_hours) * 100 if total_hours > 0 else 0

    return [
        {
            "Metric": "Number of Fortnightly Regulars",
            "Value": num_fortnight_regulars
        },
        {
            "Metric": "Total Hours by Fortnightly Regulars (%)",
            "Value": f"{hours_pct:.1f}%"
        },
    ]

def calculate_average_metrics(df):
    """Calculates average hours per week/month based on the filtered data period."""
    if df.empty:
        return []

    data_min_date = df["start_time"].min().normalize()
    data_max_date = df["start_time"].max().normalize()
    total_days = (data_max_date - data_min_date).days + 1

    if total_days == 0:
        return []

    total_hours = df["duration"].sum()

    # Calculate average hours per calendar week (7 days)
    avg_hours_per_week = (total_hours / total_days) * 7

    # Calculate average hours per calendar month (approx 30.44 days)
    avg_hours_per_month = (total_hours / total_days) * 30.44

    return [
        {
            "Metric": "Avg Hours per Week (Filtered Period)",
            "Value": f"{avg_hours_per_week:.1f}"
        },
        {
            "Metric": "Avg Hours per Month (Filtered Period)",
            "Value": f"{avg_hours_per_month:.1f}"
        },
    ]
def analyze_pray_days(df, pray_day_dates, exclude_gwop_from_new_logic=False):
    """
    Analyzes data specifically for Pray Days.

    Args:
        df: The main dataframe with 'start_time', 'end_time', 'person_key', 'duration'.
        pray_day_dates: A list of datetime.date objects representing the Pray Days.
        exclude_gwop_from_new_logic: If True, ignores 'gwop.csv' data when determining if a user is 'New'.

    Returns:
        A dictionary containing:
        - 'repeaters_count': int
        - 'summary_df': pd.DataFrame with per-Pray Day metrics.
    """
    # Create a copy to avoid SettingWithCopy warnings on the original df
    df = df.copy()

    # Ensure pray_day_dates are sorted unique dates
    pray_day_dates = sorted(list(set(pray_day_dates)))

    # Add date column for easy filtering
    df['date'] = df['start_time'].dt.date

    # 1. Global Metric: Repeaters (Users who attended > 1 Pray Day)
    pray_day_bookings = df[df['date'].isin(pray_day_dates)]

    if pray_day_bookings.empty:
        return {
            "repeaters_count": 0,
            "summary_df": pd.DataFrame()
        }

    user_pray_day_counts = pray_day_bookings.groupby('person_key')['date'].nunique()
    repeaters_count = (user_pray_day_counts > 1).sum()
    total_unique_participants = pray_day_bookings['person_key'].nunique()

    summary_data = []
    newbies_details_map = {}
    details_data = [] # Stores rows for the detailed breakdown CSV

    for p_date in pray_day_dates:
        # Filter for this specific pray day
        day_bookings = df[df['date'] == p_date]

        # If no bookings for this configured day, show 0s
        if day_bookings.empty:
            summary_data.append({
                "Date": p_date,
                "Total Participants": 0,
                ">1 Hour Participants": 0,
                "Done Pray Day Before": 0,
                "Done Regular Before": 0,
                "New Users Retained": 0,
                "Total New Users": 0,
                "Total Hours": 0.0
            })
            newbies_details_map[p_date] = pd.DataFrame(columns=["Name", "Total Hours"])
            continue

        # Prepare helpers for detailed breakdown
        # Map person_key to person_name for this day's attendees
        day_attendees_info = day_bookings[['person_key', 'person_name']].drop_duplicates('person_key').set_index('person_key')
        def get_name(pkey):
            return day_attendees_info.loc[pkey, 'person_name'] if pkey in day_attendees_info.index else "Unknown"

        attendees = day_bookings['person_key'].unique()
        total_participants = len(attendees)
        total_hours = day_bookings['duration'].sum()

        # Breakdown: Total Participants
        # We need user total hours on THIS day for evidence
        user_hours = day_bookings.groupby('person_key')['duration'].sum()

        for pkey in attendees:
            hours_val = user_hours.get(pkey, 0)
            details_data.append({
                "Date": p_date,
                "Metric": "Total Participants",
                "Name": get_name(pkey),
                "Evidence": f"{hours_val:.2f} hours"
            })

        # Q2: >1 hour on single pray day
        # Sum duration per user on this day
        # user_hours calculated above
        over_1h_users = user_hours[user_hours > 1].index.tolist()
        over_1h_count = len(over_1h_users)

        for pkey in over_1h_users:
            hours_val = user_hours.get(pkey, 0)
            details_data.append({
                "Date": p_date,
                "Metric": ">1 Hour Participants",
                "Name": get_name(pkey),
                "Evidence": f"{hours_val:.2f} hours"
            })

        # Q3: Done a pray day before
        # Find all previous pray days
        prev_pray_days = [d for d in pray_day_dates if d < p_date]
        if prev_pray_days:
            # Users who booked on any previous pray day
            prev_attendees = df[df['date'].isin(prev_pray_days)]['person_key'].unique()
            users_done_pray_day_before = list(set(attendees).intersection(prev_attendees))
            done_pray_day_before = len(users_done_pray_day_before)
        else:
            done_pray_day_before = 0
            users_done_pray_day_before = []

        for pkey in users_done_pray_day_before:
             details_data.append({
                "Date": p_date,
                "Metric": "Done Pray Day Before",
                "Name": get_name(pkey),
                "Evidence": "Attended previous Pray Day"
            })

        # Q4: Done a regular hour before
        # Regular hours are bookings on dates NOT in pray_day_dates and BEFORE p_date
        regular_history_bookings = df[
            (df['date'] < p_date) &
            (~df['date'].isin(pray_day_dates))
        ]
        regular_history_users = regular_history_bookings['person_key'].unique()
        users_done_regular_before = list(set(attendees).intersection(regular_history_users))
        done_regular_before = len(users_done_regular_before)

        for pkey in users_done_regular_before:
             details_data.append({
                "Date": p_date,
                "Metric": "Done Regular Before",
                "Name": get_name(pkey),
                "Evidence": "Attended regular slot before"
            })

        # Q5: First time on Pray Day -> Retained
        # "How many people who sign up for the first time for a PRAY DAY then sign up for another hour afterwards?"

        # Optimization: Pre-calculate first booking for all attendees on this day
        # Filter global df for these users once
        attendees_bookings = df[df['person_key'].isin(attendees)]

        # Determine which history to use for finding the "First Booking"
        if exclude_gwop_from_new_logic and 'source_file' in attendees_bookings.columns:
            # Filter out GWOP bookings for the purpose of checking "newness"
            history_for_newness = attendees_bookings[attendees_bookings['source_file'] != 'gwop.csv']
        else:
            history_for_newness = attendees_bookings

        # Group by user to find first booking date
        user_first_booking = history_for_newness.groupby('person_key')['start_time'].min().dt.date

        # Identify new users (whose first booking is exactly this p_date)
        new_users_series = user_first_booking[user_first_booking == p_date]
        new_users_on_this_day = new_users_series.index.tolist()

        for pkey in new_users_on_this_day:
             details_data.append({
                "Date": p_date,
                "Metric": "Total New Users",
                "Name": get_name(pkey),
                "Evidence": "First booking ever"
            })

        # Check retention: Do these new users have bookings > p_date?
        # We can reuse attendees_bookings but we need to check if they have bookings > p_date
        # Wait, attendees_bookings contains ALL bookings for these users.

        retained_count = 0
        details_df = pd.DataFrame(columns=["Name", "Total Hours"])
        users_with_future_bookings = []

        if new_users_on_this_day:
            # Filter bookings for just the new users
            new_users_bookings = attendees_bookings[attendees_bookings['person_key'].isin(new_users_on_this_day)]

            # Check for any booking date > p_date per user
            users_with_future_bookings = new_users_bookings[new_users_bookings['date'] > p_date]['person_key'].unique()
            retained_count = len(users_with_future_bookings)

            # Calculate details for new users (Name and Total Hours)
            # 1. Group by person_key to sum duration
            grouped = new_users_bookings.groupby("person_key")["duration"].sum().reset_index(name="Total Hours")

            # 2. Get names. Drop duplicates on person_key to get names
            names_df = new_users_bookings[["person_key", "person_name"]].drop_duplicates("person_key")

            # 3. Merge
            details_df = grouped.merge(names_df, on="person_key")

            # 4. Format
            details_df = details_df[["person_name", "Total Hours"]].rename(columns={"person_name": "Name"})
            details_df["Total Hours"] = details_df["Total Hours"].round(2)
            details_df = details_df.sort_values("Total Hours", ascending=False)

        for pkey in users_with_future_bookings:
             details_data.append({
                "Date": p_date,
                "Metric": "New Users Retained",
                "Name": get_name(pkey),
                "Evidence": "Booked subsequent slot"
            })

        newbies_details_map[p_date] = details_df

        summary_data.append({
            "Date": p_date,
            "Total Participants": total_participants,
            ">1 Hour Participants": over_1h_count,
            "Done Pray Day Before": done_pray_day_before,
            "Done Regular Before": done_regular_before,
            "New Users Retained": retained_count,
            "Total New Users": len(new_users_on_this_day),
            "Total Hours": total_hours
        })

    summary_df = pd.DataFrame(summary_data)
    breakdown_details_df = pd.DataFrame(details_data)

    return {
        "repeaters_count": repeaters_count,
        "total_unique_participants": total_unique_participants,
        "summary_df": summary_df,
        "newbies_details": newbies_details_map,
        "breakdown_details": breakdown_details_df
    }


def create_email_map(filepath="email_duplicates.csv"):
    """
    Loads email duplicates and creates a lookup map for primary emails.
    """
    if not os.path.exists(filepath):
        return {}

    df = pd.read_csv(filepath)
    df.dropna(subset=['primary', 'additional'], inplace=True)

    primary_emails = df['primary'].astype(str).str.strip().str.lower()
    additional_emails = df['additional'].astype(str).str.strip().str.lower()

    return dict(zip(additional_emails, primary_emails))


def create_merge_map(filepath="person_merges.csv"):
    """
    Loads person merges and creates a lookup map for merging person keys.
    Raises:
        Exception: If the file cannot be read or processed.
    """
    if not os.path.exists(filepath):
        return {}

    df = pd.read_csv(filepath)
    df.dropna(subset=['source_key', 'target_key'], inplace=True)

    source_keys = df['source_key'].astype(str).str.strip()
    target_keys = df['target_key'].astype(str).str.strip()

    return dict(zip(source_keys, target_keys))
