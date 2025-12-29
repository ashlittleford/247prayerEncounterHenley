import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import html
from io import BytesIO
from datetime import timedelta
import re
import csv
import subprocess

# import streamlit_authenticator as stauth  <-- REMOVED

# Assuming prayer_analytics_lib.py is in the same directory
import prayer_analytics_lib as pal 

# --- CONFIGURATION & SETUP ---
LOGO_FILE = "logo.png"          
CUSTOM_LOGO_WIDTH = 100         
PRAY_DAYS_FILE = "pray_days.json"

# 1. Set page config FIRST
st.set_page_config(layout="wide", page_title="Encounter Henley Prayer Room Insights")


# ======================================================================
# 🔐 AUTHENTICATION BLOCK REMOVED 🔐
# The application will now start immediately.
# ======================================================================


# --- CUSTOM CSS INJECTION ---
CUSTOM_STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Montserrat', sans-serif;
}

/* 1. FIXES THE WARNING BOX COLOR */
.stAlert-warning {
    background-color: #7A8AB2 !important; 
    color: white !important; 
}

/* Ensures icons and specific text elements are white */
.stAlert-warning .st-emotion-cache-1c5c0d6,
.stAlert-warning .st-emotion-cache-1hch22i,
.stAlert-warning .st-emotion-cache-1l0f3w6,
.stAlert-warning p { 
    color: white !important; 
}
</style>
"""
st.markdown(CUSTOM_STYLE, unsafe_allow_html=True)
# ---------------------------------------------------


# --- Streamlit Title and Logo ---

logo_path_exists = os.path.exists(LOGO_FILE)
if logo_path_exists:
    # Center the logo using columns
    left_co, cent_co, last_co = st.sidebar.columns(3)
    with cent_co:
        st.image(LOGO_FILE, width=CUSTOM_LOGO_WIDTH)
else:
    st.sidebar.warning(f"Logo file '{LOGO_FILE}' not found. Please ensure it is in the same folder as the script.")
    
st.title("Encounter Henley Prayer Room Insights")
st.markdown(f"Discovering prayer impact from the city to beach. Customise the analysis below and press **Run Analysis**.")
st.sidebar.markdown('---')


# --- Helper functions to run the library and display results ---

def commit_and_push_changes(filepath, commit_message):
    """
    Commits and pushes a specific file to the GitHub repository.
    Requires 'GITHUB_TOKEN' to be set in st.secrets.
    """
    # Check if token exists in secrets
    if "GITHUB_TOKEN" not in st.secrets:
        return False, "GITHUB_TOKEN not found in secrets. Changes saved locally only."

    token = st.secrets["GITHUB_TOKEN"]

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
        # Fallback if remote is not standard https (e.g. ssh), though Streamlit Cloud usually uses https
        return False, "Git remote URL format not supported (requires https)."

    try:
        # Configure git user (local to this run)
        # We ignore errors here in case it's already configured
        subprocess.run(["git", "config", "user.email", "streamlit-bot@example.com"], check=False)
        subprocess.run(["git", "config", "user.name", "Streamlit Bot"], check=False)

        # Add file
        subprocess.run(["git", "add", filepath], check=True, capture_output=True, text=True)

        # Commit
        # Check if there are changes first to avoid empty commit error
        status = subprocess.run(["git", "status", "--porcelain", filepath], capture_output=True, text=True)
        if not status.stdout.strip():
            return True, "No changes to commit."

        subprocess.run(["git", "commit", "-m", commit_message], check=True, capture_output=True, text=True)

        # Pull (Rebase to avoid merge conflicts)
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

def display_pray_day_results(df, pray_day_dates, exclude_gwop_new_logic=False):
    """
    Computes and displays Pray Days analytics.
    """
    if not pray_day_dates:
        st.info("No Pray Day dates provided.")
        return

    # Call the library function
    results = pal.analyze_pray_days(df, pray_day_dates, exclude_gwop_from_new_logic=exclude_gwop_new_logic)

    st.header("Pray Days Analysis")

    # Extract Metrics
    total_participants = results.get("total_unique_participants", 0)
    total_repeaters = results.get("repeaters_count", 0)

    summary_df = results["summary_df"]
    newbies_details = results.get("newbies_details", {})

    # Calculate Latest Pray Day Newbies Total
    latest_newbies = 0
    latest_date = None
    if not summary_df.empty:
        # Sort by Date to find the latest
        summary_df["Date"] = pd.to_datetime(summary_df["Date"])
        latest_row = summary_df.sort_values("Date").iloc[-1]
        latest_newbies = latest_row["Total New Users"]
        latest_date = latest_row["Date"].date()

    # Display Top Metrics in requested order
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Individual Pray Day Participants", total_participants)
    c2.metric("Total Repeaters", total_repeaters)
    c3.metric("New to Pray Day", latest_newbies)

    # Expander for Newbies details
    if latest_date and latest_date in newbies_details:
         with st.expander("New to Pray Day Details"):
              st.dataframe(newbies_details[latest_date], hide_index=True, use_container_width=True)

    st.subheader("Per Pray Day Breakdown")

    if summary_df.empty:
        st.write("No bookings found for the specified Pray Days.")
    else:
        # Transpose Logic
        # 1. Set Date as Index
        display_df = summary_df.set_index("Date").copy()

        # 2. Transpose
        display_df = display_df.T

        # 3. Format Columns (Dates)
        new_columns = [format_date_custom(pd.to_datetime(c)) for c in display_df.columns]
        display_df.columns = new_columns

        st.dataframe(display_df, use_container_width=True)

        # Helper to convert DF to CSV for download
        @st.cache_data
        def convert_df_to_csv(df):
            return df.to_csv(index=False).encode('utf-8')

        # Download Button for Breakdown
        if "breakdown_details" in results and not results["breakdown_details"].empty:
             breakdown_csv = convert_df_to_csv(results["breakdown_details"])
             st.download_button(
                 label="Download Breakdown Details (Excel/CSV)",
                 data=breakdown_csv,
                 file_name='pray_day_breakdown.csv',
                 mime='text/csv',
             )

        st.markdown("""
        **Metric Explanations:**
        - **Total Participants:** Unique users who booked on this Pray Day.
        - **>1 Hour Participants:** Users who prayed for more than 1 hour (aggregated) on this day.
        - **Done Pray Day Before:** Users who had participated in at least one *previous* Pray Day from your list.
        - **Done Regular Before:** Users who have booked a slot on a non-Pray Day date *before* this Pray Day.
        - **New Users Retained:** Users whose *first ever* booking was on this Pray Day, and who subsequently booked another slot on a later date.
        - **Total New Users:** Users whose *first ever* booking was on this Pray Day.
        """)


def format_change_metric(change_pct):
    """
    Ensures Green is positive, Red is negative for Streamlit's delta metrics.
    """
    if change_pct != 0:
        delta = f"{change_pct:.1f}%"
        delta_color = "normal"
    else:
        delta = "No Change"
        delta_color = "off"
    return delta, delta_color

@st.cache_data(show_spinner=False)
def run_full_analysis(input_files, outdir, start_filter, end_filter, goal_percentage, cache_key=None):
    """
    Executes the analysis logic from prayer_analytics_lib.py.
    cache_key is used to force re-computation when config files change.
    """
    
    # ----------------------------------------------------
    # 1. LOAD, NORMALIZE, AND PREPARE UNFILTERED DATA
    # ----------------------------------------------------
    df_unfiltered = pal.load_and_combine_csvs(input_files)
    df_unfiltered = pal.normalize_columns(df_unfiltered)
    
    today = pd.Timestamp.now().normalize()
    data_min_date = df_unfiltered["start_time"].min().normalize()
    
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

    # Load person merges
    merge_map = {}
    person_merges_path = "person_merges.csv"
    if os.path.exists(person_merges_path):
        pm_df = pd.read_csv(person_merges_path)
        for _, row in pm_df.iterrows():
             src = str(row["source_key"] or "").strip()
             tgt = str(row["target_key"] or "").strip()
             if src and tgt:
                 merge_map[src] = tgt

    # Apply keys and calculate hours on the UNFILTERED data
    df_unfiltered["person_key"] = df_unfiltered.apply(pal.make_person_key, axis=1, args=(email_map,))

    # Apply merges
    if merge_map:
        df_unfiltered["person_key"] = df_unfiltered["person_key"].replace(merge_map)

    # Apply keys
    df_unfiltered["person_key"] = df_unfiltered.apply(pal.make_person_key, axis=1, args=(email_map,))

    # Apply Key Merges (Overrides)
    person_merges_path = "person_merges.csv"
    if os.path.exists(person_merges_path):
        try:
            merges_df = pd.read_csv(person_merges_path)
            # Create a dictionary for mapping source_key -> target_key
            merge_map = {}
            for _, row in merges_df.iterrows():
                if pd.notna(row['source_key']) and pd.notna(row['target_key']):
                    merge_map[row['source_key'].strip()] = row['target_key'].strip()

            if merge_map:
                df_unfiltered["person_key"] = df_unfiltered["person_key"].replace(merge_map)
        except Exception as e:
            st.warning(f"Error loading person_merges.csv: {e}")

    # Calculate hours on the UNFILTERED data
    df_unfiltered["person_name"] = df_unfiltered.apply(
        lambda r: "UNKNOWN"
        if r["person_key"] == "UNKNOWN"
        else f"{str(r['firstname'] or '').strip()} {str(r['lastname'] or '').strip()}".strip(),
        axis=1,
    )
    df_unfiltered = pal.calculate_hours(df_unfiltered)
    pal.check_short_sessions(df_unfiltered) 
    
    # Calculate UNFILTERED recent stats (as requested)
    h_7, u_7, c_7 = pal.calculate_time_period_stats(df_unfiltered, 7)
    h_28, u_28, c_28 = pal.calculate_time_period_stats(df_unfiltered, 28)
    recently_stats = {
        "H7": h_7, "U7": u_7, "C7": c_7,
        "H28": h_28, "U28": u_28, "C28": c_28,
    }


    # ----------------------------------------------------
    # 2. APPLY FILTERING TO CREATE DF (FILTERED DATA)
    # ----------------------------------------------------
    
    # Process date inputs from Streamlit
    filter_start_date = pd.to_datetime(start_date) if start_date else data_min_date
    filter_end_date = pd.to_datetime(end_date) if end_date else today

    # Ensure start is before end and handle time boundary
    if filter_start_date > filter_end_date:
        filter_start_date, filter_end_date = filter_end_date, filter_start_date
    
    # Apply filtering
    df = df_unfiltered[(df_unfiltered["start_time"].dt.normalize() >= filter_start_date) & (df_unfiltered["start_time"].dt.normalize() <= filter_end_date)].copy()
    
    if df.empty:
        raise ValueError(f"No data found between {filter_start_date.strftime('%Y-%m-%d')} and {filter_end_date.strftime('%Y-%m-%d')}.")

    # Final cleanup before analysis
    df = df[df["start_time"] <= today]
    df = df.dropna(subset=["start_time", "end_time"])
    
    # FIX for ValueError: cannot convert float NaN to integer issue on re-filter
    df = df.dropna(subset=['start_time', 'end_time', 'person_key'])


    # ----------------------------------------------------
    # 3. RUN CORE ANALYSIS ON FILTERED DATA
    # ----------------------------------------------------
    
    # Group by month start for plotting purposes
    df_monthly_total = df.groupby(pd.Grouper(key="start_time", freq="MS"))["duration"].sum()

    # Calculate metrics
    os.makedirs(outdir, exist_ok=True)
    
    metrics = pal.calculate_dashboard_metrics(df, df_monthly_total)
    
    # Calculate user summary and regulars metrics
    user_summary = pal.export_user_summary_to_csv(df, outdir)
    regular_metrics = pal.calculate_regulars_metrics(df, user_summary, outdir)
    
    # Calculate fortnightly regulars metrics
    fortnightly_metrics = pal.calculate_fortnightly_regulars_metrics(df, user_summary, outdir)
    
    average_metrics = pal.calculate_average_metrics(df)
    
    # Include all metrics
    all_metrics = metrics + regular_metrics + fortnightly_metrics + average_metrics
    metrics_df = pd.DataFrame(all_metrics)
    
    # Calculate slot likelihoods
    likelihood_df = pal.export_weekly_likelihood_metrics(df, outdir)
    
    # Manually calculate Total Sessions
    total_sessions_count = len(df)
    
    # Return dataframes needed for display (including df_unfiltered for Pray Day analysis)
    return df, df_monthly_total, metrics_df, user_summary, likelihood_df, recently_stats, total_sessions_count, df_unfiltered

# ------------------------------------------------------------------------------------------------------------------
# Display Logic
# ------------------------------------------------------------------------------------------------------------------

def display_results(df, df_monthly_total, metrics_df, user_summary, likelihood_df, recently_stats, goal_percentage, outdir, total_sessions_count):
    """Displays the metrics and plots in the Streamlit interface."""
    
    # Helper for plots (now more robust against File Not Found errors)
    def generate_and_display_plot(plot_func, data, title, outdir, goal_percentage=None):
        args = [data, outdir]
        if goal_percentage is not None:
            args.append(goal_percentage)
        
        plot_filepath = os.path.join(outdir, title) 

        try:
            # Ensure output directory exists (crucial if analysis was cached)
            os.makedirs(outdir, exist_ok=True)

            # 1. Execute the plotting function (MUST save the file to plot_filepath inside pal.py)
            plot_func(*args)
            
            # 2. Check if the file was created.
            if not os.path.exists(plot_filepath):
                 raise FileNotFoundError(f"Plot function executed but did not save the plot as '{title}' to the '{outdir}' directory. Check your pal.py save path.")
            
            # 3. Read and display the file
            with open(plot_filepath, "rb") as f:
                st.image(f.read(), caption=title, use_container_width=True)
            
            # 4. Clean up
            os.remove(plot_filepath)

        except Exception as e:
            st.warning(f"Could not generate plot for '{title}'. Error: {e}")


    # Re-structure metrics for a cleaner display using Streamlit columns
    metric_dict = {m['Metric']: m['Value'] for m in metrics_df.to_dict('records')}
    
    st.markdown("---")
    
    # --- Recent Activity (MOVED TO TOP) ---
    st.header("Recent Activity (Unfiltered Data)")

    # Use 4 columns for a flatter structure to save vertical space on mobile (removes nested subheaders)
    col_h7, col_u7, col_h28, col_u28 = st.columns(4)
    
    # Last 7 Days
    delta_7, delta_color_7 = format_change_metric(recently_stats["C7"])

    col_h7.metric(
        label="Hours (7 Days)",
        value=recently_stats["H7"],
        delta=delta_7,
        delta_color=delta_color_7
    )
    col_u7.metric(
        label="Users (7 Days)",
        value=recently_stats["U7"]
    )

    # Last 28 Days
    delta_28, delta_color_28 = format_change_metric(recently_stats["C28"])

    col_h28.metric(
        label="Hours (28 Days)",
        value=recently_stats["H28"],
        delta=delta_28,
        delta_color=delta_color_28
    )
    col_u28.metric(
        label="Users (28 Days)",
        value=recently_stats["U28"]
    )
    
    st.markdown("---")

    # --- Analysis Overview ---
    st.header("Analysis Overview (Filtered Data)")
    
    # Overview
    cols = st.columns(4)
    cols[0].metric("Total Hours of Prayer (Filtered)", metric_dict.get("Total Hours of Prayer", "N/A"))
    cols[1].metric("Total Unique Users (Filtered)", metric_dict.get("Total Individual Users", "N/A"))
    cols[2].metric("Avg Hours per Week (Filtered Period)", metric_dict.get("Avg Hours per Week (Filtered Period)", "N/A"))
    cols[3].metric("Avg Hours per Month (Filtered Period)", metric_dict.get("Avg Hours per Month (Filtered Period)", "N/A"))

    st.subheader("Regulars and Occupancy")
    cols_reg = st.columns(4)
    cols_reg[0].metric("Weekly Regulars (Avg $\ge$ 1hr/wk)", metric_dict.get("Number of Weekly Regulars", "N/A"))
    cols_reg[1].metric("Fortnightly Regulars (Avg $\ge$ 1hr/fn)", metric_dict.get("Number of Fortnightly Regulars", "N/A")) 
    cols_reg[2].metric("Hours by Weekly Regulars (%)", metric_dict.get("Total Hours by Weekly Regulars (%)", "N/A"))
    cols_reg[3].metric("Total Sessions", total_sessions_count)

    # --- Display lists of regulars ---

    # 1. Weekly Regulars
    with st.expander("Weekly Regulars Details", expanded=False):
        if "weekly_avg_hours" in user_summary.columns:
            # Filter for weekly regulars
            weekly_df = user_summary[user_summary["weekly_avg_hours"] >= 1].copy()
            if not weekly_df.empty:
                 # Display Name and Weekly Avg
                 display_df = weekly_df[["person_name", "weekly_avg_hours"]].sort_values("weekly_avg_hours", ascending=False)
                 display_df.rename(columns={"person_name": "Name", "weekly_avg_hours": "Avg Hours/Week"}, inplace=True)
                 st.dataframe(display_df, hide_index=True, use_container_width=True)
            else:
                 st.write("No weekly regulars found.")
        else:
             st.write("Data not available.")

    # 2. Fortnightly Regulars
    with st.expander("Fortnightly Regulars Details", expanded=False):
        if "avg_hours_per_fortnight" in user_summary.columns:
            # Filter for fortnightly regulars
            fortnightly_df = user_summary[user_summary["avg_hours_per_fortnight"] > 1].copy()
            if not fortnightly_df.empty:
                 # Display Name and Weekly Avg (as requested "avg per week number")
                 # We use weekly_avg_hours column for this.
                 display_df = fortnightly_df[["person_name", "weekly_avg_hours"]].sort_values("weekly_avg_hours", ascending=False)
                 display_df.rename(columns={"person_name": "Name", "weekly_avg_hours": "Avg Hours/Week"}, inplace=True)
                 st.dataframe(display_df, hide_index=True, use_container_width=True)
            else:
                 st.write("No fortnightly regulars found.")
        else:
             st.write("Data not available.")

    st.markdown("---")
    
    # --- Time-Series & Distribution Charts ---
    st.header("Time-Series & Distribution Charts")
    
    col_w, col_m = st.columns(2)
    with col_w:
        generate_and_display_plot(pal.plot_weekly_total_hours, df, "weekly_total_hours.png", outdir, goal_percentage)
    with col_m:
        generate_and_display_plot(pal.plot_monthly_total_hours, df_monthly_total, "monthly_total_hours.png", outdir, goal_percentage)

    col_h, col_hu = st.columns(2)
    with col_h:
        generate_and_display_plot(pal.plot_hourly_distribution, df, "hourly_distribution.png", outdir)
    with col_hu:
        generate_and_display_plot(pal.plot_binary_hourly_distribution, df, "hourly_users.png", outdir)
    
    st.subheader("User Average Hours Distribution")
    generate_and_display_plot(pal.plot_avg_hours_distribution, user_summary, "avg_hours_distribution.png", outdir)
    
    st.markdown("---")
    
    # --- Slot Details ---
    st.header("Slot Details")

    st.subheader("Top Booking Timeslots")
    top_slots_df = metrics_df[metrics_df["Metric"].str.contains("Top") & ~metrics_df["Metric"].str.contains("%")].copy()
    
    # Clean up and rename columns for display
    top_slots_df.rename(columns={"Value": "Total Bookings"}, inplace=True)
    
    # Top 3 Slot Display with Emojis
    for i in range(min(3, len(top_slots_df))):
        row = top_slots_df.iloc[i]
        
        metric_name = row["Metric"]
        bookings = row["Total Bookings"]
        
        # 1. FINAL STRING CLEANUP (Needed to identify icon, but text is now simpler)
        junk_prefixes = ["Top Timeslot", "Timeslot", "Top"]
        
        clean_slot_description = metric_name
        for prefix in junk_prefixes:
            clean_slot_description = clean_slot_description.replace(prefix, "").strip()
        
        if clean_slot_description.startswith(str(i+1)):
            clean_slot_description = clean_slot_description[len(str(i+1)):].strip()

        if ' with ' in clean_slot_description:
            clean_slot_description = clean_slot_description.split(' with ')[0].strip()
            
        # 2. Determine the icon to use
        if i == 0:
            icon_for_rank = "🏆" 
        else:
            icon_for_rank = "⭐" 

        # 3. Construct the final desired string format: 
        final_text = f"**# {i+1} - {bookings}**" # Simplified as requested
        
        st.warning(final_text, icon=icon_for_rank)

    # If there are more slots, display them in a smaller table
    if len(top_slots_df) > 3:
        st.markdown("**Other Top Slots:**")
        st.dataframe(top_slots_df.iloc[3:], hide_index=True, use_container_width=True)

    
    st.subheader("Weekly Slot Likelihood Comparison")
    st.dataframe(likelihood_df.rename(columns={"hour": "Hour (24hr)"}), use_container_width=True)
    st.markdown(
        """
        - **Avg Bookings per Week (Volume):** Shows the average number of bookings for that slot per week in the filtered period.
        - **Slot Selection Probability (Unique User):** The probability that a *unique user* will choose this specific slot.
        """
    )
    
    # Add a download link for the full user summary
    @st.cache_data
    def convert_df_to_csv(df):
        return df.to_csv(index=False).encode('utf-8')

    user_summary_csv = convert_df_to_csv(user_summary)
    st.download_button(
        label="Download Full User Summary CSV",
        data=user_summary_csv,
        file_name='user_summary_export.csv',
        mime='text/csv',
    )


# --- 1. Sidebar for User Inputs (Customization) ---

st.sidebar.header("Data Management")

# Display Last Updated Date
if os.path.exists("current.csv"):
    last_modified_timestamp = os.path.getmtime("current.csv")
    last_modified_date = pd.to_datetime(last_modified_timestamp, unit='s').strftime('%Y-%m-%d %H:%M:%S')
    st.sidebar.write(f"**current.csv** last updated:\n{last_modified_date}")
else:
    st.sidebar.warning("current.csv not found.")

# File Uploader
# NOTE: Removed type=['csv'] restriction to support mobile uploads where MIME types can be tricky.
uploaded_file = st.sidebar.file_uploader("Upload new current.csv", type=None)

if uploaded_file is not None:
    # Validate extension or MIME type here if strictly needed, but for now we trust the user
    # or handle read errors in the library.

    # Check filename extension to warn user if it looks wrong, but allow upload.
    if not uploaded_file.name.lower().endswith('.csv'):
        st.sidebar.warning("File does not have .csv extension. Please ensure it is a valid CSV.")

    # Option to commit/replace
    if st.sidebar.button("Replace current.csv with uploaded file"):
        # Save the file
        try:
            with open("current.csv", "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.session_state['upload_success'] = True
            # Trigger a rerun to update the date and potentially the analysis
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Error saving file: {e}")

if st.session_state.get('upload_success'):
    st.sidebar.success("current.csv updated successfully!")
    st.session_state['upload_success'] = False

st.sidebar.markdown('---')

st.sidebar.header("Configuration")

# Customization 1: Include GWOP Data
include_gwop = st.sidebar.checkbox("Include 'gwop.csv' data", value=True)

# Customization 2: Date Filters
st.sidebar.subheader("Date Range Filter")
# Default start date hardcoded to Jan 5th 2025 as requested
default_start = pd.Timestamp("2025-01-05").date()
start_date = st.sidebar.date_input("Start Date (optional)", value=default_start)
end_date = st.sidebar.date_input("End Date (optional)", value=None)

# Customization 3: Goal Percentage
st.sidebar.subheader("Chart Goal Line")
goal_percentage = st.sidebar.slider(
    "Goal Percentage for Charts (0-100%)",
    min_value=0,
    max_value=100,
    value=10,
    step=5
)
if goal_percentage == 0:
    goal_percentage = None

# Logic to load/save (Handles migration from old TXT if JSON missing)
def save_pray_days_config(data):
    with open(PRAY_DAYS_FILE, "w") as f:
        json.dump(data, f, indent=4)

    # Sync to Git
    success, msg = commit_and_push_changes(PRAY_DAYS_FILE, f"Update {PRAY_DAYS_FILE}: Modified Pray Days list")
    if not success:
        st.sidebar.warning(f"Saved locally, but Git sync failed: {msg}")

def load_pray_days_config():
    # 1. Try JSON first
    if os.path.exists(PRAY_DAYS_FILE):
        try:
            with open(PRAY_DAYS_FILE, "r") as f:
                return json.load(f)
        except:
            pass

    # 2. Migration: Check for old TXT file
    old_file = "pray_days.txt"
    if os.path.exists(old_file):
        dates = []
        try:
            with open(old_file, "r") as f:
                lines = f.read().split('\n')
                for line in lines:
                    line = line.strip()
                    if line:
                        try:
                            # Try parsing
                            try:
                                dt = pd.to_datetime(line, dayfirst=False).date()
                            except:
                                dt = pd.to_datetime(line, dayfirst=True).date()
                            dates.append({"date": dt.isoformat(), "label": ""})
                        except:
                            pass

            # Save to new JSON format immediately
            save_pray_days_config(dates)
            return dates
        except Exception as e:
            st.sidebar.error(f"Migration error: {e}")
            return []

    return []

# Initialize session state if not set
if 'pray_days_list' not in st.session_state:
    st.session_state['pray_days_list'] = load_pray_days_config()


# Extract simple list of dates for analysis
pray_day_dates = [pd.to_datetime(x['date']).date() for x in st.session_state['pray_days_list']]

# Define a temporary in-memory output directory (required by the lib functions)
OUTPUT_DIR = "temp_dashboard_out"


# --- 3. Main execution block with a Run button ---
# (Old block removed in favor of RESTRUCTURING UI block below)


# --- RESTRUCTURING UI ---
# The previous block put tabs inside the "Run Analysis" button.
# This makes the "Admin" tab hidden until you run analysis.
# The user wants to configure Pray Days in the Admin tab.
# So the Admin tab must be accessible independently.

# New Structure:
# 1. Sidebar: Data Management (Current file), Run Button.
# 2. Main Area: Tabs (General, Pray Days, Admin).
#    - General & Pray Days: Show placeholder or empty state if no analysis run yet.
#    - Admin: Always active.

# Let's fix the layout.

# Clear the previous "if st.sidebar.button..." block logic mentally.

# Define Tabs at the top level
tab_general, tab_praydays, tab_admin = st.tabs(["General Analytics", "Pray Days Analytics", "Admin"])

# --- ADMIN TAB CONTENT ---
with tab_admin:
    st.header("Pray Day Configuration")

    with st.container(border=False):
        st.caption("Global Settings")
        # Customization 5: Exclude GWOP from New to Pray Day Logic
        exclude_gwop_new = st.checkbox(
            "Exclude GWOP from 'New to Pray Day' Logic",
            value=False,
            help="If checked, GWOP data will be ignored when determining if a user is 'new' (first-time booking). However, their total hours will still include GWOP time."
        )

        st.divider()
        st.caption("Manage Days")

        # --- Input Area for Pray Days ---
        with st.form("add_pray_day_form", clear_on_submit=True, border=False):
            # Using vertical_alignment="bottom" to align button with input fields
            c1, c2, c3 = st.columns([3, 4, 2], vertical_alignment="bottom")

            with c1:
                # value=None defaults to today
                new_date = st.date_input("Date", value=None)
            with c2:
                new_label = st.text_input("Label", placeholder="Optional (e.g. 'Global Day')")
            with c3:
                submitted = st.form_submit_button("Add Pray Day", use_container_width=True)

            if submitted:
                if new_date:
                    d_str = new_date.isoformat()
                    # Check if already exists
                    exists = any(d['date'] == d_str for d in st.session_state['pray_days_list'])
                    if not exists:
                        st.session_state['pray_days_list'].append({
                            "date": d_str,
                            "label": new_label.strip() if new_label else ""
                        })
                        # Save immediately
                        save_pray_days_config(st.session_state['pray_days_list'])
                        st.rerun()
                    else:
                        st.warning("This date is already in the list.")

        # --- Display List (Styled Bubbles) ---
        st.write("") # Spacer
        st.write("**Configured Days:**")

        if st.session_state['pray_days_list']:
            # Sort by date
            sorted_days = sorted(st.session_state['pray_days_list'], key=lambda x: x['date'])

            # Prepare strings for st.pills
            # We need a way to map the display string back to the original item (date)
            # Unique mapping: date is unique? Yes, logical constraint in add_pray_day checks for duplicate date.

            # Create mapping: "01 Jun 2024 - Label" -> item
            # We use a dictionary to lookup later
            pills_options = []
            pills_map = {}

            for item in sorted_days:
                d_str = item['date']
                lbl = item['label']

                d_obj = pd.to_datetime(d_str).date()
                display_str = d_obj.strftime("%d %b %Y")
                if lbl:
                    display_str += f" - {lbl}"

                # Add visual "X" to indicate deletability
                display_str_with_x = f"{display_str}  ✕"

                pills_options.append(display_str_with_x)
                pills_map[display_str_with_x] = d_str

            st.caption("Tap a day to remove it from the list.")

            # Use st.pills for display and removal
            # We pass the full list as 'default', so they appear selected.
            # If the user clicks one, it becomes deselected.
            selected_pills = st.pills(
                "Pray Days",
                options=pills_options,
                default=pills_options,
                selection_mode="multi",
                label_visibility="collapsed",
                key="pray_days_pills"
            )

            # Logic to handle removal
            # If the length of selected_pills is less than our source list, something was removed.
            if len(selected_pills) < len(pills_options):
                # Find what is missing
                current_set = set(selected_pills)
                removed_display_str = None
                for opt in pills_options:
                    if opt not in current_set:
                        removed_display_str = opt
                        break

                if removed_display_str:
                    removed_date_str = pills_map[removed_display_str]
                    # Update session state
                    st.session_state['pray_days_list'] = [x for x in st.session_state['pray_days_list'] if x['date'] != removed_date_str]
                    save_pray_days_config(st.session_state['pray_days_list'])
                    st.rerun()
        else:
            st.info("No Pray Days configured.")

    st.markdown("---")
    st.header("Data Management")

    dm_c1, dm_c2 = st.columns(2)

    with dm_c1:
        st.subheader("Email Duplicates")

        # Display existing
        if os.path.exists("email_duplicates.csv"):
            with st.expander("View Current Duplicates"):
                try:
                    current_dupes = pd.read_csv("email_duplicates.csv")

                    if not current_dupes.empty:
                        st.dataframe(current_dupes, use_container_width=True, hide_index=True)
                    else:
                        st.info("No duplicates mapped yet.")

                except Exception as e:
                    st.error(f"Error reading file: {e}")
        else:
            st.info("No duplicates mapped yet.")

    with dm_c2:
        st.subheader("Person Merges")

        # Display existing
        if os.path.exists("person_merges.csv"):
            try:
                current_merges = pd.read_csv("person_merges.csv")
                merge_count = len(current_merges)
                merge_label = f"View Current Merges ({merge_count})"
            except:
                current_merges = pd.DataFrame()
                merge_label = "View Current Merges (Error)"

            with st.expander(merge_label):
                if not current_merges.empty:
                    st.dataframe(current_merges, use_container_width=True, hide_index=True)
        else:
            st.info("No merges configured yet.")


# --- ANALYSIS EXECUTION ---
# Since tabs are now top-level, we populate them when the button is pressed.

# We need a container to hold the results so they persist or we rely on session state?
# Streamlit reruns the whole script on interaction.
# If "Run Analysis" is a button, it returns True only once.
# So we should store the results in session_state if we want them to persist across tab switches
# (although tab switches don't necessarily trigger full rerun if on frontend, but interacting with Admin tab will).

if 'analysis_results' not in st.session_state:
    st.session_state['analysis_results'] = None

if st.sidebar.button("Run Analysis"):
    # Define input files
    base_input_files = ["initial.csv", "current.csv", "prayday1.csv", "prayday2.csv", "prayday3.csv", "prayday4.csv"]
    input_files = base_input_files + ["gwop.csv"] if include_gwop else base_input_files

    input_files = [f for f in input_files if os.path.exists(f)]

    if not input_files:
        st.error(f"Cannot run analysis. No valid input files found.")
    else:
        try:
            # Clean up temp dir
            if os.path.exists(OUTPUT_DIR):
                for f in os.listdir(OUTPUT_DIR):
                    os.remove(os.path.join(OUTPUT_DIR, f))
                os.rmdir(OUTPUT_DIR)

            with st.spinner("Running complex analysis..."):
                 # Get timestamps for config files
                config_timestamps = {}
                for f in ["email_duplicates.csv", "person_merges.csv"]:
                    if os.path.exists(f):
                        config_timestamps[f] = os.path.getmtime(f)

                results = run_full_analysis(
                    input_files,
                    OUTPUT_DIR,
                    start_date,
                    end_date,
                    goal_percentage,
                    config_timestamps
                )

                # Ensure output directory exists (fix for cached runs where dir was deleted)
                os.makedirs(OUTPUT_DIR, exist_ok=True)

                # Store results in session state
                st.session_state['analysis_results'] = results
                st.session_state['analysis_params'] = {
                    'include_gwop': include_gwop,
                    'exclude_gwop_new': exclude_gwop_new
                }

        except ValueError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"Error: {e}")

# Check if we have results to display
if st.session_state['analysis_results']:
    df, df_monthly_total, metrics_df, user_summary, likelihood_df, recently_stats, total_sessions_count, df_unfiltered = st.session_state['analysis_results']
    params = st.session_state.get('analysis_params', {})

    with tab_general:
        display_results(df, df_monthly_total, metrics_df, user_summary, likelihood_df, recently_stats, goal_percentage, OUTPUT_DIR, total_sessions_count)

    with tab_praydays:
         # Re-calculate pray_day_dates from session state to ensure it's fresh
        pray_day_dates_fresh = [pd.to_datetime(x['date']).date() for x in st.session_state['pray_days_list']]
        display_pray_day_results(df_unfiltered, pray_day_dates_fresh, exclude_gwop_new_logic=params.get('exclude_gwop_new', False))

else:
    with tab_general:
        st.info("Set your analysis parameters in the sidebar and click **Run Analysis** to generate the dashboard.")
    with tab_praydays:
        st.info("Run analysis to see Pray Day insights.")
# Force update
