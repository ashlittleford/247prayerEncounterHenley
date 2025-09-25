import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from io import BytesIO
from datetime import timedelta
import re 
# import streamlit_authenticator as stauth  <-- REMOVED

# Assuming prayer_analytics_lib.py is in the same directory
import prayer_analytics_lib as pal 

# --- CONFIGURATION & SETUP ---
LOGO_FILE = "logo.png"          
CUSTOM_LOGO_WIDTH = 100         

# 1. Set page config FIRST
st.set_page_config(layout="wide", page_title="Prayer Room Analytics Dashboard")


# ======================================================================
# 🔐 AUTHENTICATION BLOCK REMOVED 🔐
# The application will now start immediately.
# ======================================================================


# --- CUSTOM CSS INJECTION ---
CUSTOM_STYLE = """
<style>
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

/* 2. CENTERS THE LOGO IN THE SIDEBAR */
.st-emotion-cache-1mn4s5 { 
    display: flex;
    flex-direction: column;
    align-items: center;
    text-align: center;
}
</style>
"""
st.markdown(CUSTOM_STYLE, unsafe_allow_html=True)
# ---------------------------------------------------


# --- Streamlit Title and Logo ---

logo_path_exists = os.path.exists(LOGO_FILE)
if logo_path_exists:
    st.sidebar.image(LOGO_FILE, width=CUSTOM_LOGO_WIDTH)
else:
    st.sidebar.warning(f"Logo file '{LOGO_FILE}' not found. Please ensure it is in the same folder as the script.")
    
st.title("Prayer Room Analytics Dashboard")
st.markdown(f"Welcome to the **Prayer Room Analytics Dashboard**! Customize the analysis below and press **Run Analysis**.")
st.sidebar.markdown('---')


# --- 1. Sidebar for User Inputs (Customization) ---

st.sidebar.header("Configuration")

# Customization 1: Include GWOP Data
include_gwop = st.sidebar.checkbox("Include 'gwop.csv' data", value=True)

# Customization 2: Date Filters
st.sidebar.subheader("Date Range Filter")
start_date = st.sidebar.date_input("Start Date (optional)", value=None)
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

# Define a temporary in-memory output directory (required by the lib functions)
OUTPUT_DIR = "temp_dashboard_out" 


# --- Helper functions to run the library and display results ---

def format_change_metric(change_pct):
    """
    Ensures Green is positive, Red is negative for Streamlit's delta metrics.
    """
    if change_pct > 0:
        delta = f"{change_pct:.1f}%"
        delta_color = "inverse" 
    elif change_pct < 0:
        delta = f"{abs(change_pct):.1f}%"
        delta_color = "normal"  
    else:
        delta = "No Change"
        delta_color = "off"
    return delta, delta_color

@st.cache_data(show_spinner=False)
def run_full_analysis(input_files, outdir, start_filter, end_filter, goal_percentage):
    """
    Executes the analysis logic from prayer_analytics_lib.py.
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

    # Apply keys and calculate hours on the UNFILTERED data
    df_unfiltered["person_key"] = df_unfiltered.apply(pal.make_person_key, axis=1, args=(email_map,))
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
    
    # Return dataframes needed for display
    return df, df_monthly_total, metrics_df, user_summary, likelihood_df, recently_stats, total_sessions_count

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
    col_7, col_28 = st.columns(2)
    
    # Last 7 Days
    delta_7, delta_color_7 = format_change_metric(recently_stats["C7"])
    with col_7:
        st.subheader("Last 7 Days") 
        cols_h, cols_u = st.columns(2)
        cols_h.metric(
            label="Total Hours", 
            value=recently_stats["H7"], 
            delta=delta_7, 
            delta_color=delta_color_7
        )
        cols_u.metric(
            label="Unique Users", 
            value=recently_stats["U7"]
        )

    # Last 28 Days
    delta_28, delta_color_28 = format_change_metric(recently_stats["C28"])
    with col_28:
        st.subheader("Last 28 Days") 
        cols_h, cols_u = st.columns(2)
        cols_h.metric(
            label="Total Hours", 
            value=recently_stats["H28"], 
            delta=delta_28, 
            delta_color=delta_color_28
        )
        cols_u.metric(
            label="Unique Users", 
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


# --- 3. Main execution block with a Run button ---

if st.sidebar.button("Run Analysis"):
    # ... (File checking and analysis logic remains the same) ...
    # Define input files
    base_input_files = ["initial.csv", "current.csv"]
    input_files = base_input_files + ["gwop.csv"] if include_gwop else base_input_files
    
    # Check if files exist
    missing_files = [f for f in input_files if not os.path.exists(f)]
    if missing_files:
        st.error(f"Cannot run analysis. Missing input files: {', '.join(missing_files)}. Please upload them to the directory.")
    else:
        try:
            # Clean up the temporary directory before running
            if os.path.exists(OUTPUT_DIR):
                for f in os.listdir(OUTPUT_DIR):
                    os.remove(os.path.join(OUTPUT_DIR, f))
                os.rmdir(OUTPUT_DIR)
            
            with st.spinner("Running complex analysis..."):
                # Updated call to retrieve total_sessions_count
                df, df_monthly_total, metrics_df, user_summary, likelihood_df, recently_stats, total_sessions_count = run_full_analysis(
                    input_files, 
                    OUTPUT_DIR, 
                    start_date, 
                    end_date, 
                    goal_percentage
                )
        
            st.success(f"Analysis complete for **{len(df)}** bookings from **{df['start_time'].min().strftime('%Y-%m-%d')}** to **{df['start_time'].max().strftime('%Y-%m-%d')}**.")
            
            # Updated call to pass total_sessions_count
            display_results(df, df_monthly_total, metrics_df, user_summary, likelihood_df, recently_stats, goal_percentage, OUTPUT_DIR, total_sessions_count)
            
            # Clean up the temporary directory after display
            if os.path.exists(OUTPUT_DIR):
                for f in os.listdir(OUTPUT_DIR):
                    os.remove(os.path.join(OUTPUT_DIR, f))
                os.rmdir(OUTPUT_DIR)

        except ValueError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"An unexpected error occurred during analysis or display. Please check `prayer_analytics_lib.py`. Error: {e}")


# Initial state or instructions
if not st.session_state.get("df_loaded", False):
    st.info("Set your analysis parameters in the sidebar and click **Run Analysis** to generate the dashboard.")
    st.session_state["df_loaded"] = True