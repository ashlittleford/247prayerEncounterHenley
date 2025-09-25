import prayer_analytics_lib as pal
import sys

# Ensure pandas is available before running
try:
    import pandas as pd
except ImportError:
    print("Error: pandas and/or matplotlib libraries not found.")
    print("Please install them using: pip install pandas matplotlib")
    sys.exit(1)

# ----------------------------------------------------------------------
# ⚠️ EDIT THESE PARAMETERS TO SET THE DATE RANGE FOR YOUR ANALYSIS ⚠️
# ----------------------------------------------------------------------

# OPTIONAL: Define the START date for the analysis.
# - To use all data: set to None
# - To filter by a year: "2024" (uses Jan 1st to Dec 31st)
# - To filter by a specific date: "2025-01-01"
START_FILTER = None

# OPTIONAL: Define the END date for the analysis.
# - To use data until today: set to None
# - To filter by a year: "2024" (uses Jan 1st to Dec 31st)
# - To filter by a specific date: "2024-12-31"
END_FILTER = None

# The percentage goal for the hours filled (e.g., 10 for 10%)
# Setting this to None will result in no goal line being drawn on the charts.
GOAL_PERCENTAGE = 10

# ----------------------------------------------------------------------
# DO NOT EDIT BELOW THIS LINE
# ----------------------------------------------------------------------

# The file(s) containing historical booking data.
# E.g., ["initial.csv", "current.csv"]
INPUT_FILES = ["initial.csv", "current.csv"]

# The base name for the output directory.
# A date range suffix will be automatically appended (e.g., "analysis_20240101_to_20241231").
OUTPUT_DIR = "out"

if __name__ == "__main__":
    print("--- Prayer Analysis Script ---")
    
    if START_FILTER or END_FILTER:
        start_display = START_FILTER if START_FILTER else "Earliest Data"
        end_display = END_FILTER if END_FILTER else "Today"
        print(f"Filter set: {start_display} to {end_display}")
    else:
        print("Filter set: Using all available data.")
        
    if GOAL_PERCENTAGE is None:
        print("Goal Percentage set to: None (No goal line will be drawn)")
    else:
        print(f"Goal Percentage set to: {GOAL_PERCENTAGE}%")
        
    try:
        pal.run_analysis(
            input_files=INPUT_FILES, 
            outdir=OUTPUT_DIR, 
            start_filter=START_FILTER, 
            end_filter=END_FILTER,
            goal_percentage=GOAL_PERCENTAGE
        )
        print("\n✅ Analysis complete. Check the output directory for results.")
    except Exception as e:
        print(f"\n--- ❌ FATAL ERROR --- \nAn error occurred during analysis: {e}")
        print("Please check your input files and date filters in run_analytics.py.")