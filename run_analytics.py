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
# ⚠️ EDIT THESE PARAMETERS TO SET THE ANALYSIS CONFIGURATION ⚠️
# ----------------------------------------------------------------------

# Option to include data from 'gwop.csv'. Set to "YES" or "NO".
INCLUDE_GWOP = "YES"

# OPTIONAL: Define the START date for the analysis.
# - To use all data: set to None
# - To filter by a year: "2024"
# - To filter by a specific date: "2025-01-01"
START_FILTER = None

# OPTIONAL: Define the END date for the analysis.
# - To use data until today: set to None
# - To filter by a year: "2024"
# - To filter by a specific date: "2024-12-31"
END_FILTER = None

# The percentage goal for the hours filled (e.g., 10 for 10%).
# Setting this to None will result in no goal line being drawn on the charts.
GOAL_PERCENTAGE = 10 

# ----------------------------------------------------------------------
# DO NOT EDIT BELOW THIS LINE
# ----------------------------------------------------------------------

# Set input files and output directory based on the GWOP option
BASE_INPUT_FILES = ["initial.csv", "current.csv"]

if INCLUDE_GWOP.upper() == "YES":
    INPUT_FILES = BASE_INPUT_FILES + ["gwop.csv"]
    # Output directory suffix for GWOP inclusion
    OUTPUT_DIR = "gwop_out"
    gwop_status = "Included"
else:
    INPUT_FILES = BASE_INPUT_FILES
    # Output directory suffix for standard analysis
    OUTPUT_DIR = "out"
    gwop_status = "Excluded"


if __name__ == "__main__":
    print("--- Prayer Room Analysis ---")
    print(f"GWOP data: {gwop_status}")
    
    if START_FILTER or END_FILTER:
        start_display = START_FILTER if START_FILTER else "Earliest Data"
        end_display = END_FILTER if END_FILTER else "Today"
        print(f"Filter set: {start_display} to {end_display}")
    else:
        print("Filter set: Using all available data.")
        
    goal_display = f"{GOAL_PERCENTAGE}%" if GOAL_PERCENTAGE is not None else "None (No line)"
    print(f"Goal Percentage set to: {goal_display}")
        
    try:
        pal.run_analysis(
            input_files=INPUT_FILES, 
            outdir=OUTPUT_DIR, 
            start_filter=START_FILTER, 
            end_filter=END_FILTER,
            goal_percentage=GOAL_PERCENTAGE
        )
        print(f"\n✅ Analysis complete. Results saved to '{OUTPUT_DIR}'")
    except Exception as e:
        print(f"\n--- ❌ FATAL ERROR --- \nAn error occurred during analysis: {e}")
        print("Please check your input files and date filters in run_analytics.py.")