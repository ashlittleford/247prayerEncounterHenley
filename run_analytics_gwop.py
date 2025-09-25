import prayer_analytics_lib as pal

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

# ----------------------------------------------------------------------
# DO NOT EDIT BELOW THIS LINE
# ----------------------------------------------------------------------

# The file(s) containing historical booking data.
INPUT_FILES = ["initial.csv", "current.csv", "gwop.csv"]

# The base name for the output directory.
# A date range suffix will be automatically appended to the folder name
# to prevent overwriting (e.g., "analysis_20240101_to_20241231").
OUTPUT_DIR = "gwop_out"

if __name__ == "__main__":
    try:
        print(f"Starting analysis with Input Files: {INPUT_FILES}")
        pal.run_analysis(
            input_files=INPUT_FILES, 
            outdir=OUTPUT_DIR, 
            start_filter=START_FILTER, 
            end_filter=END_FILTER
        )
        print("\nAnalysis complete. Check the output directory for results.")
    except Exception as e:
        print(f"\n--- FATAL ERROR --- \nAn error occurred during analysis: {e}")
        print("Please check your input files and date filters in run_analytics.py.")