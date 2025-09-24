import prayer_analytics_lib as pa_lib

def main():
    input_files = ["initial.csv", "current.csv"]
    outdir = "out"
    pa_lib.run_analysis(input_files, outdir)

if __name__ == "__main__":
    main()