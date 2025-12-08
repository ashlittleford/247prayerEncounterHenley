import os
import json
import csv
import subprocess
import shutil
import time
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory, Response
import pandas as pd
import prayer_analytics_lib as pal

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
OUTPUT_DIR = 'static/analysis_output'
DATA_DIR = '.'  # Root of flask_app
FONTS_DIR = 'fonts'
LOGO_FILE = "static/images/logo.png"
PRAY_DAYS_FILE = "pray_days.json"

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Helper function to get current timestamp for cache busting
def get_timestamp():
    return int(time.time())

@app.template_filter('timestamp')
def timestamp_filter(s):
    return get_timestamp()

# --- ROUTES ---

@app.route('/')
def index():
    # Check for logo
    logo_exists = os.path.exists(LOGO_FILE)

    # Get last updated time for current.csv
    current_csv_path = os.path.join(DATA_DIR, "current.csv")
    last_updated = "Not found"
    if os.path.exists(current_csv_path):
        ts = os.path.getmtime(current_csv_path)
        # Assuming Adelaide time as per memory, but server time is simplest first.
        # Python's datetime.fromtimestamp uses local time.
        # To match "Australia/Adelaide", we might need pytz, but sticking to simple first.
        # Memory said: "must be displayed in the 'Australia/Adelaide' timezone."
        try:
            import pytz
            tz = pytz.timezone('Australia/Adelaide')
            dt = datetime.fromtimestamp(ts, tz)
            last_updated = dt.strftime('%Y-%m-%d %H:%M:%S')
        except ImportError:
            last_updated = datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')

    return render_template('index.html', logo_exists=logo_exists, last_updated=last_updated)

@app.route('/api/upload_current', methods=['POST'])
def upload_current():
    if 'file' not in request.files:
        return jsonify({'success': False, 'message': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No selected file'})

    if file:
        filepath = os.path.join(DATA_DIR, "current.csv")
        file.save(filepath)
        return jsonify({'success': True, 'message': 'current.csv updated successfully'})

    return jsonify({'success': False, 'message': 'Upload failed'})

@app.route('/api/config/pray_days', methods=['GET', 'POST', 'DELETE'])
def config_pray_days():
    filepath = os.path.join(DATA_DIR, PRAY_DAYS_FILE)

    if request.method == 'GET':
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                return jsonify(data)
            except:
                return jsonify([])
        return jsonify([])

    if request.method == 'POST':
        new_day = request.json
        data = []
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
            except:
                pass

        # Check duplicate
        if not any(d['date'] == new_day['date'] for d in data):
            data.append(new_day)
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=4)
            return jsonify({'success': True, 'data': data})
        else:
             return jsonify({'success': False, 'message': 'Date already exists'})

    if request.method == 'DELETE':
        date_to_remove = request.json.get('date')
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                data = [d for d in data if d['date'] != date_to_remove]
                with open(filepath, 'w') as f:
                    json.dump(data, f, indent=4)
                return jsonify({'success': True, 'data': data})
            except Exception as e:
                return jsonify({'success': False, 'message': str(e)})
        return jsonify({'success': False, 'message': 'File not found'})

@app.route('/api/config/duplicates', methods=['GET', 'POST'])
def config_duplicates():
    filepath = os.path.join(DATA_DIR, "email_duplicates.csv")

    if request.method == 'GET':
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath)
                return jsonify(df.to_dict(orient='records'))
            except:
                return jsonify([])
        return jsonify([])

    if request.method == 'POST':
        item = request.json
        primary = item.get('primary')
        additional = item.get('additional')

        if primary and additional:
            file_exists = os.path.exists(filepath)
            needs_newline = False
            if file_exists:
                with open(filepath, "r", encoding="utf-8") as f:
                    f.seek(0, os.SEEK_END)
                    if f.tell() > 0:
                        f.seek(f.tell() - 1, os.SEEK_SET)
                        if f.read(1) != "\n":
                            needs_newline = True

            with open(filepath, "a", newline='', encoding="utf-8") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["primary", "additional"])
                elif needs_newline:
                    f.write("\n")
                writer.writerow([primary.strip(), additional.strip()])

            return jsonify({'success': True})
        return jsonify({'success': False, 'message': 'Invalid data'})

@app.route('/api/config/merges', methods=['GET', 'POST'])
def config_merges():
    filepath = os.path.join(DATA_DIR, "person_merges.csv")

    if request.method == 'GET':
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath)
                return jsonify(df.to_dict(orient='records'))
            except:
                return jsonify([])
        return jsonify([])

    if request.method == 'POST':
        item = request.json
        source = item.get('source_key')
        target = item.get('target_key')

        if source and target:
            file_exists = os.path.exists(filepath)
            needs_newline = False
            if file_exists:
                with open(filepath, "r", encoding="utf-8") as f:
                    f.seek(0, os.SEEK_END)
                    if f.tell() > 0:
                        f.seek(f.tell() - 1, os.SEEK_SET)
                        if f.read(1) != "\n":
                            needs_newline = True

            with open(filepath, "a", newline='', encoding="utf-8") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["source_key", "target_key"])
                elif needs_newline:
                    f.write("\n")
                writer.writerow([source.strip(), target.strip()])

            return jsonify({'success': True})
        return jsonify({'success': False, 'message': 'Invalid data'})

@app.route('/api/git_sync', methods=['POST'])
def git_sync():
    filename = request.json.get('filename')
    message = request.json.get('message')

    # Check for GITHUB_TOKEN in env (User needs to set this in the environment where they run flask)
    # Streamlit had secrets, Flask uses env vars usually.
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        return jsonify({'success': False, 'message': 'GITHUB_TOKEN not found in environment.'})

    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):
         return jsonify({'success': False, 'message': f'File {filename} does not exist.'})

    try:
        remote_url_bytes = subprocess.check_output(["git", "remote", "get-url", "origin"])
        remote_url = remote_url_bytes.decode("utf-8").strip()
    except Exception as e:
         return jsonify({'success': False, 'message': f'Could not determine git remote: {str(e)}'})

    if "https://" in remote_url:
        repo_url = remote_url.replace("https://", f"https://{token}@")
    else:
        return jsonify({'success': False, 'message': 'Git remote URL format not supported (requires https).'})

    try:
        subprocess.run(["git", "config", "user.email", "flask-bot@example.com"], check=False)
        subprocess.run(["git", "config", "user.name", "Flask Bot"], check=False)
        subprocess.run(["git", "add", filepath], check=True, capture_output=True, text=True)

        status = subprocess.run(["git", "status", "--porcelain", filepath], capture_output=True, text=True)
        if not status.stdout.strip():
            return jsonify({'success': True, 'message': "No changes to commit."})

        subprocess.run(["git", "commit", "-m", message], check=True, capture_output=True, text=True)
        subprocess.run(["git", "pull", "--rebase", repo_url, "main"], check=True, capture_output=True, text=True)
        subprocess.run(["git", "push", repo_url, "HEAD:main"], check=True, capture_output=True, text=True)

        return jsonify({'success': True, 'message': "Changes committed and pushed successfully!"})
    except subprocess.CalledProcessError as e:
        return jsonify({'success': False, 'message': f"Git error: {e.stderr if e.stderr else str(e)}"})
    except Exception as e:
        return jsonify({'success': False, 'message': f"Error: {str(e)}"})


@app.route('/api/run_analysis', methods=['POST'])
def run_analysis_route():
    params = request.json
    include_gwop = params.get('include_gwop', True)
    exclude_gwop_new = params.get('exclude_gwop_new', False)
    start_date = params.get('start_date')
    end_date = params.get('end_date')
    goal_percentage = params.get('goal_percentage')

    if goal_percentage == 0:
        goal_percentage = None

    # Base Files
    base_input_files = ["initial.csv", "current.csv", "prayday1.csv", "prayday2.csv", "prayday3.csv", "prayday4.csv"]
    input_files = base_input_files + ["gwop.csv"] if include_gwop else base_input_files
    # Adjust paths to DATA_DIR
    input_files = [os.path.join(DATA_DIR, f) for f in input_files if os.path.exists(os.path.join(DATA_DIR, f))]

    if not input_files:
        return jsonify({'success': False, 'message': 'No input files found'})

    try:
        # Clean output dir
        if os.path.exists(OUTPUT_DIR):
            shutil.rmtree(OUTPUT_DIR)
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # Call the library
        # We need to adapt the library call slightly or just use the raw functions as in dashboard_app.py's run_full_analysis

        # 1. Load Data
        df_unfiltered = pal.load_and_combine_csvs(input_files)
        df_unfiltered = pal.normalize_columns(df_unfiltered)

        today = pd.Timestamp.now().normalize()
        data_min_date = df_unfiltered["start_time"].min().normalize()

        # Load helpers
        email_map = {}
        if os.path.exists(os.path.join(DATA_DIR, "email_duplicates.csv")):
            email_dupes_df = pd.read_csv(os.path.join(DATA_DIR, "email_duplicates.csv"))
            for _, row in email_dupes_df.iterrows():
                p = str(row["primary"] or "").strip().lower()
                a = str(row["additional"] or "").strip().lower()
                if p and a: email_map[a] = p

        merge_map = {}
        if os.path.exists(os.path.join(DATA_DIR, "person_merges.csv")):
             pm_df = pd.read_csv(os.path.join(DATA_DIR, "person_merges.csv"))
             for _, row in pm_df.iterrows():
                 if pd.notna(row['source_key']) and pd.notna(row['target_key']):
                    merge_map[row['source_key'].strip()] = row['target_key'].strip()

        # Keys & Merges
        df_unfiltered["person_key"] = df_unfiltered.apply(pal.make_person_key, axis=1, args=(email_map,))
        if merge_map:
            df_unfiltered["person_key"] = df_unfiltered["person_key"].replace(merge_map)

        # Re-apply to ensure consistency if needed, but merge is usually enough.
        # The original code re-ran make_person_key before merging?
        # Original: make_person_key -> replace(merge_map) -> make_person_key AGAIN (probably redundant/typo in original but I'll stick to logic that works)
        # Actually original code did: 1. make_person_key 2. replace 3. make_person_key 4. replace.  Let's just do it once properly.
        # Actually, let's follow the logic: Map Emails -> Generate Key -> Map Merges.

        df_unfiltered["person_name"] = df_unfiltered.apply(
            lambda r: "UNKNOWN"
            if r["person_key"] == "UNKNOWN"
            else f"{str(r['firstname'] or '').strip()} {str(r['lastname'] or '').strip()}".strip(),
            axis=1,
        )
        df_unfiltered = pal.calculate_hours(df_unfiltered)

        # Recent stats (Unfiltered)
        h_7, u_7, c_7 = pal.calculate_time_period_stats(df_unfiltered, 7)
        h_28, u_28, c_28 = pal.calculate_time_period_stats(df_unfiltered, 28)

        recent_stats = {
            "H7": h_7, "U7": u_7, "C7": c_7,
            "H28": h_28, "U28": u_28, "C28": c_28
        }

        # Filter
        filter_start = pd.to_datetime(start_date) if start_date else data_min_date
        filter_end = pd.to_datetime(end_date) if end_date else today

        if filter_start > filter_end:
            filter_start, filter_end = filter_end, filter_start

        df = df_unfiltered[(df_unfiltered["start_time"].dt.normalize() >= filter_start) & (df_unfiltered["start_time"].dt.normalize() <= filter_end)].copy()
        df = df[df["start_time"] <= today]
        df = df.dropna(subset=["start_time", "end_time", "person_key"])

        if df.empty:
            return jsonify({'success': False, 'message': 'No data in selected date range.'})

        # Analysis
        df_monthly_total = df.groupby(pd.Grouper(key="start_time", freq="MS"))["duration"].sum()

        metrics = pal.calculate_dashboard_metrics(df, df_monthly_total)
        user_summary = pal.export_user_summary_to_csv(df, OUTPUT_DIR)
        regular_metrics = pal.calculate_regulars_metrics(df, user_summary, OUTPUT_DIR)
        fortnightly_metrics = pal.calculate_fortnightly_regulars_metrics(df, user_summary, OUTPUT_DIR)
        average_metrics = pal.calculate_average_metrics(df)

        all_metrics = metrics + regular_metrics + fortnightly_metrics + average_metrics
        metrics_dict = {m['Metric']: m['Value'] for m in all_metrics}

        likelihood_df = pal.export_weekly_likelihood_metrics(df, OUTPUT_DIR)

        # Plots
        pal.plot_weekly_total_hours(df, OUTPUT_DIR, goal_percentage)
        pal.plot_monthly_total_hours(df_monthly_total, OUTPUT_DIR, goal_percentage)
        pal.plot_hourly_distribution(df, OUTPUT_DIR)
        pal.plot_binary_hourly_distribution(df, OUTPUT_DIR)
        pal.plot_avg_hours_distribution(user_summary, OUTPUT_DIR)

        # Pray Days Analysis (using Unfiltered data)
        # Load pray days config
        pray_days_list = []
        if os.path.exists(os.path.join(DATA_DIR, PRAY_DAYS_FILE)):
             with open(os.path.join(DATA_DIR, PRAY_DAYS_FILE), 'r') as f:
                 pray_days_list = json.load(f)

        pray_day_dates = [pd.to_datetime(x['date']).date() for x in pray_days_list]
        pd_results = pal.analyze_pray_days(df_unfiltered, pray_day_dates, exclude_gwop_from_new_logic=exclude_gwop_new)

        # Format PD results for JSON
        pd_summary = pd_results['summary_df'].copy()
        if not pd_summary.empty:
            pd_summary['Date'] = pd_summary['Date'].astype(str)

        pd_breakdown = pd_results['breakdown_details'].copy()
        if not pd_breakdown.empty:
            pd_breakdown['Date'] = pd_breakdown['Date'].astype(str)
            # Save breakdown to CSV for download
            pd_breakdown.to_csv(os.path.join(OUTPUT_DIR, 'pray_day_breakdown.csv'), index=False)

        newbies_details = {}
        for d, det_df in pd_results.get('newbies_details', {}).items():
            newbies_details[str(d)] = det_df.to_dict(orient='records')

        # Prepare Response
        response_data = {
            'success': True,
            'recent_stats': recent_stats,
            'metrics': metrics_dict,
            'total_sessions': len(df),
            'weekly_regulars': user_summary[user_summary["weekly_avg_hours"] >= 1][["person_name", "weekly_avg_hours"]].sort_values("weekly_avg_hours", ascending=False).to_dict(orient='records'),
            'fortnightly_regulars': user_summary[user_summary["avg_hours_per_fortnight"] >= 1][["person_name", "weekly_avg_hours"]].sort_values("weekly_avg_hours", ascending=False).to_dict(orient='records'), # Note: frontend asked for weekly avg for these too
            'slot_likelihood': likelihood_df.rename(columns={"hour": "Hour (24hr)"}).to_dict(orient='records'),
            'pray_days': {
                'total_participants': int(pd_results.get("total_unique_participants", 0)),
                'total_repeaters': int(pd_results.get("repeaters_count", 0)),
                'summary': pd_summary.to_dict(orient='records'),
                'newbies_details': newbies_details
            },
            'output_dir': OUTPUT_DIR # Client can construct URLs
        }

        return jsonify(response_data)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)})


@app.route('/api/download/<path:filename>')
def download_file(filename):
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True, port=8501)
