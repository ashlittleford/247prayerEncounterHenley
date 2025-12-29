from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory, jsonify, session
import os
import pandas as pd
import logic
import prayer_analytics_lib as pal
import matplotlib
# Set backend to Agg before importing pyplot to avoid GUI errors
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)
app.secret_key = os.urandom(24) # Secure random key for session

# Configuration
UPLOAD_FOLDER = 'uploads' # Not used directly, files saved to root
STATIC_PLOTS_DIR = os.path.join('static', 'plots')
app.config['STATIC_PLOTS_DIR'] = STATIC_PLOTS_DIR

# Ensure directories exist
os.makedirs(app.config['STATIC_PLOTS_DIR'], exist_ok=True)

# Default Settings
DEFAULT_START_DATE = "2025-01-05"
DEFAULT_GOAL = 10

def get_settings():
    return {
        'start_date': session.get('start_date', DEFAULT_START_DATE),
        'end_date': session.get('end_date', None),
        'goal_percentage': session.get('goal_percentage', DEFAULT_GOAL),
        'include_gwop': session.get('include_gwop', True),
        'exclude_gwop_new': session.get('exclude_gwop_new', False)
    }

def get_input_files(include_gwop):
    base = ["initial.csv", "current.csv", "prayday1.csv", "prayday2.csv", "prayday3.csv", "prayday4.csv"]
    if include_gwop:
        base.append("gwop.csv")
    return [f for f in base if os.path.exists(f)]

@app.route('/')
def index():
    settings = get_settings()
    input_files = get_input_files(settings['include_gwop'])

    # Run Analysis
    # Note: Logic saves plots to outdir. We need to point outdir to static/plots
    # Cleaning up old plots might be necessary or we overwrite

    results = logic.run_full_analysis(
        input_files,
        app.config['STATIC_PLOTS_DIR'],
        settings['start_date'],
        settings['end_date'],
        settings['goal_percentage']
    )

    if not results:
        flash("No data found for the selected range.", "warning")
        return render_template('index.html', settings=settings)

    # Generate Plots
    # The logic function calls pal functions which save plots to outdir
    # We need to manually trigger plot generation here if logic didn't do it or if we need to pass the DF
    # logic.run_full_analysis calls pal which generates:
    # dashboard_summary.csv, user.csv, top_10_slots.csv...
    # BUT logic.py as I wrote it does NOT call the plotting functions directly except inside run_full_analysis?
    # Let's check logic.py... It does NOT call plotting functions. pal.calculate_dashboard_metrics etc are called.
    # We need to call plotting functions here using the results.

    df = results['df']
    df_monthly = results['df_monthly_total']
    user_summary = results['user_summary']
    goal = int(settings['goal_percentage']) if settings['goal_percentage'] else None

    pal.plot_weekly_total_hours(df, app.config['STATIC_PLOTS_DIR'], goal)
    pal.plot_monthly_total_hours(df_monthly, app.config['STATIC_PLOTS_DIR'], goal)
    pal.plot_hourly_distribution(df, app.config['STATIC_PLOTS_DIR'])
    pal.plot_binary_hourly_distribution(df, app.config['STATIC_PLOTS_DIR'])
    pal.plot_avg_hours_distribution(user_summary, app.config['STATIC_PLOTS_DIR'])

    # Prepare metrics for template
    metrics_list = results['metrics_df'].to_dict('records')
    metrics_dict = {m['Metric']: m['Value'] for m in metrics_list}

    # Extract specific metrics for dashboard cards
    dashboard_data = {
        'total_hours': metrics_dict.get("Total Hours of Prayer", "N/A"),
        'total_users': metrics_dict.get("Total Individual Users", "N/A"),
        'avg_hours_week': metrics_dict.get("Avg Hours per Week (Filtered Period)", "N/A"),
        'weekly_regulars': metrics_dict.get("Number of Weekly Regulars", "N/A"),
        'fortnightly_regulars': metrics_dict.get("Number of Fortnightly Regulars", "N/A"),
        'total_sessions': results['total_sessions_count']
    }

    # Top Booking Timeslots Logic for Display
    top_slots = []
    for i in range(1, 4):
         key = f"Top {i} Timeslot"
         if key in metrics_dict:
             val = metrics_dict[key]
             # Clean string
             val = val.split(' with ')[0].strip()
             top_slots.append({'rank': i, 'value': val, 'full': metrics_dict[key]})

    return render_template(
        'index.html',
        settings=settings,
        data=dashboard_data,
        metrics=metrics_dict,
        recent=results['recently_stats'],
        weekly_regulars=user_summary[user_summary["weekly_avg_hours"] >= 1].sort_values("weekly_avg_hours", ascending=False).to_dict('records'),
        fortnightly_regulars=user_summary[user_summary["avg_hours_per_fortnight"] >= 1].sort_values("weekly_avg_hours", ascending=False).to_dict('records'),
        likelihood=results['likelihood_df'].head(10).to_dict('records'),
        top_slots=top_slots,
        plots=[
            "weekly_total_hours.png", "monthly_total_hours.png",
            "hourly_distribution.png", "hourly_users.png",
            "avg_hours_distribution.png"
        ]
    )

@app.route('/pray-days')
def pray_days():
    settings = get_settings()
    input_files = get_input_files(settings['include_gwop'])

    # Need unfiltered data for pray days
    # We can reuse logic.run_full_analysis but it might be overkill if we cache?
    # For now, let's run it.
    results = logic.run_full_analysis(
        input_files,
        app.config['STATIC_PLOTS_DIR'],
        settings['start_date'],
        settings['end_date'],
        settings['goal_percentage']
    )

    if not results:
         flash("Error running analysis.", "danger")
         return redirect(url_for('index'))

    df_unfiltered = results['df_unfiltered']
    pray_days_config = logic.load_pray_days_config()
    pray_day_dates = [pd.to_datetime(x['date']).date() for x in pray_days_config]

    # Analyze
    pd_results = pal.analyze_pray_days(df_unfiltered, pray_day_dates, exclude_gwop_from_new_logic=settings['exclude_gwop_new'])

    # Transpose Summary for Display (Date as Columns)
    summary_df = pd_results['summary_df']
    display_table = None
    if not summary_df.empty:
        display_df = summary_df.set_index("Date").T
        # Format columns
        display_df.columns = [logic.format_date_custom(pd.to_datetime(c)) for c in display_df.columns]
        display_table = display_df.to_html(classes="table table-striped table-bordered", border=0)

    # Newbies Details
    newbies = pd_results.get('newbies_details', {})
    # Convert dict of dfs to dict of html tables
    newbies_html = {k.isoformat(): v.to_html(classes="table table-sm", index=False) for k, v in newbies.items() if not v.empty}

    return render_template(
        'pray_days.html',
        total_participants=pd_results.get("total_unique_participants", 0),
        total_repeaters=pd_results.get("repeaters_count", 0),
        latest_newbies=summary_df.sort_values("Date").iloc[-1]["Total New Users"] if not summary_df.empty else 0,
        summary_table=display_table,
        newbies_details=newbies_html
    )

@app.route('/admin', methods=['GET', 'POST'])
def admin():
    settings = get_settings()
    pray_days = logic.load_pray_days_config()

    # Load Duplicates and Merges for Read-Only View
    duplicates = []
    if os.path.exists("email_duplicates.csv"):
        duplicates = pd.read_csv("email_duplicates.csv").to_dict('records')

    merges = []
    if os.path.exists("person_merges.csv"):
        merges = pd.read_csv("person_merges.csv").to_dict('records')

    return render_template('admin.html', settings=settings, pray_days=pray_days, duplicates=duplicates, merges=merges)

@app.route('/update-settings', methods=['POST'])
def update_settings():
    session['start_date'] = request.form.get('start_date') or None
    session['end_date'] = request.form.get('end_date') or None
    try:
        session['goal_percentage'] = int(request.form.get('goal_percentage'))
    except:
        session['goal_percentage'] = DEFAULT_GOAL

    session['include_gwop'] = 'include_gwop' in request.form
    session['exclude_gwop_new'] = 'exclude_gwop_new' in request.form

    flash("Settings updated.", "success")
    return redirect(url_for('admin')) # Or back to referrer

@app.route('/upload-csv', methods=['POST'])
def upload_csv():
    if 'file' not in request.files:
        flash('No file part', 'danger')
        return redirect(url_for('admin'))
    file = request.files['file']
    if file.filename == '':
        flash('No selected file', 'danger')
        return redirect(url_for('admin'))

    # Save as current.csv
    try:
        file.save("current.csv")
        flash("current.csv successfully updated!", "success")
    except Exception as e:
        flash(f"Error saving file: {e}", "danger")

    return redirect(url_for('admin'))

@app.route('/add-pray-day', methods=['POST'])
def add_pray_day():
    date_str = request.form.get('date')
    label = request.form.get('label', '')

    if date_str:
        days = logic.load_pray_days_config()
        if any(d['date'] == date_str for d in days):
            flash("Date already exists.", "warning")
        else:
            days.append({'date': date_str, 'label': label})
            success, msg = logic.save_pray_days_config(days)
            if success:
                flash("Pray Day added.", "success")
            else:
                flash(f"Pray Day added locally but Git sync failed: {msg}", "warning")

    return redirect(url_for('admin'))

@app.route('/delete-pray-day', methods=['POST'])
def delete_pray_day():
    date_str = request.form.get('date')
    days = logic.load_pray_days_config()
    new_days = [d for d in days if d['date'] != date_str]

    if len(new_days) < len(days):
        success, msg = logic.save_pray_days_config(new_days)
        if success:
            flash("Pray Day removed.", "success")
        else:
             flash(f"Pray Day removed locally but Git sync failed: {msg}", "warning")

    return redirect(url_for('admin'))

if __name__ == '__main__':
    app.run(debug=True, port=5000)
