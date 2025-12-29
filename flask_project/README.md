# Flask Dashboard Deployment Guide

This directory contains a standalone Flask application that replaces the previous Streamlit dashboard. It is designed to be hosted on standard Python hosting environments like cPanel (using Passenger/WSGI), Heroku, or a VPS.

## Prerequisites

- Python 3.8+
- A hosting environment that supports Python (e.g., cPanel with Python Setup, Heroku).

## Local Development

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the App:**
    ```bash
    python app.py
    ```

3.  **Access:**
    Open `http://127.0.0.1:5000` in your browser.

## Deployment on cPanel (with Passenger)

1.  **Upload Files:**
    Upload all files in this directory to your cPanel hosting (e.g., via File Manager or Git).

2.  **Setup Python App:**
    - Go to **"Setup Python App"** in cPanel.
    - Click **"Create Application"**.
    - **Python Version**: Select 3.8 or newer.
    - **Application Root**: The folder where you uploaded these files.
    - **Application URL**: The domain/subdomain you want to use.
    - **Application Startup File**: `passenger_wsgi.py` (This file is already included and configured).
    - **Application Entry Point**: `application` (This is the default callable in `passenger_wsgi.py`).
    - Click **Create**.

3.  **Install Dependencies:**
    - In the Python App settings (where you just created the app), look for the "Configuration files" section.
    - Ensure `requirements.txt` is detected.
    - Click **"Run Pip Install"** to install Pandas, Flask, Matplotlib, etc.

4.  **Environment Variables (Optional but Recommended for Git Sync):**
    - If you want the "Pray Day" configuration changes to be synced back to GitHub automatically (like in the previous app), you need to set the `GITHUB_TOKEN` environment variable.
    - You can usually set Environment Variables in the cPanel "Setup Python App" interface.
    - Key: `GITHUB_TOKEN`
    - Value: Your GitHub Personal Access Token.

5.  **Restart:**
    - Click **"Restart"** to launch the application.

## Troubleshooting

- **Images not showing?** ensure the `static/plots` directory is writable by the web server user. The app attempts to create it if it doesn't exist.
- **500 Internal Server Error?** Check the `stderr.log` in your application root directory for Python errors.

## Key Files

- `app.py`: The main Flask application and route definitions.
- `logic.py`: Contains the core data processing logic (migrated from the old Streamlit app).
- `prayer_analytics_lib.py`: The shared library for data analysis and plotting.
- `passenger_wsgi.py`: The entry point for cPanel/Passenger servers.
- `templates/`: HTML files for the frontend.
- `static/`: CSS, fonts, and generated plot images.
