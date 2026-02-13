# Deployment Guide & Troubleshooting

This guide explains how to fix, refresh, or start over with your deployment of the 24-7 Prayer Analytics Dashboard.

## 1. Identify Your Platform
Based on the error screenshot provided ("deploy failed for 247prayerEncounterHenley" from Render), your application is currently configured to deploy to **Render**.

- **Render**: A modern cloud platform that automatically builds and deploys your code from GitHub.
- **cPanel**: A traditional web hosting control panel (which you mentioned using previously).

If you are seeing emails from Render, the live version of your site is likely hosted there.

## 2. Fixing the Render Deployment (Recommended)
The most common cause for Render deployment failures is a missing start command. We have fixed this by adding a `Procfile`.

### Steps to Fix:
1.  **Push the latest changes** to GitHub.
    - The new `Procfile` tells Render how to start the app (`gunicorn app:app`).
    - The `requirements.txt` file ensures all dependencies (Flask, Pandas, Gunicorn) are installed.
2.  **Check Render Dashboard**:
    - Go to your Render Dashboard.
    - Click on the service `247prayerEncounterHenley`.
    - Look at the **Events** or **Logs** tab.
    - You should see a new deployment starting.
    - If it fails again, read the error log. Common errors include:
        - "Build failed": Usually a dependency issue.
        - "Deploy failed": Usually a startup crash (check "Logs" for python errors).

### Troubleshooting "Deploy Failed"
If the deployment still fails:
- **Check Logs**: In Render, click "Logs". Look for lines starting with `Error:` or `Traceback`.
- **Python Version**: Ensure your Render service is using Python 3.10 or higher. You can set this in the "Environment" tab or by adding a `runtime.txt` file with `python-3.11.5`.
- **Environment Variables**: Ensure `GITHUB_TOKEN` is set in the Render "Environment" settings if your app needs to push data back to GitHub (for saving settings/Pray Days).

## 3. Refreshing or Restarting Data
If the application is running but the data is wrong or you want to start fresh:

1.  **Use the Admin Panel**:
    - Go to `/admin` on your deployed site.
    - Use the "Upload CSV" feature to upload a new `current.csv`.
2.  **Reset via Git** (Advanced):
    - If you want to completely wipe data, you can delete `current.csv` from the GitHub repository and commit the change. The app will revert to `initial.csv` or an empty state.

## 4. Starting Over (Re-deploying from Scratch)
If you want to completely delete the current deployment and start fresh:

### Option A: Fresh Start on Render
1.  **Delete Service**: In Render dashboard, go to Settings -> Delete Service.
2.  **New Web Service**:
    - Click "New +" -> "Web Service".
    - Connect your GitHub repository.
    - **Name**: `247prayer` (or similar).
    - **Runtime**: `Python 3`.
    - **Build Command**: `pip install -r requirements.txt`.
    - **Start Command**: `gunicorn app:app`.
    - **Plan**: Free (or as needed).
    - **Advanced**: Add Environment Variable `GITHUB_TOKEN` (if you need the app to save config changes back to repo).
3.  **Deploy**: Click "Create Web Service".

### Option B: Deploying to cPanel (If you prefer cPanel)
1.  **Login to cPanel**.
2.  **Python App Setup**:
    - Go to "Setup Python App".
    - Create New Application.
    - **Python Version**: 3.10+ (Recommended).
    - **App Directory**: `prayer_dashboard` (or where you cloned the repo).
    - **App Domain/URI**: select your domain.
    - **Application Startup File**: `passenger_wsgi.py`.
    - **Application Entry Point**: `application`.
3.  **Install Dependencies**:
    - Enter the virtual environment command provided by cPanel (e.g., `source .../bin/activate`).
    - Run: `pip install -r requirements.txt`.
4.  **Restart**: Click "Restart" in the Python App interface.

## Summary of Changes Made
- **Added `Procfile`**: Validates the start command for Render (`gunicorn app:app`).
- **Updated `requirements.txt`**: Ensures `gunicorn` is listed.
- **Code Check**: Verified `app.py` and logic files are consistent.
