# Deployment and Access Guide

## Accessing the Live Application

The application is deployed at:
[https://prayerstats.encounteradelaide.com.au](https://prayerstats.encounteradelaide.com.au)

## Uploading Data

### Data Update (`current.csv`)

To upload the latest `current.csv` file:

1.  Open the application URL.
2.  In the sidebar (left panel), locate the **Data Management** section.
3.  Click "Browse files" under "Upload new current.csv".
4.  Select your `current.csv` file from your computer.
5.  Click **Replace current.csv with uploaded file**.

### Configuration

You can configure "Pray Days", "Email Duplicates", and "Person Merges" in the **Admin** tab of the application.

## Deploying Code Changes

The application runs on a cPanel hosting environment. You can deploy changes using the **Terminal** or manage files via the **File Manager**.

### Option 1: Using the Terminal (Recommended for Git)

1.  **Open the Terminal:**
    Log in to cPanel and search for "Terminal" to open the command line interface.

2.  **Navigate to the Application Folder:**
    The application resides in `/home/encosnpm/prayerstats.encounteradelaide.com.au/prayer_dashboard/Prayer-Analysis`.

    Run the following command to go there:
    ```bash
    cd /home/encosnpm/prayerstats.encounteradelaide.com.au/prayer_dashboard/Prayer-Analysis
    ```

3.  **Pull Latest Code:**
    Ensure your local changes are pushed to GitHub, then run:
    ```bash
    git pull origin main
    ```

4.  **Restart the Application:**
    To apply the changes, touch the `passenger_wsgi.py` file:
    ```bash
    touch passenger_wsgi.py
    ```

### Option 2: Using cPanel File Manager

1.  **Open File Manager:**
    Log in to cPanel and click on **File Manager**.

2.  **Navigate to the Folder:**
    -   Locate and open the folder `prayerstats.encounteradelaide.com.au`.
    -   Inside that, open the `prayer_dashboard` folder.
    -   Finally, open the `Prayer-Analysis` folder.

    This is where the application files (like `dashboard_app.py` and `current.csv`) are stored.

3.  **Restarting via File Manager:**
    To restart the app without the terminal, you can locate `passenger_wsgi.py`, edit it (e.g., add a space and delete it), and save. Alternatively, use the **Setup Python App** tool in cPanel and click "Restart".

## Troubleshooting

-   **Application Error:** Check the `stderr.log` in the application root directory (visible in File Manager).
-   **Images Missing:** Ensure the `static/plots` directory exists and is writable.
