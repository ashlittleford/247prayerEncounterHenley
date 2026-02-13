# Deployment Instructions

You have successfully migrated the codebase to the new Flask application structure. To finalize the deployment on your cPanel hosting:

1.  **Pull the latest changes:**
    Go to your git repository directory on the server and run:
    ```bash
    git pull
    ```

2.  **Install/Update Dependencies:**
    The new application requires Flask and other libraries. Run:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: If you are using the "Setup Python App" interface in cPanel, you can also click "Run Pip Install" in the configuration section.*

3.  **Restart the Application:**
    Restart the application to load the new code.
    - **Option A (Command Line):** Touch the restart file:
      ```bash
      touch passenger_wsgi.py
      ```
    - **Option B (cPanel UI):** Go to "Setup Python App", find your application, and click **"Restart"**.

4.  **Verify:**
    Visit your website URL (e.g., `encounteradelaide.com.au/prayerinsights/` or `prayerstats.encounteradelaide.com.au`) to verify the new dashboard is loading.

## Notes
- The old Streamlit application files have been moved to the `legacy_streamlit/` directory for backup.
- The `current.csv` and other data files in the root directory were preserved.
- If you see a "500 Internal Server Error", check the `stderr.log` file in the root directory for details.
