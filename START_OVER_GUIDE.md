# Start Over Guide: "247prayer" Deployment

This guide explains how to set up the application from scratch on cPanel with the new name "247prayer".

## 1. Prepare the Directory

1.  **File Manager:** Go to your cPanel File Manager.
2.  **Create Folder:** Create a new folder named `247prayer` in your home directory (or inside `public_html` if you want it served directly, but home is safer for Python apps).
    *   Example: `/home/encosnpm/247prayer`
3.  **Upload Code:** Upload all the files from this repository into that `247prayer` folder.
    *   Ensure `app.py`, `passenger_wsgi.py`, `requirements.txt`, `static/`, `templates/`, and your `.csv` data files are in the root of this folder.

## 2. Setup Python App in cPanel

1.  **Go to "Setup Python App":** Search for this tool in cPanel.
2.  **Create Application:**
    *   **Python Version:** Select 3.8, 3.9, or newer.
    *   **Application Root:** Enter `247prayer` (the folder you just created).
    *   **Application URL:** select your domain and enter `247prayer` (or whatever path you want, e.g., `encounteradelaide.com.au/247prayer`).
    *   **Startup File:** Leave blank or enter `passenger_wsgi.py` (it is already configured).
    *   **Entry Point:** `application` (this is the default).
3.  **Click Create.**

## 3. Configure Dependencies

1.  Once created, scroll down to the "Configuration files" section.
2.  Type `requirements.txt` and click **Add**.
3.  Click **Run Pip Install** to install Flask, Pandas, etc.

## 4. Final Verification

1.  **Restart:** Click the **Restart** button in the Python App dashboard.
2.  **Test:** Visit your new URL: `encounteradelaide.com.au/247prayer` (or whichever URL you configured).
3.  **Diagnostics:** If it fails, try adding `/test` to the end of the URL (e.g., `encounteradelaide.com.au/247prayer/test`). It should say "Flask App is Running!".

## Troubleshooting

-   **404 Not Found:** Check that the "Application URL" in cPanel matches exactly what you are typing in the browser.
-   **500 Internal Server Error:** Check the `stderr.log` file inside the `247prayer` folder for error messages.
