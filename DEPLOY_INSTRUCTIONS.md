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
    Visit your website URL to verify the new dashboard is loading.

---

## Troubleshooting

### 1. "This site can't be reached" (DNS_PROBE_FINISHED_NXDOMAIN)
If you see this error when accessing `prayerstats.encounteradelaide.com.au`:
- **Cause:** The domain name `prayerstats.encounteradelaide.com.au` does not exist or has no DNS records.
- **Solution:** You must create a **Subdomain** in cPanel or add a DNS `A` record or `CNAME` for `prayerstats` pointing to your server's IP address. This is a networking configuration, not a code issue.

### 2. "OOPS! The page you are looking for is no longer here" (404)
If you see this error when accessing `encounteradelaide.com.au/prayerinsights/`:
- **Cause:** This is a WordPress 404 page. It means the request is hitting your main website instead of the Python application.
- **Solution:**
    - Ensure your Python App in cPanel is "Started".
    - Check the "Application URL" setting in "Setup Python App". It must match the URL you are trying to visit.
    - If you are deploying to a subfolder (e.g. `/prayerinsights`), ensure the route is mapped correctly in cPanel.

### 3. Verify the App is Running
I have added a special test page to confirm the application code is active.
Try visiting: `your-url/test` (e.g., `encounteradelaide.com.au/prayerinsights/test` or `prayerstats.encounteradelaide.com.au/test`)
- If you see **"Flask App is Running!"**, the app is working correctly, and the issue might be with data or the main page.
- If you see a 404 or error, the app is not receiving the request.

### 4. "500 Internal Server Error"
- Check the `stderr.log` file in the root directory for Python error details.

### 5. GLOBAL OUTAGE: All Python Sites Down / Giving 404
If multiple Python websites hosted on the same server suddenly stop working and show 404 errors, this indicates a server-level configuration issue.

**Likely Causes:**
1.  **cPanel Configuration Reset:** The "Setup Python App" configuration might have been lost or corrupted during a server update.
2.  **Web Server Configuration Change:** The `.htaccess` file handling routing to Python apps might have been overwritten or deleted.
3.  **Python Version Change:** The system Python version might have been updated, breaking existing virtual environments.

**Steps to Fix:**
1.  **Check "Setup Python App" in cPanel:**
    - Log in to cPanel.
    - Go to "Setup Python App".
    - Verify that your applications are listed there.
    - If they are missing, you must recreate them (see "Setup Python App in cPanel" in `START_OVER_GUIDE.md`).
    - If they are present, try restarting them.

2.  **Check `.htaccess` File:**
    - Go to File Manager in cPanel.
    - Navigate to the `public_html` folder (or wherever your main site is).
    - Ensure "Show Hidden Files" is enabled in Settings (top right).
    - Look for an `.htaccess` file. If it was recently modified (check "Last Modified" date), it might have lost the rules for your Python apps.
    - Compare it with a backup if available. Note that cPanel usually manages the Python app routing automatically, but sometimes manual intervention is needed if it breaks.

3.  **Re-save Python App Configuration:**
    - In "Setup Python App", edit an application.
    - Make a trivial change (e.g., add a space to a description if possible, or just click "Save" or "Update" if available).
    - Sometimes simply clicking "Restart" forces cPanel to regenerate the necessary configuration files.

4.  **Contact Hosting Support:**
    - If all else fails, contact VentraIP support. Mention that "All Python (Phusion Passenger) apps on my account are returning 404 errors." They can check the server logs.
