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
