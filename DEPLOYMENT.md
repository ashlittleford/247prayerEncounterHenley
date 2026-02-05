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
4.  Select your `current.csv` file.
5.  Click **Replace current.csv with uploaded file**.

### Configuration

You can configure "Pray Days", "Email Duplicates", and "Person Merges" in the **Admin** tab of the application.

## Deploying Code Changes

The application runs on a cPanel hosting environment using Phusion Passenger.

### Steps to Deploy

1.  **Commit and Push Changes:**
    Push your changes to the `main` branch of the GitHub repository.

    ```bash
    git add .
    git commit -m "Update application logic"
    git push origin main
    ```

2.  **Update the Server:**
    Log in to the server (via SSH or cPanel Terminal) and pull the latest changes.

    ```bash
    cd /home/encosnpm/prayerstats.encounteradelaide.com.au/prayer_dashboard/Prayer-Analysis
    git pull origin main
    ```

3.  **Restart the Application:**
    Restart the application by touching the `passenger_wsgi.py` file.

    ```bash
    touch passenger_wsgi.py
    ```

    Alternatively, you can restart the application via the cPanel "Setup Python App" interface.

## Troubleshooting

-   **Application Error:** Check the `stderr.log` in the application root directory.
-   **Images Missing:** Ensure the `static/plots` directory is writable by the web server user.
