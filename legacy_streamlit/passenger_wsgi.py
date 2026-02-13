import sys
import os
import subprocess

# 1. --- Configuration ---
# Use the directory of this file as the application directory
APP_DIR = os.path.dirname(os.path.abspath(__file__))
APP_FILE = 'dashboard_app.py' # <-- Ensure this matches your Streamlit script name!

# 2. --- Setup Environment ---
# Add the application directory to the Python path
sys.path.insert(0, APP_DIR)

# Change the current working directory to the app directory
os.chdir(APP_DIR)

# 3. --- WSGI Application Function ---
def application(environ, start_response):
    # Get the port assigned by Phusion Passenger/CPanel
    passenger_port = environ.get('SERVER_PORT', '8501')

    # This is the actual Streamlit command array
    cmd = [
        # Use the Python executable from the virtual environment
        sys.executable,
        '-m', 'streamlit',
        'run',
        APP_FILE,
        '--server.port', passenger_port,
        '--server.address', '0.0.0.0'
    ]

    # Use subprocess.Popen to launch Streamlit in the background
    try:
        # Check if the process is already running to prevent duplicates
        tmp_dir = os.environ.get('TMPDIR', '/tmp')
        flag_file = os.path.join(tmp_dir, 'streamlit_running')

        if not os.path.exists(flag_file):
            # Launch the command
            subprocess.Popen(cmd, close_fds=True)

            # Create a simple flag file to indicate the process was launched
            with open(flag_file, 'w') as f:
                f.write('1')

        status = '200 OK'
        response_headers = [('Content-type', 'text/html')]
        start_response(status, response_headers)

        # We return a simple message, hoping Passenger proxies to the Streamlit port
        return [b"<h1>Streamlit is starting up. Please refresh in a moment.</h1>"]

    except Exception as e:
        # If the launch fails, display the error in the browser
        status = '500 Internal Server Error'
        response_headers = [('Content-type', 'text/plain')]
        start_response(status, response_headers)
        return [f"Streamlit launcher failed: {e}".encode('utf-8')]
