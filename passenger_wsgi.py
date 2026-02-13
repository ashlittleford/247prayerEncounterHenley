import sys
import os

# Insert the app directory into sys.path so that the application can be found
sys.path.insert(0, os.path.dirname(__file__))

# Change CWD to the app directory
os.chdir(os.path.dirname(__file__))

# Set environment variables for Matplotlib
os.environ['MPLCONFIGDIR'] = '/tmp'

from app import app as application
