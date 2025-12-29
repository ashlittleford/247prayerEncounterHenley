import sys
import os

# Insert the app directory into sys.path so that the application can be found
sys.path.insert(0, os.path.dirname(__file__))

from app import app as application
