# run.py
import os
import sys

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Now import and run your app
from app.app import main
from dotenv import load_dotenv
load_dotenv() 

if __name__ == "__main__":
    main()