# main.py

import os
from dotenv import load_dotenv
from src.ui import display_app

def run():
    # Loads environment variables and then launches the Streamlit UI.
    load_dotenv()
    display_app()

if __name__ == '__main__':
    run()
