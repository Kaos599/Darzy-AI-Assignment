# src/constants.py

import os
from dotenv import load_dotenv

load_dotenv() 

MAX_IMAGE_SIZE = (1024, 1024)
JPEG_QUALITY = 85
TARGET_FORMAT = "JPEG"
DEFAULT_FONT_SIZE = 15

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash")
