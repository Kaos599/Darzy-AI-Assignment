# AI Fashion Analysis Tool

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red)](https://streamlit.io/)

## USE LINK:- https://darzy-ai-assignment.streamlit.app/

## Project Overview

This application provides an AI-powered solution for analyzing fashion images. Users can upload an image, and the tool leverages advanced AI models to identify individual fashion items, categorize them, determine their dominant colors, and mark their locations with bounding boxes. Beyond core detection, it offers features like size estimation, AI-driven marketing copy generation, and smart styling recommendations. The comprehensive analysis results can be easily exported in both JSON and CSV formats.

Our goal is to assist fashion professionals, e-commerce businesses, and enthusiasts in gaining deeper insights from visual fashion data.

## Features

*   **Image Upload & Processing:** Supports common image formats (JPG, JPEG, PNG, WEBP). Images are automatically resized, converted to a consistent format (JPEG), and compressed for optimal AI processing while maintaining quality. Robust error handling is in place for invalid files.
*   **AI-Powered Fashion Item Detection:** Utilizes Google Gemini Pro Vision to detect multiple distinct fashion items, providing their names, categories (e.g., "Outerwear", "Topwear"), bounding box coordinates, and a detailed palette of dominant colors (HEX, name, percentage) for each item.
*   **Visual & Detailed Results:** Displays the processed image with AI-generated bounding boxes and labels. Comprehensive details for each detected item, including color palettes, are presented in an intuitive, expandable interface.
*   **Data Export:** Allows downloading of complete, structured analysis results as a **JSON** file or a summarized, flattened view as a **CSV** file.
*   **AI Size Estimation:** Provides potential size suggestions for garments based on visual cues and AI reasoning.
*   **AI Fashion Copywriter:** Generates engaging product descriptions, practical styling tips, and catchy social media captions.
*   **Smart Recommendations:** Offers AI-driven suggestions for complementary items/colors, similar styles, and seasonal advice, leveraging detected fashion items.

## Project Structure

The project is organized into logical directories for clarity and maintainability:

```
├── main.py                 # Main entry point to run the Streamlit app
├── app.py                  # Core Streamlit application logic and UI components
├── requirements.txt        # Python dependencies
├── README.md               # Project overview and setup instructions (this file)
├── .env                    # Environment variables (e.g., API keys, gitignored)
├── assets/                 # Static assets like example images
├── docs/                   # Detailed project documentation (architecture, features, usage)
│   ├── index.md
│   ├── setup.md
│   ├── usage.md
│   ├── architecture.md
│   ├── features_implemented.md
│   └── future_enhancements.md
├── src/                    # Source code for core application logic
│   ├── __init__.py
│   ├── ui.py               # Streamlit UI components and layout
│   ├── image_utils.py      # Image processing functions
│   ├── ai_services.py      # AI model interaction logic
│   ├── data_exporters.py   # CSV/JSON conversion utilities
│   ├── constants.py        # Global application constants
│   └── database_manager.py # Handles MongoDB interactions for persistence
└── tests/                  # Unit tests for various modules
    ├── __init__.py
    └── test_suite.py
```

## Technologies Used

*   **Programming Language:** Python 3.8+
*   **Web Framework/UI:** [Streamlit](https://streamlit.io/)
*   **AI Model Interaction:** [Google Gemini Pro Vision API](https://python.langchain.com/docs/integrations/llms/google_generative_ai) (via `langchain-google-genai`)
*   **Image Processing:** [Pillow (PIL)](https://python-pillow.org/)
*   **Database:** [MongoDB](https://www.mongodb.com/) (via `pymongo`)
*   **Testing:** `unittest` (Python's built-in testing framework)
*   **Environment Management:** `python-dotenv`

## Setup Instructions

### 1. Prerequisites

*   Python 3.8 or newer.
*   `pip` (Python package installer).
*   `git` (optional, for cloning the repository).

### 2. Clone the Repository (Optional)

If you have git, clone the repository:
```bash
git clone <repository_url> # Replace with your repository URL
cd AI-Fashion-Analysis-Tool # Navigate into the project root directory
```
If you don't use git, download the source code as a ZIP and extract it. Ensure the project structure matches the "Project Structure" section above.

### 3. Create a Virtual Environment (Recommended)

From the project root directory:
```bash
python -m venv venv
# On macOS/Linux:
source venv/bin/activate
# On Windows:
.\venv\Scripts\activate
```

### 4. Install Dependencies

With your virtual environment activated:
```bash
pip install -r requirements.txt
```

### 5. Set Environment Variables

You need a Google Gemini API key and MongoDB connection details for AI analysis and data persistence.

*   **Create a `.env` file:** In the project's root directory, create a file named `.env`.
*   **Add your API key and MongoDB URI:**
    ```env
    GEMINI_API_KEY="YOUR_ACTUAL_GEMINI_API_KEY_HERE"
    # Optional: Specify a Gemini Vision model name if you don't want to use the default
    # GEMINI_MODEL_NAME_VISION="gemini-pro-vision"

    MONGO_URI="mongodb://localhost:27017/" # Or your MongoDB connection string
    MONGO_DB_NAME="fashion_analysis_db"    # Your preferred database name
    ```
The application will load these variables at startup. If `GEMINI_API_KEY` is not found, the app will prompt for it.

## Running the Application

Ensure your virtual environment is activated and you are in the project root directory.
Run the Streamlit application:

```bash
streamlit run main.py
```
This will open the application in your default web browser (e.g., `http://localhost:8501`).

## Running Unit Tests

From the project root directory with your virtual environment activated:

```bash
python -m unittest tests.test_suite
# Alternatively, to discover all tests in the tests directory:
# python -m unittest discover tests
```

