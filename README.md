# AI Fashion Analysis Tool

## Project Overview

This application leverages AI to analyze images of fashion items. Users can upload an image, and the tool will detect individual fashion items, identify their categories, determine their dominant colors, and draw bounding boxes around them. The analysis results, including color palettes for each item, can be exported in JSON or CSV format.

This project is structured into multiple Python files for better organization and maintainability.

## Features

*   **Image Upload:** Supports uploading fashion images (JPG, JPEG, PNG, WEBP).
*   **Image Processing:**
    *   Automatic resizing for very large images while maintaining aspect ratio.
    *   Conversion to a standard format (JPEG) optimized for AI model input.
    *   Compression to reduce file size.
    *   Error handling for invalid or corrupted image files.
*   **AI-Powered Fashion Item Detection (via Google Gemini Pro Vision):**
    *   Detects multiple distinct fashion items in the uploaded image.
    *   For each detected item, it provides:
        *   **Item Name:** A descriptive name (e.g., "Blue Denim Jacket").
        *   **Category:** General type (e.g., "Outerwear", "Topwear", "Accessory").
        *   **Bounding Box:** Coordinates defining the item's location in the image.
        *   **Dominant Colors:** A specific color palette (HEX codes, color names, percentages) for that individual item.
*   **Visual Feedback:** Displays the processed image with bounding boxes and labels overlaid on detected items.
*   **Detailed Results Display:** Clearly lists each detected item and its properties, including its specific color palette, using an expandable interface.
*   **Data Export:**
    *   Download the complete, structured analysis results for all detected items as a **JSON** file.
    *   Download a summary of detected items and their key features (including flattened color data) as a **CSV** file.

## Project Structure

The project is organized as follows:

aifashion/
├── main.py                 # Main entry point to run the Streamlit app
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── .env                    # For API keys (user-created, gitignored)
│
├── src/                    # Source code for the application
│   ├── __init__.py
│   ├── ui.py               # Streamlit UI components and layout
│   ├── image_utils.py      # Image processing functions
│   ├── ai_services.py      # AI model interaction logic
│   ├── data_exporters.py   # CSV/JSON conversion utilities
│   └── constants.py        # Global application constants
│
└── tests/                  # Unit tests
    ├── __init__.py
    └── test_suite.py       # Main test suite

## Technologies Used

*   **Programming Language:** Python 3.8+
*   **Web Framework/UI:** Streamlit
*   **AI Model Interaction:** Google Gemini Pro Vision API (via `langchain-google-genai`)
*   **Image Processing:** Pillow (PIL)
*   **Testing:** `unittest` (Python's built-in testing framework)
*   **Environment Management:** `python-dotenv`

## Setup Instructions

### 1. Prerequisites

*   Python 3.8 or newer.
*   `pip` (Python package installer).
*   `git` (for cloning, optional if you download the code directly).

### 2. Clone the Repository (Optional)

If you have git, you can clone the repository:
```bash
git clone <repository_url>
cd aifashion  # Navigate into the project root
```
Otherwise, download and extract the source code files into a directory named `aifashion`. Ensure all files and the `src` and `tests` subdirectories are correctly placed.

### 3. Create a Virtual Environment (Recommended)

From the project root directory (`aifashion/`):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 4. Install Dependencies

Install the required Python packages using `requirements.txt` located in the project root:
```bash
pip install -r requirements.txt
```

### 5. Set Environment Variables

You need a Google Gemini API key for the AI analysis features.

*   **Create a `.env` file:** In the project's root directory (`aifashion/`), create a file named `.env`.
*   **Add your API key:** Inside the `.env` file, add your Gemini API key like this:
    ```
    GEMINI_API_KEY="YOUR_ACTUAL_GEMINI_API_KEY_HERE"
    ```
    You can also optionally set `GEMINI_MODEL_NAME="your-preferred-gemini-vision-model"` if you want to override the default.
The application (`main.py`) uses `python-dotenv` to load these variables. If `GEMINI_API_KEY` is not found, the application will prompt for it.

## Running the Application

Ensure your virtual environment is activated and you are in the project root directory (`aifashion/`).
Run the Streamlit application using:

```bash
streamlit run main.py
```

This will typically open the application in your default web browser. If not, it will display a local URL (e.g., `http://localhost:8501`) that you can open manually.

## Running Unit Tests

From the project root directory (`aifashion/`), ensure your virtual environment is activated.
Run the unit tests using:

```bash
python -m unittest tests.test_suite
```
Alternatively, to discover all tests within the `tests` directory:
```bash
python -m unittest discover tests
```

## Interpreting Fashion Detection Results

When you upload an image and run the analysis:

*   **Annotated Image:** You'll see your uploaded image with colored boxes drawn around detected fashion items. Each box will have a label showing the item's name and category.
*   **Detected Item Details:** Below the image, each detected item will be listed in an expandable section. You'll find its name, category, and specific dominant color palette.
*   **Data Export:** You can download the analysis in JSON or CSV format.
