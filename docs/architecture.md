# Application Architecture

This document provides a high-level overview of the AI Fashion Analysis Tool's project structure and the roles of its key components.

## Project Directory Structure

The project is organized into the following main directories and files at the root level:

```
aifashion/
├── main.py                 # Main entry point to launch the Streamlit application.
├── app.py                  # Contains the main Streamlit application logic and UI.
├── requirements.txt        # Lists Python dependencies for the project.
├── README.md               # Quick start guide, links to this detailed documentation.
├── .env                    # Local environment variables (user-created, e.g., for API keys).
│
├── docs/                   # Contains detailed documentation markdown files (like this one).
│   ├── index.md            # Main page for documentation.
│   ├── setup.md            # Installation and setup guide.
│   ├── usage.md            # How to use the application and run tests.
│   ├── architecture.md     # This file, describing the project structure.
│   ├── features_implemented.md # Detailed list of current features.
│   └── future_enhancements.md  # Ideas for future development.
│
├── src/                    # Source code for the application logic.
│   ├── __init__.py         # Makes 'src' a Python package.
│   ├── constants.py        # Stores global constants used across the application.
│   ├── image_utils.py      # Handles image processing and annotation tasks.
│   ├── ai_services.py      # Manages interaction with the AI (Google Gemini).
│   └── data_exporters.py   # Contains functions for data conversion (e.g., to CSV).
│
└── tests/                  # Contains unit tests for the application.
    ├── __init__.py         # Makes 'tests' a Python package.
    └── test_suite.py       # The main collection of unit tests.
```

## Core Components in `src/`

The `src/` directory houses the core logic of the application, broken down into modules for better organization:

*   **`constants.py`:**
    *   Defines global constants used throughout the application, such as image size limits (`MAX_IMAGE_SIZE`), JPEG quality (`JPEG_QUALITY`), the target image format for processing (`TARGET_FORMAT`), and default font sizes for annotations. This centralizes configuration values.

*   **`image_utils.py`:**
    *   **`process_image()`:** Responsible for all pre-AI image manipulations. This includes validating the uploaded image, resizing it if it's too large (while maintaining aspect ratio), converting it to the `TARGET_FORMAT` (typically JPEG), and applying compression.
    *   **`draw_detections_on_image()`:** Takes a processed image and the AI analysis data (specifically, fashion item detections) to draw bounding boxes and labels directly onto the image, providing visual feedback to the user.

*   **`ai_services.py`:**
    *   **`get_fashion_details_from_image()`:** Constructs a prompt for the Gemini API to detect fashion items, their categories, bounding boxes, and per-item dominant colors. It parses the JSON response and performs validation.
    *   **`get_size_estimation_for_items()`:** Interacts with the AI to estimate garment sizes based on the image and (optionally) detected item context.
    *   **`generate_fashion_copy()`:** Calls the AI to generate textual content like product descriptions, styling tips, and social media captions, using image and (optionally) item context.
    *   **`get_smart_recommendations()`:** Uses detected item details to prompt the AI for complementary items/colors, similar styles, and seasonal advice.
    *   All functions in this module are designed to be UI-agnostic, returning data or None and logging errors internally.

*   **`data_exporters.py`:**
    *   Contains utility functions to convert the AI analysis data into different downloadable formats.
    *   **`convert_fashion_details_to_csv()`:** Transforms the structured fashion item data (potentially nested) into a flat CSV format. It includes a helper `_flatten_colors_for_csv()` for handling per-item color lists.
    *   **`export_to_json_string()`:** Converts Python dictionaries into a pretty-printed JSON string for easy download.

## Main Application File

*   **`app.py`:**
    *   This file contains the main Streamlit application logic and user interface (`main_app` function). It is responsible for:
        *   Rendering the page layout, title, and informational text.
        *   Handling API key input and management (via session state).
        *   Providing the file uploader widget.
        *   Orchestrating the application flow by calling functions from `image_utils.py`, `ai_services.py`, and `data_exporters.py`.
        *   Managing all session state variables to maintain UI consistency.
        *   Displaying errors, warnings, and informational messages.
        *   Integrating new UI sections and logic for Size Estimation, AI Fashion Copywriter, and Smart Recommendations.

## Entry Point

*   **`main.py`:**
    *   Located in the project root, this is the script executed by `streamlit run main.py`.
    *   Its primary responsibilities are to:
        1.  Load environment variables from a `.env` file (using `python-dotenv`) at startup, making API keys available to the application.
        2.  Import and call the `main_app()` function from `app.py` to launch the user interface.

## Testing

*   **`tests/test_suite.py`:**
    *   Contains unit tests written using Python's `unittest` framework.
    *   These tests cover key functions in `image_utils.py`, `data_exporters.py`, and (using mocks) `ai_services.py` to ensure their logic behaves as expected under various conditions.

This modular architecture aims to separate concerns, making the codebase easier to understand, maintain, and extend.
