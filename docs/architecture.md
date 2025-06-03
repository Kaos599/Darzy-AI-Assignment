# Application Architecture

This document provides a high-level overview of the AI Fashion Analysis Tool's project structure and the roles of its key components.

## Project Directory Structure

The project is organized into the following main directories and files at the root level:

```
aifashion/
├── main.py                 # Main entry point to launch the Streamlit application.
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
│   ├── data_exporters.py   # Contains functions for data conversion (e.g., to CSV).
│   └── ui.py               # Defines the Streamlit user interface and application flow.
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
    *   **`get_fashion_details_from_image()`:** This is the sole interface to the Google Gemini Pro Vision model. It constructs the detailed prompt asking for fashion item detection, categorization, bounding boxes, and per-item dominant colors. It then sends the image and prompt to the AI, parses the JSON response, and performs initial validation on the received data structure. This module is designed to be UI-agnostic.

*   **`data_exporters.py`:**
    *   Contains utility functions to convert the AI analysis data into different downloadable formats.
    *   **`convert_fashion_details_to_csv()`:** Transforms the structured fashion item data (potentially nested) into a flat CSV format. It includes a helper `_flatten_colors_for_csv()` for handling per-item color lists.
    *   **`export_to_json_string()`:** Converts Python dictionaries into a pretty-printed JSON string for easy download.
    *   (Includes a deprecated `convert_palette_to_csv()` for a simpler, older data format).

*   **`ui.py`:**
    *   **`display_app()`:** This is the heart of the user interface. It uses Streamlit components to:
        *   Render the page layout, title, and informational text.
        *   Handle API key input and management (via session state).
        *   Provide the file uploader widget.
        *   Orchestrate the application flow:
            1.  Calls `process_image()` from `image_utils.py` after upload.
            2.  Displays the processed image.
            3.  Provides a button to trigger AI analysis, which then calls `get_fashion_details_from_image()` from `ai_services.py`.
            4.  If analysis is successful, calls `draw_detections_on_image()` from `image_utils.py` to get the annotated image.
            5.  Displays the annotated image and the structured item details (using expanders, color pickers).
            6.  Provides download buttons that use functions from `data_exporters.py`.
        *   Manages all session state variables to maintain UI consistency across user interactions.
        *   Displays errors, warnings, and informational messages to the user.

## Entry Point

*   **`main.py`:**
    *   Located in the project root, this is the script executed by `streamlit run main.py`.
    *   Its primary responsibilities are to:
        1.  Load environment variables from a `.env` file (using `python-dotenv`) at startup, making API keys available to the application.
        2.  Import and call the `display_app()` function from `src.ui` to launch the user interface.

## Testing

*   **`tests/test_suite.py`:**
    *   Contains unit tests written using Python's `unittest` framework.
    *   These tests cover key functions in `image_utils.py`, `data_exporters.py`, and (using mocks) `ai_services.py` to ensure their logic behaves as expected under various conditions.

This modular architecture aims to separate concerns, making the codebase easier to understand, maintain, and extend.
