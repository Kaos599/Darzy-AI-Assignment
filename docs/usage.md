# Application Usage Guide

This guide explains how to run the AI Fashion Analysis Tool, interact with its features, and execute unit tests. Ensure you have completed all steps in the [Setup and Installation Guide (`setup.md`)](setup.md) before proceeding.

## 1. Running the Application

1.  **Activate Your Virtual Environment:**
    If you haven't already, navigate to the project's root directory (`aifashion/`) in your terminal and activate the virtual environment:
    *   On macOS and Linux: `source venv/bin/activate`
    *   On Windows: `venv\Scripts\activate`

2.  **Start the Streamlit Application:**
    From the project root directory (`aifashion/`), run the following command:
    ```bash
    streamlit run main.py
    ```

3.  **Access the Application:**
    Streamlit will typically open the application automatically in your default web browser. If not, your terminal will display a local URL (usually `http://localhost:8501`). Open this URL in your browser to access the tool.

    *If you did not set the `GEMINI_API_KEY` in a `.env` file or as an environment variable, the application will prompt you to enter it in a text field at the top of the page.*

## 2. Interacting with the User Interface

The application interface is designed to be straightforward:

### Step 1: Upload Fashion Image
*   Use the **"1. Upload Fashion Image"** file uploader widget to select an image from your computer. You can drag and drop a file or click to browse.
*   Supported formats are JPG, JPEG, PNG, and WEBP.
*   Once uploaded, the application will automatically process the image (resize, convert to JPEG, compress). A preview of the "Processed Image" will be displayed along with a status message about the processing.

### Step 2: Analyze Fashion Details
*   If the image is processed successfully and your Gemini API key is provided, a button labeled **"2. Analyze Fashion Details with AI ✨"** will become active.
*   Click this button to send the processed image to the Google Gemini API for analysis. This may take a few moments; a spinner animation will indicate that analysis is in progress.

### Interpreting the Results

Once the AI analysis is complete:

*   **Annotated Image:**
    *   An **"Annotated Image with Detected Items"** section will appear, displaying your image with:
        *   **Bounding Boxes:** Colored rectangles drawn around each detected fashion item.
        *   **Labels:** Text labels next to each box showing the item's name and category (e.g., "Denim Jacket (Outerwear)"). The color of the box and label often corresponds to a dominant color of the item.
*   **Detected Item Details:**
    *   Below the annotated image, a section titled **"Detected Item Details"** will list each fashion item found by the AI.
    *   Each item is presented in an expandable section (click the item's name to expand/collapse).
    *   Inside each item's section, you'll find:
        *   **Category:** The general category assigned by the AI.
        *   **Dominant Colors (for this item):** A palette of 1-3 dominant colors specific to that item, shown as color swatches with their HEX codes, common names, and estimated percentages.

### Step 3: Download Full Analysis
*   If the AI analysis was successful and items were detected, a **"3. Download Full Analysis"** section will appear.
*   **Download Details as JSON:** Click this button to download a JSON file containing the complete structured data for all detected items (names, categories, bounding boxes, full per-item color palettes). The filename will be based on your original image's name (e.g., `yourimage_fashion_details.json`).
*   **Download Details as CSV:** Click this button to download a CSV file. This file provides a tabular summary of the detected items, including their names, categories, bounding box coordinates, and flattened details for their dominant colors. The filename will also be based on your original image's name (e.g., `yourimage_fashion_details.csv`).

## 3. Running Unit Tests

To verify that the core components of the application are functioning correctly, you can run the provided unit tests.

1.  **Activate Your Virtual Environment** (if not already active).
2.  **Navigate to the Project Root:** Ensure your terminal is in the `aifashion/` directory.
3.  **Run the Tests:** Execute one of the following commands:
    *   To run the specific test suite file:
        ```bash
        python -m unittest tests.test_suite
        ```
    *   To let unittest discover all tests within the `tests` directory (useful if you add more test files later):
        ```bash
        python -m unittest discover tests
        ```
    The tests will run, and you'll see output indicating the number of tests passed, failed, or with errors. All tests should pass if the environment is set up correctly and no code changes have broken functionality.

---

This covers the main usage scenarios for the AI Fashion Analysis Tool. If you encounter any issues, please refer to the setup guide or check for error messages in the application or your terminal.
