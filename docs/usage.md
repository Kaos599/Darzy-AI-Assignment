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

### Step 2: Choose AI Analysis Feature(s)
After the image is processed and if your API key is set, you'll see several buttons to trigger different AI analyses:

*   **Analyze Fashion Details (👁️ Analyze Details):**
    *   Click this button to perform the primary analysis: detecting items, their categories, colors, and drawing bounding boxes.
    *   This is often the first AI step to run as other features can use its output as context.
    *   Results include the **Annotated Image** and **Detected Item Details**.

*   **Estimate Garment Sizes (📏 Estimate Sizes):**
    *   Click this button to get size suggestions for garments in the image.
    *   If "Analyze Fashion Details" was run first, its findings (e.g., item names) can provide context to this feature.
    *   Results appear in the **"📏 Size Estimations"** section, showing item descriptions, estimated sizes, and reasoning.

*   **Generate Fashion Text (✍️ Generate Text):**
    *   Click this button to create AI-generated textual content.
    *   Like size estimation, it can use context from "Analyze Fashion Details" if available.
    *   Results appear in the **"✍️ AI Generated Fashion Text"** section, including a Product Description, Styling Suggestions, and a Social Media Caption.

*   **Get Smart Recommendations (💡 Get Recommendations):**
    *   This button is **enabled only after "Analyze Fashion Details" has successfully identified items.**
    *   Click it to receive AI-powered advice based on the detected items.
    *   Results appear in the **"💡 Smart Recommendations"** section, featuring Complementary Suggestions, Similar Styles, and Seasonal Styling Advice.

### Interpreting the Results

Once an AI analysis is complete, the relevant results will be displayed:

*   **Annotated Image:**
    *   Appears after running "Analyze Fashion Details".
    *   Shows your image with colored rectangles (bounding boxes) around detected fashion items and text labels (item name, category).
*   **Detected Item Details:**
    *   Appears after running "Analyze Fashion Details".
    *   Lists each detected item in an expandable section, showing its category and a palette of its dominant colors (with swatches, HEX codes, names, percentages).
*   **Size Estimations:**
    *   Appears after running "Estimate Sizes".
    *   Each estimation describes the item, suggests a size, and provides the AI's reasoning.
*   **AI Generated Fashion Text:**
    *   Appears after running "Generate Fashion Text".
    *   Displays the AI-crafted product description, styling suggestions, and social media caption.
*   **Smart Recommendations:**
    *   Appears after running "Get Recommendations".
    *   Shows suggestions for complementary items/colors, similar styles, and seasonal advice based on the initially detected items.

### Step 3: Download Full Analysis (Fashion Details)
*   If the "Analyze Fashion Details" step was successful and items were detected, a **"Download Fashion Details"** section will appear under the item details.
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
