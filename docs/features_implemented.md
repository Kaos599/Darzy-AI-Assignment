# Implemented Features

This document details the features currently implemented in the AI Fashion Analysis Tool.

## 1. Image Upload

*   **Supported File Types:** Users can upload images in `JPG`, `JPEG`, `PNG`, and `WEBP` formats.
*   **Interface:** A simple drag-and-drop or file selection widget is provided via Streamlit's `st.file_uploader`.
*   **Basic Validation:** The uploader filters by allowed extensions. Further validation occurs during processing.

## 2. Image Processing

Once an image is uploaded, it undergoes several automated processing steps before AI analysis or display:

*   **Validation:** The system attempts to open and verify the image using the Pillow library to ensure it's a valid and supported image format. Corrupted files or unsupported types will result in an error message.
*   **Resizing:**
    *   If an image's dimensions (width or height) exceed `1024 pixels` (configurable in `src/constants.py:MAX_IMAGE_SIZE`), it is resized.
    *   Resizing maintains the original aspect ratio to prevent distortion.
    *   The Lanczos resampling algorithm is used for high-quality downscaling.
*   **Format Conversion:**
    *   All processed images are converted to `JPEG` format (configurable in `src/constants.py:TARGET_FORMAT`). This standardization is beneficial for consistency in AI model input and web display.
    *   Images with alpha channels (e.g., RGBA PNGs) are converted to RGB.
*   **Compression:**
    *   JPEG images are saved with a quality setting of `85%` (configurable in `src/constants.py:JPEG_QUALITY`) to balance file size and visual quality.
*   **Feedback:** The user is shown the processed image and a message indicating the processing steps taken (e.g., "Image processed as JPEG (resized to fit 1024x1024px) (quality: 85%).").

## 3. AI-Powered Fashion Item Detection

After processing, users can trigger AI analysis using the Google Gemini Pro Vision model:

*   **Detection Scope:** The AI is prompted to identify all distinct fashion items within the image. This includes clothing, footwear, and significant accessories.
*   **Output per Item:** For each detected item, the AI provides:
    *   **Item Name:** A concise descriptive name (e.g., "Blue Denim Shirt", "Leather Ankle Boots").
    *   **Category:** A general classification (e.g., "Topwear", "Footwear", "Accessory").
    *   **Bounding Box:** Normalized coordinates (`[ymin, xmin, ymax, xmax]`) that define the rectangular region where the item is located in the image. These coordinates are between 0.0 and 1.0.

## 4. Per-Item Dominant Color Analysis

As part of the fashion item detection, the AI also analyzes the dominant colors for *each individual detected item*:

*   **Color Details:** For each item, the AI identifies 1 to 3 of its most dominant colors. For each of these colors, it provides:
    *   **HEX Code:** The color represented as a standard hexadecimal value (e.g., `#FF0000` for red).
    *   **Color Name:** A common, human-readable name for the color (e.g., "Crimson Red", "Forest Green").
    *   **Percentage:** An estimated percentage representing how much of that color is present on that specific item (e.g., "70%").

## 5. Annotated Image Display

*   **Visual Feedback:** After successful AI analysis, the processed image is displayed again, but this time with annotations.
*   **Bounding Boxes:** Rectangular boxes are drawn around each detected fashion item based on the coordinates provided by the AI.
*   **Labels:** Each bounding box is accompanied by a text label displaying the item's name and category (e.g., "T-shirt (Topwear)").
*   **Color Coding:** The color of the bounding box and its label background often corresponds to one of the dominant colors of the item itself, providing a quick visual link.

## 6. Detailed Results Display

*   **Structured Information:** Below the annotated image, details for each detected fashion item are presented in a structured and expandable format using Streamlit's `st.expander`.
*   **Per-Item View:** Each expander section is dedicated to one item and shows:
    *   The item's name and category.
    *   A visual representation of its dominant colors using `st.color_picker` swatches (disabled, for display only), along with their names and percentages.

## 7. Data Export

Users can download the detailed analysis results:

*   **JSON Export:**
    *   Provides the complete, structured data returned by the AI for all detected fashion items.
    *   This includes item names, categories, bounding boxes, and the full list of dominant colors (with hex, name, percentage) for each item.
    *   The file is named based on the original uploaded image filename (e.g., `myimage_fashion_details.json`).
*   **CSV Export:**
    *   Provides a flattened, tabular representation of the fashion item details.
    *   Each row corresponds to a detected item.
    *   Columns include: `item_name`, `category`, bounding box coordinates (`bbox_ymin`, `bbox_xmin`, `bbox_ymax`, `bbox_xmax`), and details for up to 3 dominant colors per item (e.g., `color_1_name`, `color_1_hex`, `color_1_percentage`, `color_2_name`, etc.).
    *   The file is named based on the original uploaded image filename (e.g., `myimage_fashion_details.csv`).

## 8. AI Size Estimation

*   **Trigger:** Activated by the "📏 Estimate Sizes" button after an image is processed.
*   **Contextual Analysis:** The AI can optionally use previously detected fashion items from the "Analyze Fashion Details" step as context to focus its estimations.
*   **Output per Garment:** For suitable garments identified in the image, the AI provides:
    *   **Item Description:** A textual description of the garment the estimation refers to (e.g., "the floral print dress," "men's dark wash jeans"). This helps associate the estimate with an item, especially if multiple garments are present.
    *   **Estimated Size:** A suggested size, which can be categorical (e.g., "S", "M", "L"), a numerical range (e.g., "US 8-10"), or descriptive (e.g., "Relaxed Fit Medium," "Appears Oversized L/XL").
    *   **Reasoning:** A brief explanation for the size recommendation, based on visual cues like drape, apparent fit, or proportions.
*   **Display:** Results are shown in a dedicated "Size Estimations" section, with each estimation presented in an expandable view.

## 9. AI Fashion Copywriter

*   **Trigger:** Activated by the "✍️ Generate Fashion Text" button after an image is processed.
*   **Contextual Generation:** Similar to size estimation, this feature can use previously detected fashion items (names, categories, colors) to generate more relevant and focused text.
*   **Generated Content Types:** The AI produces three distinct pieces of text:
    *   **Product Description:** An engaging and descriptive text (2-4 sentences) suitable for e-commerce listings, highlighting key features and style.
    *   **Styling Suggestions:** 2-3 brief tips or outfit recommendations (1-2 sentences each) on how to wear or pair the detected item(s).
    *   **Social Media Caption:** A short, catchy caption (1-2 sentences, max ~280 characters) suitable for platforms like Instagram or Twitter, often including relevant hashtags.
*   **Display:** The generated texts are displayed in a dedicated "AI Generated Fashion Text" section, typically using markdown for readability.

## 10. Smart Recommendations

*   **Trigger:** Activated by the "💡 Get Recommendations" button. This button is enabled only *after* the "Analyze Fashion Details" step has successfully identified fashion items, as it uses these items as input.
*   **Input:** Takes the list of detected fashion items (including their names, categories, and dominant colors) from the primary analysis.
*   **Recommendation Types:** The AI generates three types of recommendations:
    *   **Complementary Suggestions:** Ideas for other specific types of fashion items or colors that would pair well with the detected items or the overall look, along with brief reasoning.
    *   **Similar Styles:** Recommendations for other specific styles or types of items that the user might also like, based on the style of the items already detected.
    *   **Seasonal Styling Advice:** Brief tips on how one or more of the detected items could be styled for a particular season (Spring, Summer, Autumn, Winter).
*   **Display:** The recommendations are presented in a dedicated "Smart Recommendations" section, with each type of advice clearly labeled.
