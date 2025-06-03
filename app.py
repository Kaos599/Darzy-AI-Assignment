# app.py
# Final version with all features, refinements, and documentation.

import streamlit as st
from PIL import Image, UnidentifiedImageError, ImageDraw, ImageFont
import io
import os
import json
import base64
import csv
import traceback # For detailed error logging

# Attempt to import Langchain and Google Generative AI
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.messages import HumanMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    # Error will be handled in the main_app UI if SDK is missing

# --- Global Constants ---
MAX_IMAGE_SIZE = (1024, 1024) # Max dimensions (width, height) for processed images
JPEG_QUALITY = 85             # Quality for JPEG compression (0-100)
TARGET_FORMAT = "JPEG"        # Standard image format for processing and AI input
DEFAULT_FONT_SIZE = 15        # Default font size for drawing labels on images

# --- Image Processing ---
def process_image(image_bytes: bytes, filename: str) -> tuple[bytes | None, str | None, str]:
    """
    Processes an uploaded image: validates, resizes if too large, converts to TARGET_FORMAT,
    and compresses.

    Args:
        image_bytes: Raw bytes of the uploaded image.
        filename: Original filename of the uploaded image.

    Returns:
        A tuple containing:
        - Processed image bytes (or None if processing fails).
        - New filename for the processed image (or None if processing fails).
        - A message indicating the outcome or error.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.load() # Check for truncated images

        original_mode = img.mode
        # Convert to RGB if TARGET_FORMAT is JPEG and mode is not suitable (e.g., RGBA, P, LA)
        # Grayscale ('L') is generally acceptable for JPEG.
        if TARGET_FORMAT == 'JPEG' and original_mode not in ['RGB', 'L']:
            if original_mode in ['RGBA', 'P', 'LA']: # Common modes with alpha or palette
                img = img.convert('RGB')
            elif original_mode != 'RGB': # Catch-all for other modes, ensure it's not already RGB
                 img = img.convert('RGB')

        resized = False
        if img.size[0] > MAX_IMAGE_SIZE[0] or img.size[1] > MAX_IMAGE_SIZE[1]:
            img.thumbnail(MAX_IMAGE_SIZE, Image.Resampling.LANCZOS) # LANCZOS for high-quality downscale
            resized = True

        processed_image_io = io.BytesIO()
        # Construct new filename, keeping original base name
        name_parts = filename.rsplit('.', 1) # Split only on the last dot
        base_name = name_parts[0] if len(name_parts) > 1 else filename
        new_filename = f"{base_name}_processed.{TARGET_FORMAT.lower()}"

        save_params = {}
        if TARGET_FORMAT == "JPEG":
            save_params['format'] = "JPEG"
            save_params['quality'] = JPEG_QUALITY
        elif TARGET_FORMAT == "PNG": # Kept for flexibility, though TARGET_FORMAT is JPEG by default
            save_params['format'] = "PNG"
            save_params['optimize'] = True # Optimize PNGs if used
        else:
            # This case should not be hit if TARGET_FORMAT is one of the above
            return None, None, f"Unsupported TARGET_FORMAT: {TARGET_FORMAT}"

        img.save(processed_image_io, **save_params)
        processed_image_bytes = processed_image_io.getvalue()

        # Construct informative message for the user
        message = f"Image processed as {TARGET_FORMAT}"
        if resized: message += f" (resized to fit {MAX_IMAGE_SIZE[0]}x{MAX_IMAGE_SIZE[1]}px)"
        if TARGET_FORMAT == "JPEG": message += f" (quality: {JPEG_QUALITY}%)."
        else: message += "."
        return processed_image_bytes, new_filename, message

    except UnidentifiedImageError: # Pillow couldn't identify the image file
        return None, None, "Processing failed: The file is not a valid image or its format is not supported."
    except IOError as e: # General I/O error, e.g., file corrupted
        return None, None, f"Processing failed: Could not read or process image file. It might be corrupted. Error: {e}"
    except Exception as e: # Catch any other unexpected errors during processing
        return None, None, f"Processing failed: An unexpected error occurred during image processing: {e}"

# --- AI Analysis ---
def get_fashion_details_from_image(image_bytes: bytes, api_key: str, model_name: str) -> dict | None:
    """
    Analyzes an image using the Gemini Pro Vision model to detect fashion items,
    their categories, bounding boxes, and dominant colors per item.

    The prompt requests the AI to return a JSON object with a 'fashion_items' list.
    Each item in the list is expected to have:
    - "item_name": str
    - "category": str
    - "bounding_box": list[float] (normalized [ymin, xmin, ymax, xmax])
    - "dominant_colors": list[dict] (each dict with "hex_code", "color_name", "percentage")

    Args:
        image_bytes: Bytes of the processed image to be analyzed.
        api_key: The Google Gemini API key.
        model_name: The name of the Gemini model to use (e.g., "gemini-pro-vision").

    Returns:
        A dictionary containing the analysis results (a list under 'fashion_items' key)
        or None if the analysis fails or the response is invalid.
        Each fashion item includes 'item_name', 'category', 'bounding_box',
        and 'dominant_colors' (a list of color dictionaries).
    """
    if not LANGCHAIN_AVAILABLE:
        st.error("Required AI libraries (langchain-google-genai) are not installed. AI features disabled.")
        return None
    if not api_key:
        st.error("GEMINI_API_KEY is not set. Cannot perform AI analysis.")
        return None

    try:
        # Initialize the ChatGoogleGenerativeAI model
        llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key, temperature=0.25) # Slightly higher temp for descriptive diversity

        # Encode image to base64 for API request
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        image_mime_type = f"image/{TARGET_FORMAT.lower()}"

        # Detailed prompt for the AI model, specifying JSON output format
        prompt_text = f"""Analyze the provided image to identify all distinct fashion items.
For EACH detected fashion item, provide the following details:
1.  "item_name": A concise name for the item (e.g., "T-shirt", "Blue Jeans", "Leather Handbag", "Sneakers").
2.  "category": A general category for the item (e.g., "Topwear", "Bottomwear", "Footwear", "Accessory", "Outerwear", "Full Body").
3.  "bounding_box": Normalized bounding box coordinates [ymin, xmin, ymax, xmax], where values are between 0.0 and 1.0. Ensure ymin < ymax and xmin < xmax.
4.  "dominant_colors": A list of 1 to 3 dominant colors for THIS specific item. Each color should be an object with:
    *   "hex_code": The HEX code (e.g., "#RRGGBB").
    *   "color_name": A common name for the color (e.g., "Navy Blue").
    *   "percentage": Estimated percentage of this color ON THIS ITEM (e.g., "70%").

Return ONLY the JSON object as your response, starting with {{ and ending with }}.
The JSON object must have a single key "fashion_items", which is a list of objects.
Each object in the list represents a detected fashion item and must contain all keys: "item_name", "category", "bounding_box", and "dominant_colors".

Example of the expected JSON structure:
{{
  "fashion_items": [
    {{
      "item_name": "Red Cotton T-shirt",
      "category": "Topwear",
      "bounding_box": [0.1, 0.2, 0.5, 0.8],
      "dominant_colors": [
        {{ "hex_code": "#FF0000", "color_name": "Red", "percentage": "90%" }},
        {{ "hex_code": "#FFFFFF", "color_name": "White", "percentage": "10%" }}
      ]
    }}
  ]
}}
If no fashion items are detected, return an empty list for "fashion_items": {{"fashion_items": []}}.
Focus only on actual clothing, shoes, and significant accessories. Ignore background elements.
Be precise with bounding boxes.
"""
        # Create the message payload for the LLM
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": f"data:{image_mime_type};base64,{base64_image}"},
            ]
        )

        # Invoke the LLM with a spinner for user feedback
        with st.spinner("👗 Analyzing fashion items with Gemini... This may take a few moments."):
            response = llm.invoke([message])

        response_content = response.content.strip()

        # Robust JSON extraction from potential markdown code blocks or raw text
        json_str = ""
        if response_content.startswith("```json") and response_content.endswith("```"):
            json_str = response_content[7:-3].strip()
        elif response_content.startswith("```") and response_content.endswith("```"): # Handle if just ``` not ```json
            json_str = response_content[3:-3].strip()
        elif response_content.startswith("{") and response_content.endswith("}"): # Standard JSON
            json_str = response_content
        else: # Fallback: try to find JSON anywhere in the string
            match = json.re.search(r'\{[\s\S]*\}', response_content) # [\s\S] matches any char including newline
            if match:
                json_str = match.group(0)
                st.warning("AI response was not clean JSON. Attempted to extract JSON. Results might be imperfect.")
            else:
                st.error(f"Could not find any JSON block in the AI response. Raw response snippet: {response_content[:500]}")
                return None # Critical failure to find JSON

        try:
            analysis_data = json.loads(json_str)
        except json.JSONDecodeError as e:
            st.error(f"Failed to parse JSON from AI response. Error: {e}. Content processed: {json_str[:500]}")
            return None # JSON parsing failed

        # Validate the structure of the parsed JSON data
        if not isinstance(analysis_data, dict) or \
           "fashion_items" not in analysis_data or \
           not isinstance(analysis_data["fashion_items"], list):
            st.error("AI response is valid JSON but not in the expected structure (e.g., missing 'fashion_items' list).")
            st.json(analysis_data) # Show what was received for debugging
            return None # Invalid structure

        # Validate each item within the 'fashion_items' list for required keys and correct types
        validated_items = []
        for item_idx, item in enumerate(analysis_data.get("fashion_items",[])): # Use .get for safety
            # Check for essential keys in each item
            if not isinstance(item, dict) or not all(k in item for k in ["item_name", "category", "bounding_box", "dominant_colors"]):
                st.warning(f"Skipping a detected item (index {item_idx}) due to missing required fields: {item.get('item_name', 'Unknown Item')}")
                continue # Skip this item

            # Validate bounding_box structure and values
            bbox = item["bounding_box"]
            if not (isinstance(bbox, list) and len(bbox) == 4 and \
                    all(isinstance(n, (float, int)) and 0.0 <= n <= 1.0 for n in bbox) and \
                    (bbox[0] < bbox[2] and bbox[1] < bbox[3])): # ymin < ymax and xmin < xmax
                st.warning(f"Item '{item['item_name']}' has an invalid or malformed bounding_box (must be list of 4 floats [0-1], ymin<ymax, xmin<xmax). Skipping item.")
                continue # Skip this item

            # Validate dominant_colors structure
            if not isinstance(item["dominant_colors"], list):
                st.warning(f"Item '{item['item_name']}' dominant_colors is not a list. Skipping item.")
                continue # Skip this item

            # Validate individual color details within dominant_colors
            valid_colors_for_item = []
            for color_detail_idx, color_detail in enumerate(item.get("dominant_colors",[])):
                 if isinstance(color_detail, dict) and all(k_color in color_detail for k_color in ["hex_code", "color_name", "percentage"]):
                      valid_colors_for_item.append(color_detail)
                 else:
                      st.warning(f"Skipping invalid color detail (index {color_detail_idx}) in item '{item['item_name']}'.")
            item["dominant_colors"] = valid_colors_for_item # Update item with only its valid colors

            validated_items.append(item) # Add the validated item to our list

        return {"fashion_items": validated_items} # Return data with only validated items

    except Exception as e: # Catch-all for any other unexpected errors
        st.error(f"An unexpected error occurred during AI fashion detail analysis: {e}")
        st.error(traceback.format_exc()) # Log detailed traceback for server-side debugging
        return None

# --- Image Annotation ---
def draw_detections_on_image(image_bytes: bytes, fashion_items_data: dict) -> bytes | None:
    """
    Draws bounding boxes and labels on an image based on fashion item detections.

    Args:
        image_bytes: Bytes of the processed image (background for annotations).
        fashion_items_data: Dictionary containing a list of 'fashion_items'. Each item
                            should have 'bounding_box', 'item_name', 'category',
                            and optionally 'dominant_colors' for box coloring.

    Returns:
        Bytes of the annotated image in PNG format, or None if an error occurs.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB") # Ensure image is in RGB for drawing
        draw = ImageDraw.Draw(img)

        # Attempt to load a common font (e.g., Arial), fallback to Pillow's default if not found
        try:
            font = ImageFont.truetype("arial.ttf", DEFAULT_FONT_SIZE)
        except IOError: # Catches error if font file is not found
            font = ImageFont.load_default() # Default font is usually small and basic

        img_width, img_height = img.size

        for item_idx, item in enumerate(fashion_items_data.get("fashion_items", [])):
            bbox = item.get("bounding_box")
            # Ensure bounding box data is valid before proceeding
            if not (isinstance(bbox, list) and len(bbox) == 4 and all(isinstance(n, (float, int)) for n in bbox)):
                st.warning(f"Skipping drawing for item '{item.get('item_name')}' due to invalid bounding box.")
                continue

            ymin, xmin, ymax, xmax = bbox

            # Convert normalized (0-1) coordinates to absolute pixel values
            left = xmin * img_width
            right = xmax * img_width
            top = ymin * img_height
            bottom = ymax * img_height

            # Determine box color: Use first dominant color of the item, or cycle through defaults
            item_colors = item.get("dominant_colors", [])
            box_color_hex = "#FF0000" # Default to Red
            if item_colors and isinstance(item_colors[0], dict) and "hex_code" in item_colors[0]:
                potential_hex = item_colors[0]["hex_code"]
                # Basic validation for hex code format before using it
                if isinstance(potential_hex, str) and potential_hex.startswith('#') and len(potential_hex) in [4, 7]: # Handles #RGB and #RRGGBB
                    box_color_hex = potential_hex
            else: # Fallback if no valid dominant color is found for the item
                default_box_colors = ["#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", "#00FFFF"]
                box_color_hex = default_box_colors[item_idx % len(default_box_colors)]


            draw.rectangle([(left, top), (right, bottom)], outline=box_color_hex, width=3)

            label = f"{item.get('item_name', 'Unknown')} ({item.get('category', 'N/A')})"

            # Calculate text size to draw a background rectangle for better readability
            try: # Pillow versions >= 8.0.0 provide textbbox
                text_bbox = draw.textbbox((0,0), label, font=font) # Use (0,0) for size calculation
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
            except AttributeError: # Fallback for older Pillow versions using textsize
                text_width, text_height = draw.textsize(label, font=font)

            # Position text label: typically above the bounding box
            text_anchor_y = top - text_height - 4
            # Adjust if text goes off the top edge of the image
            if text_anchor_y < 2: # Give a small margin from the top
                text_anchor_y = top + 2 # Position inside the box, near the top

            # Draw a filled rectangle as background for the text
            draw.rectangle(
                [(left, text_anchor_y), (left + text_width + 4, text_anchor_y + text_height + 2)],
                fill=box_color_hex
            )
            # Draw the text
            draw.text((left + 2, text_anchor_y), label, fill="white", font=font)

        annotated_image_io = io.BytesIO()
        img.save(annotated_image_io, format="PNG") # Save annotated image as PNG to preserve quality
        return annotated_image_io.getvalue()
    except Exception as e: # Catch any errors during drawing
        st.error(f"Error drawing detections on image: {e}")
        # st.error(traceback.format_exc()) # Uncomment for detailed debugging if needed
        return None # Return None if drawing fails

# --- Data Export Helpers ---
def _flatten_colors_for_csv(colors_list: list, max_colors: int = 3) -> dict:
    """
    Helper function to flatten a list of color dictionaries into a single
    dictionary for inclusion in a CSV row. Each color's details (name, hex, percentage)
    become prefixed columns (e.g., color_1_name, color_1_hex, ...).

    Args:
        colors_list: A list of color dictionaries, where each dict has
                     "color_name", "hex_code", and "percentage".
        max_colors: The maximum number of colors to flatten.

    Returns:
        A dictionary with flattened color data.
    """
    flat_colors = {}
    for i in range(max_colors):
        # Get color data if available, otherwise use an empty dict
        color_data = colors_list[i] if i < len(colors_list) and isinstance(colors_list[i],dict) else {}
        flat_colors[f"color_{i+1}_name"] = color_data.get("color_name", "") # Default to empty string
        flat_colors[f"color_{i+1}_hex"] = color_data.get("hex_code", "")
        flat_colors[f"color_{i+1}_percentage"] = color_data.get("percentage", "")
    return flat_colors

def convert_fashion_details_to_csv(analysis_data: dict, max_colors_per_item: int = 3) -> str:
    """
    Converts fashion item analysis data (which includes multiple items and their
    respective details like bounding boxes and per-item colors) into a CSV formatted string.

    Args:
        analysis_data: The dictionary returned by `get_fashion_details_from_image`,
                       expected to contain a "fashion_items" list.
        max_colors_per_item: The maximum number of dominant colors to include as separate
                             columns for each fashion item in the CSV.

    Returns:
        A string containing the data in CSV format. Returns an empty string if
        there are no fashion items to process.
    """
    if not analysis_data or "fashion_items" not in analysis_data or not analysis_data.get("fashion_items"):
        return "" # Return empty if no items or invalid data

    output = io.StringIO() # Use an in-memory text buffer to build the CSV

    # Define base fieldnames for each item
    base_fieldnames = ["item_name", "category",
                       "bbox_ymin", "bbox_xmin", "bbox_ymax", "bbox_xmax"]

    # Dynamically create fieldnames for the flattened color data
    color_fieldnames = []
    for i in range(max_colors_per_item):
        color_fieldnames.extend([f"color_{i+1}_name", f"color_{i+1}_hex", f"color_{i+1}_percentage"])

    full_fieldnames = base_fieldnames + color_fieldnames

    # Use DictWriter for robust CSV generation, ensuring consistent line endings
    writer = csv.DictWriter(output, fieldnames=full_fieldnames, lineterminator='\n')
    writer.writeheader() # Write the header row

    for item in analysis_data["fashion_items"]:
        bbox_list = item.get("bounding_box", [None]*4) # Default to list of Nones if bbox missing
        row = {
            "item_name": item.get("item_name", ""), # Default to empty string if missing
            "category": item.get("category", ""),
            # Ensure bounding box coordinates are correctly accessed and defaulted
            "bbox_ymin": bbox_list[0] if len(bbox_list) == 4 else "",
            "bbox_xmin": bbox_list[1] if len(bbox_list) == 4 else "",
            "bbox_ymax": bbox_list[2] if len(bbox_list) == 4 else "",
            "bbox_xmax": bbox_list[3] if len(bbox_list) == 4 else "",
        }

        # Get dominant colors for the item and flatten them
        item_colors = item.get("dominant_colors", [])
        flattened_color_data = _flatten_colors_for_csv(item_colors, max_colors_per_item)
        row.update(flattened_color_data) # Add flattened color data to the row

        writer.writerow(row) # Write the row to the CSV buffer

    return output.getvalue() # Return the complete CSV string

# Deprecated: Old CSV converter for simple overall palette.
# Kept for reference or if a dual-mode analysis is ever re-introduced.
def convert_palette_to_csv(analysis_data: dict) -> str:
    """
    Converts overall color palette data (old, simpler format) to a CSV string.
    This function is considered deprecated in favor of `convert_fashion_details_to_csv`.
    """
    if not analysis_data or "colors" not in analysis_data or not analysis_data.get("colors"):
        return ""
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=["color_name", "hex_code", "percentage"], lineterminator='\n')
    writer.writeheader()
    for color_entry in analysis_data["colors"]:
        writer.writerow({
            "color_name": color_entry.get("color_name", ""),
            "hex_code": color_entry.get("hex_code", ""),
            "percentage": color_entry.get("percentage", "")
        })
    return output.getvalue()

# --- Streamlit Application UI ---
def main_app():
    """
    Main function to define and run the Streamlit web application.
    It handles the UI for file uploading, image processing, triggering AI analysis,
    displaying results (including images with bounding boxes and detailed item information),
    and providing options to download the analysis data.
    """
    # Configure page settings (title, layout). Must be the first Streamlit command.
    st.set_page_config(page_title="AI Fashion Analysis Pro", layout="wide")

    st.title("AI Fashion Analysis Pro 🧥🎨")
    st.markdown(
        "Upload an image to detect fashion items, their categories, specific colors, and more. "
        "Results can be downloaded in JSON or CSV formats."
    )

    # Check if essential AI libraries are available
    if not LANGCHAIN_AVAILABLE:
        st.error("Core AI libraries (langchain-google-genai) are not installed. "
                 "Please ensure dependencies are installed correctly (e.g., from requirements.txt). "
                 "AI-powered features will be disabled.")
        st.stop() # Halt execution if critical AI components are missing

    # --- API Key Management Section ---
    # Initialize API key in session state if not already present
    if 'GEMINI_API_KEY' not in st.session_state:
        st.session_state.GEMINI_API_KEY = None
        # Attempt to load from Colab secrets if in that environment
        try:
            from google.colab import userdata
            st.session_state.GEMINI_API_KEY = userdata.get('GEMINI_API_KEY')
        except (ImportError, ModuleNotFoundError): # Not in Colab or 'google' module not found
            pass # Silently pass if not in Colab; will try .env next

        # If not found via Colab secrets, try to load from .env file (via os.environ)
        if not st.session_state.GEMINI_API_KEY:
            # python-dotenv should have loaded .env vars into os.environ if app is run locally
            st.session_state.GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

    # If API key is still not found after checks, prompt the user for it
    if not st.session_state.GEMINI_API_KEY:
        st.session_state.GEMINI_API_KEY = st.text_input(
            "Enter your Google Gemini API Key:",
            type="password",
            help="Required for AI analysis. For local use, set GEMINI_API_KEY in a .env file. For Colab, use secrets."
        )

    # Gemini model name for vision analysis
    GEMINI_MODEL_NAME = os.environ.get("GEMINI_MODEL_NAME") 

    # --- File Upload Section ---
    uploaded_file = st.file_uploader(
        "1. Upload Fashion Image",
        type=["jpg", "jpeg", "png", "webp"], # Supported image types
        key="file_uploader_main" # Unique key for this widget
    )

    # Initialize session state keys for tracking file and analysis stages
    session_keys_to_init = ['uploaded_fashion_filename', 'processed_fashion_image_bytes',
                            'fashion_analysis_results', 'annotated_image_bytes']
    for key in session_keys_to_init:
        if key not in st.session_state: # Initialize if key doesn't exist
            st.session_state[key] = None

    # --- Main Logic Flow: Process based on file upload status ---
    if uploaded_file is not None:
        # If a new file is uploaded, reset relevant session state variables
        if st.session_state.get('uploaded_fashion_filename') != uploaded_file.name:
            st.session_state.uploaded_fashion_filename = uploaded_file.name
            for key in session_keys_to_init: # Clear all related analysis data
                st.session_state[key] = None
            st.info("New image uploaded. Ready for processing and analysis.") # User feedback

        # Process the image if it hasn't been processed yet for the current file
        if not st.session_state.processed_fashion_image_bytes:
            raw_image_bytes = uploaded_file.getvalue()
            with st.spinner("⚙️ Processing image... This may take a moment for larger files."):
                p_img_data, _, p_msg = process_image(raw_image_bytes, uploaded_file.name)

            if p_img_data:
                st.session_state.processed_fashion_image_bytes = p_img_data
                # Display the processed image to the user
                st.image(p_img_data, caption=f"Processed Image Preview ({p_msg})", use_column_width=True)
            else: # Image processing failed
                st.error(f"Image processing failed: {p_msg}")
                # Clear related session state to allow re-upload or prevent further actions
                st.session_state.uploaded_fashion_filename = None
                st.session_state.processed_fashion_image_bytes = None
                return # Stop further execution for this run to prevent errors

        # --- AI Analysis Trigger Section ---
        # Show analysis button only if image is processed and API key is available
        if st.session_state.processed_fashion_image_bytes:
            if not st.session_state.GEMINI_API_KEY:
                st.warning("Please enter your Gemini API Key above to enable the AI-powered analysis features.")
            else:
                # Button to trigger AI analysis
                if st.button("2. Analyze Fashion Details with AI ✨", key="analyze_fashion_button"):
                    # Clear previous analysis results before starting a new one
                    st.session_state.fashion_analysis_results = None
                    st.session_state.annotated_image_bytes = None

                    # Call the AI analysis function
                    analysis_data = get_fashion_details_from_image(
                        st.session_state.processed_fashion_image_bytes,
                        st.session_state.GEMINI_API_KEY,
                        GEMINI_MODEL_NAME
                    )

                    if analysis_data: # If analysis returned some data (even if no items found)
                        st.session_state.fashion_analysis_results = analysis_data
                        if analysis_data.get("fashion_items"): # Check if any items were actually found and validated
                            st.success("Fashion analysis complete! Items detected.")
                            # Attempt to draw detections on the image
                            with st.spinner("🎨 Generating annotated image with detected items..."):
                                annotated_bytes = draw_detections_on_image(
                                    st.session_state.processed_fashion_image_bytes,
                                    analysis_data
                                )
                            if annotated_bytes:
                                st.session_state.annotated_image_bytes = annotated_bytes
                            else: # Drawing failed, but analysis data might still be useful
                                st.warning("Could not generate the annotated image, but analysis data is available below.")
                        else: # AI ran, but returned no items or items were invalid
                            st.info("AI analysis ran successfully, but no specific fashion items were detected or validated in this image.")
                    # If analysis_data is None, it means a critical error occurred in get_fashion_details_from_image,
                    # and st.error messages would have been shown there.

    else: # No file uploaded
        # Clear all session state if no file is present to ensure a clean slate for next upload
        for key in session_keys_to_init:
            st.session_state.pop(key, None) # Use pop to safely remove if exists
        st.info("👋 Welcome! Please upload an image to begin the AI Fashion Analysis.")

    # --- Display Results Section (Annotated Image and Item Details) ---
    # Display annotated image if available
    if st.session_state.annotated_image_bytes:
        st.subheader("🖼️ Annotated Image with Detected Items")
        st.image(st.session_state.annotated_image_bytes, caption="Detected Fashion Items Highlighted", use_column_width=True)

    # Display detailed analysis results if available and items were found
    if st.session_state.fashion_analysis_results and \
       st.session_state.fashion_analysis_results.get("fashion_items"):

        results = st.session_state.fashion_analysis_results
        st.subheader("📋 Detected Item Details")

        # Iterate through each detected fashion item and display its information
        for i, item in enumerate(results["fashion_items"]):
            with st.expander(f"Item {i+1}: {item.get('item_name', 'N/A')} ({item.get('category', 'N/A')})", expanded=True):
                st.markdown(f"**Category:** {item.get('category', 'N/A')}")
                # Bounding box display can be verbose, uncomment if detailed coordinates are needed by users
                # st.write(f"Bounding Box (ymin, xmin, ymax, xmax): `{item.get('bounding_box')}`")

                st.markdown("**Dominant Colors (for this item):**")
                item_dominant_colors = item.get("dominant_colors", [])
                if item_dominant_colors:
                    # Dynamically create columns for color swatches, max 3-4 for neatness
                    num_item_colors = len(item_dominant_colors)
                    item_color_cols = st.columns(min(num_item_colors, 4))
                    for j, color in enumerate(item_dominant_colors):
                        if j >= 4: break # Show max 4 color swatches in columns to avoid clutter
                        with item_color_cols[j]:
                            hex_color_value = color.get('hex_code', '#FFFFFF') # Default to white if hex is missing
                            # Basic validation for hex code before passing to color_picker
                            if not (isinstance(hex_color_value, str) and \
                                    hex_color_value.startswith('#') and \
                                    len(hex_color_value) in [4, 7]): # Supports short hex like #RGB and full #RRGGBB
                                hex_color_value = '#FFFFFF' # Default to white on invalid format
                                st.caption(f"Color {j+1}: Invalid hex, showing white.")

                            st.color_picker(
                                f"{color.get('color_name', 'N/A')} ({color.get('percentage', '')})",
                                value=hex_color_value,
                                key=f"item_{i}_color_{j}_picker", # Unique key for each color picker
                                disabled=True # Display only, not for user input
                            )
                else: # No dominant colors specified for this item
                    st.write("No dominant colors were specified for this item by the AI.")

        # --- Data Export Section ---
        st.subheader("💾 3. Download Full Analysis")
        # Use original uploaded filename (without extension) as base for download filenames
        base_download_filename = st.session_state.get('uploaded_fashion_filename', 'fashion_analysis').rsplit('.', 1)[0]

        # JSON Download Button
        try:
            json_export_data = json.dumps(results, indent=2) # Pretty print JSON
            st.download_button(
                label="Download Details as JSON",
                data=json_export_data,
                file_name=f"{base_download_filename}_fashion_details.json",
                mime="application/json",
                key="download_fashion_json" # Unique key for the button
            )
        except Exception as e_json: # Catch errors during JSON preparation
            st.error(f"Error preparing JSON data for download: {e_json}")

        # CSV Download Button for Fashion Details
        try:
            csv_export_data = convert_fashion_details_to_csv(results)
            if csv_export_data: # Only show button if CSV string is not empty
                st.download_button(
                    label="Download Details as CSV",
                    data=csv_export_data,
                    file_name=f"{base_download_filename}_fashion_details.csv",
                    mime="text/csv",
                    key="download_fashion_csv" # Unique key
                )
        except Exception as e_csv: # Catch errors during CSV preparation
            st.error(f"Error preparing CSV data for download: {e_csv}")
            # st.error(traceback.format_exc()) # Uncomment for more detailed debug info if needed

    elif st.session_state.fashion_analysis_results and \
         not st.session_state.fashion_analysis_results.get("fashion_items") and \
         st.session_state.get('uploaded_fashion_filename'): # Analysis ran, no items found, and a file was processed
        # This state indicates that AI analysis completed but found no items.
        st.info("AI analysis completed, but no fashion items were identified in the image to display details for.")


if __name__ == '__main__':
    # This allows the script to be run directly using `python app.py` (though `streamlit run app.py` is standard).
    # It's good practice to have the main execution logic within a function like main_app().
    # Load .env file for local development if it exists (for GEMINI_API_KEY)
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        # python-dotenv not installed, API key must be set via environment or Colab secrets
        # This is not a critical error for the app's core logic if key is provided otherwise
        pass

    main_app()
