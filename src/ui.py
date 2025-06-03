# src/ui.py

import streamlit as st
import os
# import traceback # Keep commented unless specific debugging is needed in UI

from .constants import MAX_IMAGE_SIZE, JPEG_QUALITY, TARGET_FORMAT, DEFAULT_FONT_SIZE
from .image_utils import process_image, draw_detections_on_image
from .ai_services import get_fashion_details_from_image, LANGCHAIN_AVAILABLE
from .data_exporters import convert_fashion_details_to_csv, export_to_json_string

def display_app():
    # Renders the Streamlit User Interface and orchestrates the application flow.
    st.set_page_config(page_title="AI Fashion Analysis Pro", layout="wide")
    st.title("AI Fashion Analysis Pro 🧥🎨")
    st.markdown(
        "Upload an image to detect fashion items, their categories, colors, and download results. "
        "This refactored version provides detailed per-item analysis."
    )

    if not LANGCHAIN_AVAILABLE:
        st.error("Core AI libraries (langchain-google-genai) are not installed or configured correctly. "
                 "Please ensure dependencies are installed. AI features will be disabled.")

    if 'GEMINI_API_KEY' not in st.session_state:
        st.session_state.GEMINI_API_KEY = None
        try:
            from google.colab import userdata
            st.session_state.GEMINI_API_KEY = userdata.get('GEMINI_API_KEY')
        except (ImportError, ModuleNotFoundError): # Not in Colab or google module not found
            pass
        if not st.session_state.GEMINI_API_KEY:
            st.session_state.GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

    if not st.session_state.GEMINI_API_KEY:
        st.session_state.GEMINI_API_KEY = st.text_input(
            "Enter your Google Gemini API Key:",
            type="password",
            help="Required for AI analysis. Store as GEMINI_API_KEY in a .env file or Colab secret."
        )

    GEMINI_MODEL_NAME = os.environ.get("GEMINI_MODEL_NAME", "gemini-pro-vision")

    uploaded_file = st.file_uploader(
        "1. Upload Fashion Image",
        type=["jpg", "jpeg", "png", "webp"],
        key="file_uploader_main_ui"
    )

    session_keys_to_init = ['uploaded_fashion_filename_ui', 'processed_fashion_image_bytes_ui',
                            'fashion_analysis_results_ui', 'annotated_image_bytes_ui']
    for key in session_keys_to_init:
        if key not in st.session_state:
            st.session_state[key] = None

    if uploaded_file is not None:
        if st.session_state.get('uploaded_fashion_filename_ui') != uploaded_file.name:
            st.session_state.uploaded_fashion_filename_ui = uploaded_file.name
            for key in session_keys_to_init:
                st.session_state[key] = None
            # st.info("New image uploaded. Ready for processing and analysis.") # Can be a bit noisy

        if not st.session_state.processed_fashion_image_bytes_ui:
            raw_image_bytes = uploaded_file.getvalue()
            with st.spinner("⚙️ Processing image..."):
                p_img_data, _, p_msg = process_image(raw_image_bytes, uploaded_file.name)

            if p_img_data:
                st.session_state.processed_fashion_image_bytes_ui = p_img_data
                st.image(p_img_data, caption=f"Processed Image ({p_msg})", use_column_width=True)
            else:
                st.error(f"Image processing failed: {p_msg}")
                st.session_state.uploaded_fashion_filename_ui = None
                st.session_state.processed_fashion_image_bytes_ui = None
                # Allow UI to stay interactive

        if st.session_state.processed_fashion_image_bytes_ui:
            if not LANGCHAIN_AVAILABLE:
                 st.error("AI features disabled due to missing libraries.")
            elif not st.session_state.GEMINI_API_KEY:
                st.warning("Please enter your Gemini API Key above to enable AI-powered analysis.")
            else:
                if st.button("2. Analyze Fashion Details with AI ✨", key="analyze_fashion_button_ui"):
                    st.session_state.fashion_analysis_results_ui = None
                    st.session_state.annotated_image_bytes_ui = None

                    analysis_data = get_fashion_details_from_image(
                        st.session_state.processed_fashion_image_bytes_ui,
                        st.session_state.GEMINI_API_KEY,
                        GEMINI_MODEL_NAME
                    )

                    if analysis_data:
                        st.session_state.fashion_analysis_results_ui = analysis_data
                        if analysis_data.get("fashion_items"):
                            st.success("Fashion analysis complete! Items detected.")
                            with st.spinner("🎨 Generating annotated image..."):
                                annotated_bytes = draw_detections_on_image(
                                    st.session_state.processed_fashion_image_bytes_ui,
                                    analysis_data
                                )
                            if annotated_bytes:
                                st.session_state.annotated_image_bytes_ui = annotated_bytes
                            else:
                                st.warning("Could not generate annotated image, but analysis data is available.")
                        else:
                            st.info("AI analysis ran successfully, but no specific fashion items were detected in this image.")
                    else:
                        st.error("AI analysis failed. Check logs or try a different image/API key.")
    else:
        for key in session_keys_to_init:
            st.session_state.pop(key, None)
        st.info("👋 Welcome! Please upload an image to begin the AI Fashion Analysis.")

    if st.session_state.annotated_image_bytes_ui:
        st.subheader("🖼️ Annotated Image with Detected Items")
        st.image(st.session_state.annotated_image_bytes_ui, caption="Detected Fashion Items", use_column_width=True)

    if st.session_state.fashion_analysis_results_ui and \
       st.session_state.fashion_analysis_results_ui.get("fashion_items"):

        results = st.session_state.fashion_analysis_results_ui
        st.subheader("📋 Detected Item Details")

        for i, item in enumerate(results["fashion_items"]):
            with st.expander(f"Item {i+1}: {item.get('item_name', 'N/A')} ({item.get('category', 'N/A')})", expanded=True):
                st.markdown(f"**Category:** {item.get('category', 'N/A')}")

                st.markdown("**Dominant Colors (for this item):**")
                item_dominant_colors = item.get("dominant_colors", [])
                if item_dominant_colors:
                    num_item_colors = len(item_dominant_colors)
                    item_color_cols = st.columns(min(num_item_colors, 3))
                    for j, color in enumerate(item_dominant_colors):
                        if j >= 3: break
                        with item_color_cols[j]:
                            hex_color_value = color.get('hex_code', '#FFFFFF')
                            if not (isinstance(hex_color_value, str) and \
                                    hex_color_value.startswith('#') and \
                                    len(hex_color_value) in [4, 7]):
                                hex_color_value = '#FFFFFF'; st.caption("Invalid hex.")

                            st.color_picker(
                                f"{color.get('color_name', 'N/A')} ({color.get('percentage', '')})",
                                value=hex_color_value,
                                key=f"item_{i}_color_{j}_picker_ui",
                                disabled=True
                            )
                else:
                    st.write("No dominant colors were specified for this item by the AI.")

        st.subheader("💾 3. Download Full Analysis")
        base_fn = st.session_state.get('uploaded_fashion_filename_ui', 'fashion_analysis').rsplit('.', 1)[0]

        try:
            json_export_data = export_to_json_string(results)
            if json_export_data:
                st.download_button(
                    label="Download Details as JSON", data=json_export_data,
                    file_name=f"{base_fn}_fashion_details.json", mime="application/json",
                    key="download_fashion_json_ui"
                )
            else:
                st.error("Failed to prepare JSON data for download (export function returned empty).")
        except Exception as e_json:
            st.error(f"Error during JSON export preparation: {e_json}")

        try:
            csv_export_data = convert_fashion_details_to_csv(results)
            if csv_export_data:
                st.download_button(
                    label="Download Details as CSV", data=csv_export_data,
                    file_name=f"{base_fn}_fashion_details.csv", mime="text/csv",
                    key="download_fashion_csv_ui"
                )
        except Exception as e_csv:
            st.error(f"Error during CSV export preparation: {e_csv}")

    elif st.session_state.fashion_analysis_results_ui and \
         not st.session_state.fashion_analysis_results_ui.get("fashion_items") and \
         st.session_state.get('uploaded_fashion_filename_ui'):
        st.info("AI analysis completed, but no fashion items were identified in the image to display details for.")

# Note: No `if __name__ == '__main__':` block here.
# This module is intended to be imported and display_app() called by main.py.
