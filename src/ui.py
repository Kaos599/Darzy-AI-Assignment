# src/ui.py
# (Full content of src/ui.py with modifications for Smart Recommendations)

import streamlit as st
import os

from .constants import MAX_IMAGE_SIZE, JPEG_QUALITY, TARGET_FORMAT, DEFAULT_FONT_SIZE, GEMINI_MODEL_NAME
from .image_utils import process_image, draw_detections_on_image
from .ai_services import (
    get_fashion_details_from_image,
    LANGCHAIN_AVAILABLE,
    get_size_estimation_for_items,
    generate_fashion_copy,
    get_smart_recommendations # Added get_smart_recommendations
)
from .data_exporters import convert_fashion_details_to_csv, export_to_json_string

def display_app():
    # Renders the Streamlit User Interface and orchestrates the application flow.
    st.set_page_config(page_title="AI Fashion Analysis Pro", layout="wide")
    st.title("AI Fashion Analysis Pro 🧥🎨")
    st.markdown(
        "Upload an image to detect fashion items, estimate sizes, generate text, get recommendations, and more."
    )

    if not LANGCHAIN_AVAILABLE:
        st.error("Core AI libraries (langchain-google-genai) are not installed or configured correctly. AI features disabled.")

    # --- API Key Management ---
    if 'GEMINI_API_KEY' not in st.session_state:
        st.session_state.GEMINI_API_KEY = None
        try: from google.colab import userdata; st.session_state.GEMINI_API_KEY = userdata.get('GEMINI_API_KEY')
        except (ImportError, ModuleNotFoundError): pass
        if not st.session_state.GEMINI_API_KEY: st.session_state.GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
    if not st.session_state.GEMINI_API_KEY:
        st.session_state.GEMINI_API_KEY = st.text_input("Enter Google Gemini API Key:", type="password", help="Required for AI. Store in .env or Colab secret.")

    # --- File Upload ---
    uploaded_file = st.file_uploader("1. Upload Fashion Image", type=["jpg", "jpeg", "png", "webp"], key="file_uploader_main_ui")

    # --- Session State Initialization ---
    session_keys_to_init = [
        'uploaded_fashion_filename_ui', 'processed_fashion_image_bytes_ui',
        'fashion_analysis_results_ui', 'annotated_image_bytes_ui',
        'size_estimation_results_ui', 'fashion_copy_results_ui',
        'smart_recommendations_ui' # Added for smart recommendations
    ]
    for key in session_keys_to_init:
        if key not in st.session_state: st.session_state[key] = None

    # --- Main Application Logic ---
    if uploaded_file is not None:
        if st.session_state.get('uploaded_fashion_filename_ui') != uploaded_file.name:
            st.session_state.uploaded_fashion_filename_ui = uploaded_file.name
            for key in session_keys_to_init: st.session_state[key] = None

        if not st.session_state.processed_fashion_image_bytes_ui:
            raw_image_bytes = uploaded_file.getvalue()
            with st.spinner("⚙️ Processing image..."):
                p_img_data, _, p_msg = process_image(raw_image_bytes, uploaded_file.name)
            if p_img_data:
                st.session_state.processed_fashion_image_bytes_ui = p_img_data
                st.image(p_img_data, caption=f"Processed Image ({p_msg})", use_container_width=True)
            else:
                st.error(f"Image processing failed: {p_msg}")
                st.session_state.uploaded_fashion_filename_ui = None
                st.session_state.processed_fashion_image_bytes_ui = None

        if st.session_state.processed_fashion_image_bytes_ui:
            if not LANGCHAIN_AVAILABLE:
                 st.error("AI features disabled due to missing libraries.")
            elif not st.session_state.GEMINI_API_KEY:
                st.warning("Please enter your Gemini API Key to enable AI features.")
            else:
                st.subheader("2. Choose AI Analysis Feature(s)")
                button_cols = st.columns(4)

                with button_cols[0]: # Fashion Details
                    if st.button("👁️ Analyze Details", key="analyze_fashion_button_ui", help="Detect items, colors, categories."):
                        st.session_state.fashion_analysis_results_ui = None; st.session_state.annotated_image_bytes_ui = None
                        st.session_state.size_estimation_results_ui = None; st.session_state.fashion_copy_results_ui = None
                        st.session_state.smart_recommendations_ui = None # Clear smart recommendations too
                        with st.spinner("🔍 Analyzing fashion items..."):
                            analysis_data = get_fashion_details_from_image(st.session_state.processed_fashion_image_bytes_ui, st.session_state.GEMINI_API_KEY, GEMINI_MODEL_NAME)
                        if analysis_data:
                            st.session_state.fashion_analysis_results_ui = analysis_data
                            if analysis_data.get("fashion_items"):
                                st.success("Fashion item analysis complete!");
                                with st.spinner("🎨 Generating annotated image..."):
                                    annotated_bytes = draw_detections_on_image(st.session_state.processed_fashion_image_bytes_ui, analysis_data)
                                if annotated_bytes: st.session_state.annotated_image_bytes_ui = annotated_bytes
                                else: st.warning("Could not generate annotated image.")
                            else: st.info("Fashion item analysis ran, but no items detected.")
                        else: st.error("Fashion item analysis failed.")

                with button_cols[1]: # Size Estimation
                    if st.button("📏 Estimate Sizes", key="estimate_size_button_ui", help="Suggest garment sizes based on the image."):
                        st.session_state.size_estimation_results_ui = None
                        detected_items_ctx = st.session_state.fashion_analysis_results_ui["fashion_items"] if st.session_state.fashion_analysis_results_ui and st.session_state.fashion_analysis_results_ui.get("fashion_items") else None
                        with st.spinner("📐 Estimating garment sizes..."):
                            size_results = get_size_estimation_for_items(st.session_state.processed_fashion_image_bytes_ui, st.session_state.GEMINI_API_KEY, GEMINI_MODEL_NAME, detected_items=detected_items_ctx)
                        if size_results and "size_estimations" in size_results:
                            st.session_state.size_estimation_results_ui = size_results
                            if size_results.get("size_estimations"): st.success("Size estimation complete!")
                            else: st.info("AI size estimation ran, but no estimates provided.")
                        else: st.error("Size estimation failed.")

                with button_cols[2]: # Fashion Copywriter
                    if st.button("✍️ Generate Text", key="generate_copy_button_ui", help="Create product descriptions, styling tips, social media captions."):
                        st.session_state.fashion_copy_results_ui = None
                        detected_items_ctx = st.session_state.fashion_analysis_results_ui["fashion_items"] if st.session_state.fashion_analysis_results_ui and st.session_state.fashion_analysis_results_ui.get("fashion_items") else None
                        with st.spinner("✍️ Generating fashion copy..."):
                            copy_results = generate_fashion_copy(st.session_state.processed_fashion_image_bytes_ui, st.session_state.GEMINI_API_KEY, GEMINI_MODEL_NAME, detected_items=detected_items_ctx)
                        if copy_results:
                            st.session_state.fashion_copy_results_ui = copy_results
                            if any(copy_results.get(key) for key in ["product_description", "styling_suggestions", "social_media_caption"]):
                                st.success("Fashion text generation complete!")
                            else: st.info("AI did not generate any fashion text content.")
                        else: st.error("Fashion text generation failed.")

                with button_cols[3]: # Smart Recommendations (New)
                    fashion_details_available = st.session_state.get('fashion_analysis_results_ui') and st.session_state.fashion_analysis_results_ui.get("fashion_items")
                    if st.button("💡 Get Recommendations", key="smart_recs_button_ui",
                                 disabled=not fashion_details_available,
                                 help="Suggest complementary items, similar styles, and seasonal advice. Requires fashion details to be analyzed first." ):
                        st.session_state.smart_recommendations_ui = None
                        
                        detected_items_for_recs = []
                        if st.session_state.get('fashion_analysis_results_ui') and st.session_state.fashion_analysis_results_ui.get("fashion_items"):
                            detected_items_for_recs = st.session_state.fashion_analysis_results_ui["fashion_items"]

                        with st.spinner("💡 Generating smart recommendations..."):
                            recs_results = get_smart_recommendations(
                                st.session_state.GEMINI_API_KEY,
                                GEMINI_MODEL_NAME,
                                detected_items=detected_items_for_recs
                            )
                        if recs_results:
                            st.session_state.smart_recommendations_ui = recs_results
                            if any(recs_results.get(key) and recs_results.get(key) not in ["Error: Could not parse AI response.", "Error: AI response was malformed JSON.", "Error: AI returned unexpected data structure.", "An error occurred while generating recommendations."] for key in ["complementary_suggestions", "similar_styles", "seasonal_advice"]):
                                st.success("Smart recommendations generated!")
                            elif "No items were provided" in recs_results.get("complementary_suggestions",""): # Check for specific no-item message
                                st.info("Recommendations require item details. Please analyze fashion details first.")
                            else: # Handles cases where service returns dict with error messages or empty strings
                                st.info("AI did not generate specific smart recommendations or encountered an issue.")
                        else:
                            st.error("Smart recommendations generation failed or returned unexpected data.")
    else:
        st.info("👋 Welcome! Please upload an image to begin.")

    st.markdown("---")

    if 'annotated_image_bytes_ui' in st.session_state and st.session_state.annotated_image_bytes_ui:
        st.subheader("🖼️ Annotated Image")
        st.image(st.session_state.annotated_image_bytes_ui, caption="Detected Fashion Items", use_container_width=True)

    if 'fashion_analysis_results_ui' in st.session_state and st.session_state.fashion_analysis_results_ui and st.session_state.fashion_analysis_results_ui.get("fashion_items"):
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
                            if not (isinstance(hex_color_value, str) and hex_color_value.startswith('#') and len(hex_color_value) in [4, 7]):
                                hex_color_value = '#FFFFFF'; st.caption("Invalid hex.")
                            st.color_picker(f"{color.get('color_name', 'N/A')} ({color.get('percentage', '')})", value=hex_color_value, key=f"item_{i}_color_{j}_picker_ui", disabled=True)
                else: st.write("No dominant colors specified for this item.")
        st.subheader("💾 Download Fashion Details")
        # Retrieve filename, ensuring it's a string before splitting
        current_filename = st.session_state.get('uploaded_fashion_filename_ui')
        if current_filename is None:
            base_fn = 'fashion_analysis'
        else:
            base_fn = current_filename.rsplit('.', 1)[0]

        try:
            json_export_data = export_to_json_string(results)
            if json_export_data: st.download_button(label="Download Details as JSON", data=json_export_data, file_name=f"{base_fn}_fashion_details.json", mime="application/json", key="download_fashion_json_ui")
            else: st.error("Failed to prepare JSON for fashion details.")
        except Exception as e: st.error(f"JSON export error (details): {e}")
        try:
            csv_export_data = convert_fashion_details_to_csv(results)
            if csv_export_data: st.download_button(label="Download Details as CSV", data=csv_export_data, file_name=f"{base_fn}_fashion_details.csv", mime="text/csv", key="download_fashion_csv_ui")
        except Exception as e: st.error(f"CSV export error (details): {e}")

    elif 'fashion_analysis_results_ui' in st.session_state and st.session_state.fashion_analysis_results_ui and not st.session_state.fashion_analysis_results_ui.get("fashion_items") and 'uploaded_fashion_filename_ui' in st.session_state and st.session_state.get('uploaded_fashion_filename_ui'):
        st.info("Fashion detail analysis completed, but no items were identified.")

    if 'size_estimation_results_ui' in st.session_state and st.session_state.size_estimation_results_ui:
        st.subheader("📏 Size Estimations")
        estimations = st.session_state.size_estimation_results_ui.get("size_estimations", [])
        if not estimations: st.info("No specific size estimations were provided.")
        else:
            for i, est in enumerate(estimations):
                with st.expander(f"Estimation for: {est.get('item_description', 'N/A')}", expanded=True):
                    st.markdown(f"**Suggested Size:** {est.get('estimated_size', 'N/A')}")
                    st.markdown(f"**Reasoning:** {est.get('reasoning', 'N/A')}")

    if 'fashion_copy_results_ui' in st.session_state and st.session_state.fashion_copy_results_ui:
        st.subheader("✍️ AI Generated Fashion Text")
        copy_data = st.session_state.fashion_copy_results_ui
        if any(copy_data.get(key) for key in ["product_description", "styling_suggestions", "social_media_caption"]):
            if copy_data.get("product_description"): st.markdown("#### Product Description"); st.markdown(copy_data["product_description"])
            if copy_data.get("styling_suggestions"): st.markdown("#### Styling Suggestions"); st.markdown(copy_data["styling_suggestions"])
            if copy_data.get("social_media_caption"): st.markdown("#### Social Media Caption"); st.markdown(copy_data["social_media_caption"])
        else:
            st.info("AI did not generate any specific text content for this image.")

    if 'smart_recommendations_ui' in st.session_state and st.session_state.smart_recommendations_ui:
        st.subheader("💡 Smart Recommendations")
        recs_data = st.session_state.smart_recommendations_ui
        if any(recs_data.get(key) for key in ["complementary_suggestions", "similar_styles", "seasonal_advice"]):
            if recs_data.get("complementary_suggestions"): st.markdown("#### Complementary Suggestions"); st.markdown(recs_data["complementary_suggestions"])
            if recs_data.get("similar_styles"): st.markdown("#### Similar Styles You Might Like"); st.markdown(recs_data["similar_styles"])
            if recs_data.get("seasonal_advice"): st.markdown("#### Seasonal Styling Advice"); st.markdown(recs_data["seasonal_advice"])
        else:
             st.info("AI did not generate any specific smart recommendations or encountered an issue processing the request.")

# No if __name__ == '__main__': here
