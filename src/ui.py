# src/ui.py
# (Full content of src/ui.py with modifications for Smart Recommendations)

import streamlit as st
import os
import base64
import uuid # To generate unique image IDs

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
from .database_manager import (
    save_image_metadata,
    load_image_metadata,
    save_analysis_results,
    load_analysis_results,
    close_db_connection
)

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

    # --- Session State for Image ID (persistent across reruns) ---
    if 'current_image_id' not in st.session_state:
        st.session_state.current_image_id = None
    if 'uploaded_file_name_display' not in st.session_state:
        st.session_state.uploaded_file_name_display = None

    # --- File Upload ---
    uploaded_file = st.file_uploader("1. Upload Fashion Image", type=["jpg", "jpeg", "png", "webp"], key="file_uploader_main_ui")

    # --- Main Application Logic ---
    if uploaded_file is not None:
        # Check if a new file has been uploaded (by name change or clearing)
        if st.session_state.uploaded_file_name_display != uploaded_file.name:
            # New file uploaded, generate new ID and clear previous analysis results from memory
            st.session_state.current_image_id = str(uuid.uuid4()) # Generate a new unique ID
            st.session_state.uploaded_file_name_display = uploaded_file.name
            print(f"DEBUG: New image uploaded: {uploaded_file.name}. New image ID: {st.session_state.current_image_id}")
            st.success(f"New image '{uploaded_file.name}' uploaded. Processing...")
            # For a new image, clear all displayed results from session state to avoid stale data
            st.session_state.fashion_analysis_results_ui = None
            st.session_state.annotated_image_bytes_ui = None
            st.session_state.size_estimation_results_ui = None
            st.session_state.fashion_copy_results_ui = None
            st.session_state.smart_recommendations_ui = None

        # Process image if it hasn't been processed and saved for the current image_id
        image_metadata = None
        if st.session_state.current_image_id:
            print(f"DEBUG: Attempting to load image metadata for ID: {st.session_state.current_image_id}")
            image_metadata = load_image_metadata(st.session_state.current_image_id)
            if image_metadata: print("DEBUG: Image metadata loaded from DB.")
            else: print("DEBUG: Image metadata not found in DB.")

        processed_image_bytes = None
        if image_metadata and "processed_image_b64" in image_metadata:
            processed_image_bytes = base64.b64decode(image_metadata["processed_image_b64"])
            st.image(processed_image_bytes, caption=f"Processed Image Preview (Loaded from DB)", use_container_width=True)
        else:
            raw_image_bytes = uploaded_file.getvalue()
            with st.spinner("⚙️ Processing image..."):
                p_img_data, _, p_msg = process_image(raw_image_bytes, uploaded_file.name)
            if p_img_data:
                processed_image_bytes = p_img_data
                st.image(processed_image_bytes, caption=f"Processed Image ({p_msg})", use_container_width=True)
                # Save processed image to DB
                if st.session_state.current_image_id:
                    print(f"DEBUG: Saving processed image metadata for ID: {st.session_state.current_image_id}")
                    save_image_metadata(st.session_state.current_image_id, uploaded_file.name, base64.b64encode(p_img_data).decode('utf-8'))
            else:
                st.error(f"Image processing failed: {p_msg}")
                st.session_state.current_image_id = None # Clear image ID if processing failed
                return

        # Only proceed if image is processed and API key is available
        if processed_image_bytes and st.session_state.GEMINI_API_KEY:
            st.subheader("2. Choose AI Analysis Feature(s)")
            button_cols = st.columns(4)

            # --- Fashion Details Button ---
            with button_cols[0]:
                if st.button("👁️ Analyze Details", key="analyze_fashion_button_ui", help="Detect items, colors, categories."):
                    st.session_state.fashion_analysis_results_ui = None # Clear current in-memory display
                    st.session_state.annotated_image_bytes_ui = None # Clear annotated image
                    st.session_state.size_estimation_results_ui = None
                    st.session_state.fashion_copy_results_ui = None
                    st.session_state.smart_recommendations_ui = None

                    with st.spinner("🔍 Analyzing fashion items..."):
                        analysis_data = get_fashion_details_from_image(
                            processed_image_bytes,
                            st.session_state.GEMINI_API_KEY,
                            GEMINI_MODEL_NAME
                        )

                    if analysis_data:
                        st.success("Fashion item analysis complete!")
                        # Save to DB
                        if st.session_state.current_image_id:
                            print(f"DEBUG: Saving fashion_details for ID: {st.session_state.current_image_id}")
                            save_analysis_results(st.session_state.current_image_id, analysis_data, "fashion_details")
                        # Update in-memory for immediate display (optional, but good for responsiveness)
                        st.session_state.fashion_analysis_results_ui = analysis_data

                        if analysis_data.get("fashion_items"):
                            with st.spinner("🎨 Generating annotated image..."):
                                annotated_bytes = draw_detections_on_image(processed_image_bytes, analysis_data)
                            if annotated_bytes:
                                st.session_state.annotated_image_bytes_ui = annotated_bytes
                            else:
                                st.warning("Could not generate annotated image.")
                        else:
                            st.info("Fashion item analysis ran, but no items detected.")
                    else:
                        st.error("Fashion item analysis failed.")

            # --- Size Estimation Button ---
            with button_cols[1]:
                if st.button("📏 Estimate Sizes", key="estimate_size_button_ui", help="Suggest garment sizes based on the image."):
                    st.session_state.size_estimation_results_ui = None # Clear current in-memory display
                    # Attempt to load fashion details from DB for context
                    fashion_details_from_db = None
                    if st.session_state.current_image_id:
                        print(f"DEBUG: Attempting to load fashion_details from DB for ID: {st.session_state.current_image_id} for size estimation.")
                        fashion_details_from_db = load_analysis_results(st.session_state.current_image_id, "fashion_details")
                        if fashion_details_from_db: print("DEBUG: Fashion details loaded from DB for size estimation.")
                        else: print("DEBUG: Fashion details not found in DB for size estimation.")

                    detected_items_ctx = fashion_details_from_db.get("fashion_items") if fashion_details_from_db else None

                    with st.spinner("📐 Estimating garment sizes..."):
                        size_results = get_size_estimation_for_items(
                            processed_image_bytes,
                            st.session_state.GEMINI_API_KEY,
                            GEMINI_MODEL_NAME,
                            detected_items=detected_items_ctx
                        )
                    if size_results and "size_estimations" in size_results:
                        st.success("Size estimation complete!")
                        # Save to DB
                        if st.session_state.current_image_id:
                            print(f"DEBUG: Saving size_estimation for ID: {st.session_state.current_image_id}")
                            save_analysis_results(st.session_state.current_image_id, size_results, "size_estimation")
                        # Update in-memory for immediate display
                        st.session_state.size_estimation_results_ui = size_results
                        if not size_results.get("size_estimations"): st.info("AI size estimation ran, but no estimates provided.")
                    else:
                        st.error("Size estimation failed.")

            # --- Fashion Copywriter Button ---
            with button_cols[2]:
                if st.button("✍️ Generate Text", key="generate_copy_button_ui", help="Create product descriptions, styling tips, social media captions."):
                    st.session_state.fashion_copy_results_ui = None # Clear current in-memory display
                    # Attempt to load fashion details from DB for context
                    fashion_details_from_db = None
                    if st.session_state.current_image_id:
                        print(f"DEBUG: Attempting to load fashion_details from DB for ID: {st.session_state.current_image_id} for fashion copy.")
                        fashion_details_from_db = load_analysis_results(st.session_state.current_image_id, "fashion_details")
                        if fashion_details_from_db: print("DEBUG: Fashion details loaded from DB for fashion copy.")
                        else: print("DEBUG: Fashion details not found in DB for fashion copy.")

                    detected_items_ctx = fashion_details_from_db.get("fashion_items") if fashion_details_from_db else None

                    with st.spinner("✍️ Generating fashion copy..."):
                        copy_results = generate_fashion_copy(
                            processed_image_bytes,
                            st.session_state.GEMINI_API_KEY,
                            GEMINI_MODEL_NAME,
                            detected_items=detected_items_ctx
                        )
                    if copy_results:
                        st.success("Fashion text generation complete!")
                        # Save to DB
                        if st.session_state.current_image_id:
                            print(f"DEBUG: Saving fashion_copy for ID: {st.session_state.current_image_id}")
                            save_analysis_results(st.session_state.current_image_id, copy_results, "fashion_copy")
                        # Update in-memory for immediate display
                        st.session_state.fashion_copy_results_ui = copy_results
                        if not any(copy_results.get(key) for key in ["product_description", "styling_suggestions", "social_media_caption"]):
                            st.info("AI did not generate any fashion text content.")
                    else:
                        st.error("Fashion text generation failed.")

            # --- Smart Recommendations Button ---
            with button_cols[3]:
                # This button should only be enabled if fashion details are in the DB.
                # We will check the DB for fashion details to enable/disable.
                db_fashion_details_available = False
                fashion_details_for_recs = None
                if st.session_state.current_image_id:
                    print(f"DEBUG: Attempting to load fashion_details from DB for ID: {st.session_state.current_image_id} for recommendations.")
                    fashion_details_for_recs = load_analysis_results(st.session_state.current_image_id, "fashion_details")
                    if fashion_details_for_recs: print("DEBUG: Fashion details loaded from DB for recommendations.")
                    else: print("DEBUG: Fashion details not found in DB for recommendations.")

                    if fashion_details_for_recs and fashion_details_for_recs.get("fashion_items"):
                        db_fashion_details_available = True

                if st.button("💡 Get Recommendations", key="smart_recs_button_ui",
                             disabled=not db_fashion_details_available,
                             help="Suggest complementary items, similar styles, and seasonal advice. Requires fashion details to be analyzed first from DB." ):
                    st.session_state.smart_recommendations_ui = None # Clear current in-memory display

                    if db_fashion_details_available and fashion_details_for_recs:
                        detected_items_for_recs = fashion_details_for_recs["fashion_items"]
                        print(f"DEBUG: Calling get_smart_recommendations with {len(detected_items_for_recs)} items.")

                        with st.spinner("💡 Generating smart recommendations..."):
                            recs_results = get_smart_recommendations(
                                st.session_state.GEMINI_API_KEY,
                                GEMINI_MODEL_NAME,
                                detected_items=detected_items_for_recs
                            )
                        if recs_results:
                            st.success("Smart recommendations generated!")
                            # Save to DB
                            if st.session_state.current_image_id:
                                print(f"DEBUG: Saving smart_recommendations for ID: {st.session_state.current_image_id}")
                                save_analysis_results(st.session_state.current_image_id, recs_results, "smart_recommendations")
                            # Update in-memory for immediate display
                            st.session_state.smart_recommendations_ui = recs_results
                            if not any(recs_results.get(key) for key in ["complementary_suggestions", "similar_styles", "seasonal_advice"]):
                                st.info("AI did not generate specific smart recommendations or encountered an issue.")
                        else:
                            st.error("Smart recommendations generation failed or returned unexpected data.")
                    else:
                        st.info("Recommendations cannot be generated without detected fashion items (from DB). Please analyze fashion details first.")
        else:
            # Message if image is processed but API key is missing/invalid
            if processed_image_bytes and not st.session_state.GEMINI_API_KEY:
                st.warning("Please enter your Gemini API Key above to enable AI features.")
            elif not processed_image_bytes:
                 st.info("👋 Welcome! Please upload an image to begin.")

    elif st.session_state.uploaded_file_name_display is None: # Initial state or after clearing all
        st.info("👋 Welcome! Please upload an image to begin.")

    st.markdown("---")

    # --- Display Results Section (Annotated Image and Item Details) ---
    # Attempt to load all results from DB if image_id exists, otherwise use session state (for initial run before DB save)
    current_image_id = st.session_state.get('current_image_id')

    display_annotated_image = st.session_state.get('annotated_image_bytes_ui')
    display_fashion_results = st.session_state.get('fashion_analysis_results_ui')
    display_size_results = st.session_state.get('size_estimation_results_ui')
    display_copy_results = st.session_state.get('fashion_copy_results_ui')
    display_smart_recs_results = st.session_state.get('smart_recommendations_ui')

    # If an image_id is present, try to load from DB for display
    if current_image_id:
        print(f"DEBUG: Display section - current_image_id: {current_image_id}")
        if not display_fashion_results: # If not already loaded in session state, try DB
            print(f"DEBUG: Loading fashion_details for display from DB for ID: {current_image_id}")
            display_fashion_results = load_analysis_results(current_image_id, "fashion_details")
            if display_fashion_results: print("DEBUG: Fashion details loaded for display from DB.")
            else: print("DEBUG: Fashion details not found for display in DB.")

            if display_fashion_results and display_fashion_results.get("fashion_items"):
                # Re-generate annotated image if fashion details loaded from DB and not in session state
                print(f"DEBUG: Loading image metadata for annotated image regeneration for ID: {current_image_id}")
                img_metadata = load_image_metadata(current_image_id)
                if img_metadata and "processed_image_b64" in img_metadata:
                    p_img_bytes = base64.b64decode(img_metadata["processed_image_b64"])
                    display_annotated_image = draw_detections_on_image(p_img_bytes, display_fashion_results)
                    if display_annotated_image: print("DEBUG: Annotated image regenerated.")
                    else: print("DEBUG: Failed to regenerate annotated image.")
                else:
                    print("DEBUG: Processed image bytes not available for annotated image regeneration.")

        if not display_size_results: # If not already loaded in session state, try DB
            print(f"DEBUG: Loading size_estimation for display from DB for ID: {current_image_id}")
            display_size_results = load_analysis_results(current_image_id, "size_estimation")
            if display_size_results: print("DEBUG: Size estimation loaded for display from DB.")
            else: print("DEBUG: Size estimation not found for display in DB.")

        if not display_copy_results: # If not already loaded in session state, try DB
            print(f"DEBUG: Loading fashion_copy for display from DB for ID: {current_image_id}")
            display_copy_results = load_analysis_results(current_image_id, "fashion_copy")
            if display_copy_results: print("DEBUG: Fashion copy loaded for display from DB.")
            else: print("DEBUG: Fashion copy not found for display in DB.")

        if not display_smart_recs_results: # If not already loaded in session state, try DB
            print(f"DEBUG: Loading smart_recommendations for display from DB for ID: {current_image_id}")
            display_smart_recs_results = load_analysis_results(current_image_id, "smart_recommendations")
            if display_smart_recs_results: print("DEBUG: Smart recommendations loaded for display from DB.")
            else: print("DEBUG: Smart recommendations not found for display in DB.")

    if display_annotated_image:
        st.subheader("🖼️ Annotated Image")
        st.image(display_annotated_image, caption="Detected Fashion Items", use_container_width=True)

    if display_fashion_results and display_fashion_results.get("fashion_items"):
        results = display_fashion_results
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

        current_filename = st.session_state.get('uploaded_file_name_display')
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

    elif display_fashion_results and not display_fashion_results.get("fashion_items") and st.session_state.get('uploaded_file_name_display'):
        st.info("Fashion detail analysis completed, but no items were identified.")

    if display_size_results:
        st.subheader("📏 Size Estimations")
        estimations = display_size_results.get("size_estimations", [])
        if not estimations: st.info("No specific size estimations were provided.")
        else:
            for i, est in enumerate(estimations):
                with st.expander(f"Estimation for: {est.get('item_description', 'N/A')}", expanded=True):
                    st.markdown(f"**Suggested Size:** {est.get('estimated_size', 'N/A')}")
                    st.markdown(f"**Reasoning:** {est.get('reasoning', 'N/A')}")

    if display_copy_results:
        st.subheader("✍️ AI Generated Fashion Text")
        copy_data = display_copy_results
        if any(copy_data.get(key) for key in ["product_description", "styling_suggestions", "social_media_caption"]):
            if copy_data.get("product_description"): st.markdown("#### Product Description"); st.markdown(copy_data["product_description"])
            if copy_data.get("styling_suggestions"): st.markdown("#### Styling Suggestions"); st.markdown(copy_data["styling_suggestions"])
            if copy_data.get("social_media_caption"): st.markdown("#### Social Media Caption"); st.markdown(copy_data["social_media_caption"])
        else:
            st.info("AI did not generate any specific text content for this image.")

    if display_smart_recs_results:
        st.subheader("💡 Smart Recommendations")
        recs_data = display_smart_recs_results
        if any(recs_data.get(key) for key in ["complementary_suggestions", "similar_styles", "seasonal_advice"]):
            if recs_data.get("complementary_suggestions"): st.markdown("#### Complementary Suggestions"); st.markdown(recs_data["complementary_suggestions"])
            if recs_data.get("similar_styles"): st.markdown("#### Similar Styles You Might Like"); st.markdown(recs_data["similar_styles"])
            if recs_data.get("seasonal_advice"): st.markdown("#### Seasonal Styling Advice"); st.markdown(recs_data["seasonal_advice"])
        else:
             st.info("AI did not generate any specific smart recommendations or encountered an issue processing the request.")

    # Close MongoDB connection when app stops or session ends (optional, for explicit cleanup)
    # This might need more sophisticated handling in a multi-user environment or deployed app.
    # For local development, it's fine, but in cloud deployment, manage connections carefully.
    # st.on_session_end(close_db_connection) # This is not officially supported / might cause issues.

# No if __name__ == '__main__': here
