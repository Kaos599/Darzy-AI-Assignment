# tests/test_suite.py
import unittest
from unittest.mock import patch, MagicMock
import io
from PIL import Image # Pillow is a direct dependency for test helpers
import os
import dotenv # Added for loading .env

dotenv.load_dotenv() # Load environment variables from .env

# Import functions and constants from the 'src' package
# When running `python -m unittest tests.test_suite` from the project root,
# the 'src' directory should be discoverable.
from src.image_utils import process_image #, draw_detections_on_image (drawing is harder to unit test directly)
from src.data_exporters import convert_fashion_details_to_csv, convert_palette_to_csv, _flatten_colors_for_csv
from src.ai_services import get_fashion_details_from_image #, LANGCHAIN_AVAILABLE (might not be needed by tests directly)
from src.constants import MAX_IMAGE_SIZE, JPEG_QUALITY, TARGET_FORMAT # Removed DEFAULT_FONT_SIZE as it's not used in tested functions here

# Helper to create a dummy image (same as before)
def create_dummy_image_bytes(width, height, img_format="PNG", mode="RGB"):
    """Create a dummy image for testing purposes."""
    img = Image.new(mode, (width, height), color="blue")
    # Handle potential conversion if mode and format are incompatible for save
    if mode == "RGBA" and img_format == "JPEG":
        img = img.convert("RGB")
    elif mode == "P" and img_format == "JPEG": # Palette mode to RGB
        img = img.convert("RGB")

    img_byte_arr = io.BytesIO()
    try:
        save_params = {}
        if img_format == "JPEG": save_params['quality'] = 90 # Use a fixed quality for tests
        img.save(img_byte_arr, format=img_format, **save_params)
    except Exception: # Broad except for Pillow save issues with modes/formats
        # Fallback to RGB if specific mode save failed
        if img.mode != 'RGB': img = img.convert('RGB')
        img.save(img_byte_arr, format=img_format)

    img_byte_arr.seek(0) # Reset stream position
    return img_byte_arr.getvalue()

class TestImageProcessing(unittest.TestCase):
    def test_process_image_valid_png_no_resize(self):
        dummy_png_bytes = create_dummy_image_bytes(100, 100, "PNG")
        # Use constants imported from src.constants
        processed_bytes, new_filename, msg = process_image(dummy_png_bytes, "test.png")
        self.assertIsNotNone(processed_bytes)
        self.assertTrue(new_filename.endswith(f".{TARGET_FORMAT.lower()}"))
        pil_image = Image.open(io.BytesIO(processed_bytes))
        self.assertEqual(pil_image.format, TARGET_FORMAT)
        self.assertEqual(pil_image.size, (100,100))

    def test_process_image_needs_resize(self):
        large_img_bytes = create_dummy_image_bytes(MAX_IMAGE_SIZE[0] + 100, 100, "JPEG")
        processed_bytes, _, _ = process_image(large_img_bytes, "large.jpg")
        self.assertIsNotNone(processed_bytes)
        pil_image = Image.open(io.BytesIO(processed_bytes))
        self.assertTrue(pil_image.size[0] <= MAX_IMAGE_SIZE[0] or \
                        pil_image.size[1] <= MAX_IMAGE_SIZE[1])

    def test_process_image_rgba_to_rgb_for_jpeg(self):
        # Temporarily set TARGET_FORMAT for this test if it's different
        # This assumes process_image uses the global TARGET_FORMAT from src.constants
        # If TARGET_FORMAT is fixed to JPEG, this test is simpler.
        # Let's assume TARGET_FORMAT is JPEG as per current constants.py
        if TARGET_FORMAT != "JPEG":
            self.skipTest("Skipping RGBA to RGB test as TARGET_FORMAT is not JPEG")

        rgba_img_bytes = create_dummy_image_bytes(50, 50, "PNG", mode="RGBA")
        processed_bytes, _, _ = process_image(rgba_img_bytes, "rgba_test.png")
        self.assertIsNotNone(processed_bytes)
        pil_image = Image.open(io.BytesIO(processed_bytes))
        self.assertEqual(pil_image.format, "JPEG")
        self.assertEqual(pil_image.mode, "RGB")

    def test_process_image_invalid_file(self):
        invalid_bytes = b"this is not an image"
        processed_bytes, _, msg = process_image(invalid_bytes, "invalid.txt")
        self.assertIsNone(processed_bytes)
        self.assertIn("Processing failed", msg)


class TestCSVConverters(unittest.TestCase):
    def test_convert_palette_to_csv(self):
        # This tests the deprecated `convert_palette_to_csv`
        sample_data = {
            "colors": [
                {"color_name": "Red", "hex_code": "#FF0000", "percentage": "50%"},
                {"color_name": "Blue", "hex_code": "#0000FF", "percentage": "50%"}
            ]
        }
        csv_string = convert_palette_to_csv(sample_data)
        self.assertIn("color_name,hex_code,percentage", csv_string)
        self.assertIn("Red,#FF0000,50%", csv_string)
        self.assertIn("Blue,#0000FF,50%", csv_string)

    def test_convert_fashion_details_to_csv(self):
        sample_data = {
            "fashion_items": [
                {
                    "item_name": "Test Shirt", "category": "Top",
                    "bounding_box": [0.1, 0.1, 0.5, 0.5],
                    "dominant_colors": [
                        {"color_name": "Green", "hex_code": "#00FF00", "percentage": "80%"},
                        {"color_name": "White", "hex_code": "#FFFFFF", "percentage": "20%"}
                    ],
                    "fabric_type": "Cotton",
                    "fabric_confidence_score": 0.9
                },
                {
                    "item_name": "Test Pants", "category": "Bottom",
                    "bounding_box": [0.5, 0.1, 0.9, 0.5], # Valid bbox
                    "dominant_colors": [
                        {"color_name": "Black", "hex_code": "#000000", "percentage": "100%"}
                    ],
                    "fabric_type": "Denim",
                    "fabric_confidence_score": 0.85
                }
            ]
        }
        csv_string = convert_fashion_details_to_csv(sample_data, max_colors_per_item=2)
        expected_header = "item_name,category,fabric_type,fabric_confidence_score,bbox_ymin,bbox_xmin,bbox_ymax,bbox_xmax,color_1_name,color_1_hex,color_1_percentage,color_2_name,color_2_hex,color_2_percentage"
        self.assertTrue(csv_string.startswith(expected_header))
        self.assertIn("Test Shirt,Top,Cotton,0.9,0.1,0.1,0.5,0.5,Green,#00FF00,80%,White,#FFFFFF,20%", csv_string)
        # For Test Pants, color_2 fields should be empty as per _flatten_colors_for_csv logic
        self.assertIn("Test Pants,Bottom,Denim,0.85,0.5,0.1,0.9,0.5,Black,#000000,100%,,,", csv_string)


    def test_convert_empty_fashion_details_to_csv(self):
        csv_string = convert_fashion_details_to_csv({"fashion_items": []})
        self.assertEqual(csv_string, "") # Function returns "" for no items


class TestAIServicesLive(unittest.TestCase):
    def setUp(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.model_name = os.getenv("GEMINI_MODEL_NAME", "gemini-pro-vision")
        self.dummy_image_bytes = create_dummy_image_bytes(200, 200, "PNG")

    @unittest.skipUnless(os.getenv("GEMINI_API_KEY"), "GEMINI_API_KEY not set, skipping live AI service test.")
    def test_get_fashion_details_from_image_live_call(self):
        print(f"\nRunning live AI service test with model: {self.model_name}")
        result = get_fashion_details_from_image(self.dummy_image_bytes, self.api_key, self.model_name)

        self.assertIsNotNone(result, "Expected a result from Gemini API, got None.")
        self.assertIn("fashion_items", result, "Result missing 'fashion_items' key.")
        self.assertIsInstance(result["fashion_items"], list, "'fashion_items' should be a list.")

        # If items are detected, validate their basic structure (now including fabric)
        if result["fashion_items"]:
            item = result["fashion_items"][0]
            self.assertIn("item_name", item)
            self.assertIn("category", item)
            self.assertIn("bounding_box", item)
            self.assertIn("dominant_colors", item)
            self.assertIn("fabric_type", item)
            self.assertIn("fabric_confidence_score", item)
            self.assertIsInstance(item["bounding_box"], list)
            self.assertEqual(len(item["bounding_box"]), 4)
            self.assertIsInstance(item["dominant_colors"], list)
            self.assertIsInstance(item["fabric_confidence_score"], (float, int))
            self.assertTrue(0.0 <= item["fabric_confidence_score"] <= 1.0)
            print("INFO: Live AI service test passed with detected items including fabric details.")
        else:
            print("INFO: Live AI service test passed with no items detected (expected for a plain blue image). If this was not expected, check the dummy image). ")

    @unittest.skipUnless(os.getenv("GEMINI_API_KEY"), "GEMINI_API_KEY not set, skipping live AI service test.")
    def test_end_to_end_fabric_analysis_silk_shirt(self):
        print("\nRunning end-to-end fabric analysis test with silk_shirt.png...")
        image_path = "assets/silk_shirt.png"
        try:
            with open(image_path, "rb") as f:
                image_bytes = f.read()
        except FileNotFoundError:
            self.fail(f"Test image not found at {image_path}. Please ensure it exists.")

        # 1. Process image
        processed_bytes, processed_filename, msg = process_image(image_bytes, os.path.basename(image_path))
        self.assertIsNotNone(processed_bytes, f"Image processing failed for {image_path}: {msg}")

        # 2. Get AI analysis
        analysis_data = get_fashion_details_from_image(processed_bytes, self.api_key, self.model_name)
        self.assertIsNotNone(analysis_data, "AI analysis returned None.")
        self.assertIn("fashion_items", analysis_data)
        self.assertIsInstance(analysis_data["fashion_items"], list)
        self.assertTrue(len(analysis_data["fashion_items"]) > 0, "No fashion items detected for silk_shirt.png.")

        # 3. Validate fabric analysis for at least one item
        found_fabric_details = False
        for item in analysis_data["fashion_items"]:
            self.assertIn("item_name", item)
            self.assertIn("category", item)
            self.assertIn("bounding_box", item)
            self.assertIn("dominant_colors", item)
            self.assertIn("fabric_type", item)
            self.assertIn("fabric_confidence_score", item)

            if item.get("fabric_type") and isinstance(item.get("fabric_confidence_score"), (float, int)):
                self.assertTrue(0.0 <= item["fabric_confidence_score"] <= 1.0)
                print(f"  Detected fabric: {item["fabric_type"]} with confidence {item["fabric_confidence_score"]:.2f} for {item["item_name"]}")
                found_fabric_details = True
        self.assertTrue(found_fabric_details, "No item in silk_shirt.png analysis contained valid fabric details.")

        # 4. Validate CSV export includes new fields
        csv_string = convert_fashion_details_to_csv(analysis_data)
        self.assertIn("fabric_type", csv_string)
        self.assertIn("fabric_confidence_score", csv_string)
        print("  CSV export verified to include fabric details.")
        print("End-to-end fabric analysis test (silk_shirt.png) passed.")

if __name__ == '__main__':
    # This allows running the tests directly from this file: `python tests/test_suite.py`
    # Make sure the project root (aifashion/) is in PYTHONPATH or run with `python -m unittest tests.test_suite`
    unittest.main()
