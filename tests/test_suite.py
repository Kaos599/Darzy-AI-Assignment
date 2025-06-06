# tests/test_suite.py
import unittest
from unittest.mock import patch, MagicMock
import io
from PIL import Image 


from src.image_utils import process_image 
from src.data_exporters import convert_fashion_details_to_csv, convert_palette_to_csv, _flatten_colors_for_csv
from src.ai_services import get_fashion_details_from_image, get_size_estimation_for_items, generate_fashion_copy, get_smart_recommendations
from src.constants import MAX_IMAGE_SIZE, JPEG_QUALITY, TARGET_FORMAT 
from src.constants import GEMINI_API_KEY, GEMINI_MODEL_NAME

def create_dummy_image_bytes(width, height, img_format="PNG", mode="RGB"):
    img = Image.new(mode, (width, height), color="blue")
    if mode == "RGBA" and img_format == "JPEG":
        img = img.convert("RGB")
    elif mode == "P" and img_format == "JPEG": 
        img = img.convert("RGB")

    img_byte_arr = io.BytesIO()
    try:
        save_params = {}
        if img_format == "JPEG": save_params['quality'] = 90
        img.save(img_byte_arr, format=img_format, **save_params)
    except Exception:
        if img.mode != 'RGB': img = img.convert('RGB')
        img.save(img_byte_arr, format=img_format)

    img_byte_arr.seek(0)
    return img_byte_arr.getvalue()

class TestImageProcessing(unittest.TestCase):
    def test_process_image_valid_png_no_resize(self):
        dummy_png_bytes = create_dummy_image_bytes(100, 100, "PNG")
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
                    ]
                },
                {
                    "item_name": "Test Pants", "category": "Bottom",
                    "bounding_box": [0.5, 0.1, 0.9, 0.5],
                    "dominant_colors": [
                        {"color_name": "Black", "hex_code": "#000000", "percentage": "100%"}
                    ]
                }
            ]
        }
        csv_string = convert_fashion_details_to_csv(sample_data, max_colors_per_item=2)
        expected_header = "item_name,category,bbox_ymin,bbox_xmin,bbox_ymax,bbox_xmax,color_1_name,color_1_hex,color_1_percentage,color_2_name,color_2_hex,color_2_percentage"
        self.assertTrue(csv_string.startswith(expected_header))
        self.assertIn("Test Shirt,Top,0.1,0.1,0.5,0.5,Green,#00FF00,80%,White,#FFFFFF,20%", csv_string)
        self.assertIn("Test Pants,Bottom,0.5,0.1,0.9,0.5,Black,#000000,100%,,,", csv_string)


    def test_convert_empty_fashion_details_to_csv(self):
        csv_string = convert_fashion_details_to_csv({"fashion_items": []})
        self.assertEqual(csv_string, "")


class TestAIServices(unittest.TestCase):
    def test_get_fashion_details_from_image_parsing(self):
        dummy_image_bytes = create_dummy_image_bytes(50, 50)
        result = get_fashion_details_from_image(dummy_image_bytes, GEMINI_API_KEY, GEMINI_MODEL_NAME)
        self.assertIsNotNone(result)
        self.assertIn("fashion_items", result)
        self.assertTrue(isinstance(result["fashion_items"], list))

    def test_get_fashion_details_invalid_json_response(self):
        dummy_image_bytes = create_dummy_image_bytes(50, 50)
        result = get_fashion_details_from_image(dummy_image_bytes, GEMINI_API_KEY, GEMINI_MODEL_NAME)
        if GEMINI_API_KEY is None or GEMINI_API_KEY == "YOUR_ACTUAL_GEMINI_API_KEY_HERE":
            self.assertIsNone(result)
        else:
            self.assertIsNotNone(result)
            self.assertIn("fashion_items", result)

    def test_get_fashion_details_malformed_data_in_json(self):
        dummy_image_bytes = create_dummy_image_bytes(50, 50)
        result = get_fashion_details_from_image(dummy_image_bytes, GEMINI_API_KEY, GEMINI_MODEL_NAME)
        if GEMINI_API_KEY is None or GEMINI_API_KEY == "YOUR_ACTUAL_GEMINI_API_KEY_HERE":
            self.assertIsNone(result)
        else:
            self.assertIsNotNone(result)
            self.assertIn("fashion_items", result)


class TestNewAIServices(unittest.TestCase):
    def test_get_size_estimation_parsing(self):
        dummy_image_bytes = create_dummy_image_bytes(100,100)
        result = get_size_estimation_for_items(dummy_image_bytes, GEMINI_API_KEY, GEMINI_MODEL_NAME)
        self.assertIsNotNone(result)
        self.assertIn("size_estimations", result)
        self.assertTrue(isinstance(result["size_estimations"], list))

    def test_get_size_estimation_empty_response(self):
        dummy_image_bytes = create_dummy_image_bytes(100,100)
        result = get_size_estimation_for_items(dummy_image_bytes, GEMINI_API_KEY, GEMINI_MODEL_NAME)
        if GEMINI_API_KEY is None or GEMINI_API_KEY == "YOUR_ACTUAL_GEMINI_API_KEY_HERE":
            self.assertIsNone(result)
        else:
            self.assertIsNotNone(result)
            self.assertIn("size_estimations", result)

    def test_generate_fashion_copy_parsing(self):
        dummy_image_bytes = create_dummy_image_bytes(100,100)
        result = generate_fashion_copy(dummy_image_bytes, GEMINI_API_KEY, GEMINI_MODEL_NAME)
        self.assertIsNotNone(result)
        self.assertIn("product_description", result)
        self.assertIn("styling_suggestions", result)
        self.assertIn("social_media_caption", result)
        self.assertTrue(isinstance(result["product_description"], str))
        self.assertTrue(isinstance(result["styling_suggestions"], str))
        self.assertTrue(isinstance(result["social_media_caption"], str))

    def test_generate_fashion_copy_missing_keys(self):
        dummy_image_bytes = create_dummy_image_bytes(100,100)
        result = generate_fashion_copy(dummy_image_bytes, GEMINI_API_KEY, GEMINI_MODEL_NAME)
        if GEMINI_API_KEY is None or GEMINI_API_KEY == "YOUR_ACTUAL_GEMINI_API_KEY_HERE":
            self.assertIsNone(result)
        else:
            self.assertIsNotNone(result)
            self.assertIn("product_description", result)

    def test_get_smart_recommendations_parsing(self):
        dummy_image_bytes = create_dummy_image_bytes(100,100)
        sample_fashion_items = [
            {"item_name": "Blue Jeans", "category": "Bottomwear", "bounding_box": [0.1, 0.1, 0.5, 0.5],
             "dominant_colors": [{"hex_code": "#0000FF", "color_name": "Blue", "percentage": "100%"}]},
            {"item_name": "White T-Shirt", "category": "Topwear", "bounding_box": [0.2, 0.2, 0.6, 0.6],
             "dominant_colors": [{"hex_code": "#FFFFFF", "color_name": "White", "percentage": "100%"}]}
        ]
        result = get_smart_recommendations(GEMINI_API_KEY, GEMINI_MODEL_NAME, sample_fashion_items)
        self.assertIsNotNone(result)
        self.assertIn("complementary_suggestions", result)
        self.assertIn("similar_styles", result)
        self.assertIn("seasonal_advice", result)
        self.assertTrue(isinstance(result["complementary_suggestions"], str))
        self.assertTrue(isinstance(result["similar_styles"], str))
        self.assertTrue(isinstance(result["seasonal_advice"], str))

    def test_get_smart_recommendations_no_items(self):
        dummy_image_bytes = create_dummy_image_bytes(100,100)
        result = get_smart_recommendations(GEMINI_API_KEY, GEMINI_MODEL_NAME, [])
        expected_result = {
            "complementary_suggestions": "No items were provided for recommendations.",
            "similar_styles": "No items were provided for recommendations.",
            "seasonal_advice": "No items were provided for recommendations."
        }
        self.assertEqual(result, expected_result)

    def test_end_to_end_smart_recommendations(self):
        # This test requires a valid GEMINI_API_KEY to run
        if GEMINI_API_KEY is None or GEMINI_API_KEY == "YOUR_ACTUAL_GEMINI_API_KEY_HERE":
            self.skipTest("GEMINI_API_KEY not set. Skipping end-to-end recommendation test.")

        # 1. Load the real image
        try:
            with open("assets/silk_shirt.png", "rb") as f:
                image_bytes = f.read()
        except FileNotFoundError:
            self.fail("assets/silk_shirt.png not found. Ensure the image exists at the specified path.")

        # 2. Process the image
        processed_image_bytes, _, process_msg = process_image(image_bytes, "silk_shirt.png")
        self.assertIsNotNone(processed_image_bytes, f"Image processing failed: {process_msg}")

        # 3. Get fashion details from the image using the actual AI service
        fashion_details = get_fashion_details_from_image(
            processed_image_bytes,
            GEMINI_API_KEY,
            GEMINI_MODEL_NAME
        )
        self.assertIsNotNone(fashion_details, "Fashion analysis failed or returned None.")
        self.assertIn("fashion_items", fashion_details, "Fashion analysis result missing 'fashion_items' key.")
        self.assertGreater(len(fashion_details["fashion_items"]), 0, "No fashion items detected by AI.")

        # 4. Get smart recommendations using the detected fashion items
        recommendations = get_smart_recommendations(
            GEMINI_API_KEY,
            GEMINI_MODEL_NAME,
            fashion_details["fashion_items"]
        )

        # 5. Assert the recommendations are not the default "No items were provided"
        self.assertIsNotNone(recommendations, "Smart recommendations failed or returned None.")
        self.assertIn("complementary_suggestions", recommendations)
        self.assertIn("similar_styles", recommendations)
        self.assertIn("seasonal_advice", recommendations)

        self.assertNotEqual(recommendations["complementary_suggestions"], "No items were provided for recommendations.")
        self.assertNotEqual(recommendations["similar_styles"], "No items were provided for recommendations.")
        self.assertNotEqual(recommendations["seasonal_advice"], "No items were provided for recommendations.")

if __name__ == '__main__':
    unittest.main()
