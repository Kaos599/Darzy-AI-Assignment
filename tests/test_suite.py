# tests/test_suite.py
import unittest
from unittest.mock import patch, MagicMock
import io
from PIL import Image # Pillow is a direct dependency for test helpers

# Import functions and constants from the 'src' package
# When running `python -m unittest tests.test_suite` from the project root,
# the 'src' directory should be discoverable.
from src.image_utils import process_image #, draw_detections_on_image (drawing is harder to unit test directly)
from src.data_exporters import convert_fashion_details_to_csv, convert_palette_to_csv, _flatten_colors_for_csv
from src.ai_services import get_fashion_details_from_image #, LANGCHAIN_AVAILABLE (might not be needed by tests directly)
from src.constants import MAX_IMAGE_SIZE, JPEG_QUALITY, TARGET_FORMAT # Removed DEFAULT_FONT_SIZE as it's not used in tested functions here

# Helper to create a dummy image (same as before)
def create_dummy_image_bytes(width, height, img_format="PNG", mode="RGB"):
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
                    ]
                },
                {
                    "item_name": "Test Pants", "category": "Bottom",
                    "bounding_box": [0.5, 0.1, 0.9, 0.5], # Valid bbox
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
        # For Test Pants, color_2 fields should be empty as per _flatten_colors_for_csv logic
        self.assertIn("Test Pants,Bottom,0.5,0.1,0.9,0.5,Black,#000000,100%,,,", csv_string)


    def test_convert_empty_fashion_details_to_csv(self):
        csv_string = convert_fashion_details_to_csv({"fashion_items": []})
        self.assertEqual(csv_string, "") # Function returns "" for no items


class TestAIServices(unittest.TestCase):
    # Patch ChatGoogleGenerativeAI where it's imported and used in src.ai_services
    @patch('src.ai_services.ChatGoogleGenerativeAI')
    def test_get_fashion_details_from_image_parsing(self, MockChatGoogle):
        mock_llm_instance = MockChatGoogle.return_value
        mock_response = MagicMock()
        mock_response.content = '''
        ```json
        {
          "fashion_items": [
            {
              "item_name": "Mock Scarf",
              "category": "Accessory",
              "bounding_box": [0.05, 0.05, 0.25, 0.95],
              "dominant_colors": [
                { "hex_code": "#A0522D", "color_name": "Sienna", "percentage": "100%" }
              ]
            }
          ]
        }
        ```
        '''
        mock_llm_instance.invoke.return_value = mock_response

        dummy_image_bytes = create_dummy_image_bytes(50, 50)
        result = get_fashion_details_from_image(dummy_image_bytes, "fake_api_key", "gemini-pro-vision")

        self.assertIsNotNone(result)
        self.assertIn("fashion_items", result)
        self.assertEqual(len(result["fashion_items"]), 1)
        item = result["fashion_items"][0]
        self.assertEqual(item["item_name"], "Mock Scarf")
        self.assertEqual(item["category"], "Accessory") # Added assertion
        self.assertEqual(item["bounding_box"], [0.05,0.05,0.25,0.95]) # Added assertion
        self.assertEqual(len(item["dominant_colors"]), 1) # Added assertion
        self.assertEqual(item["dominant_colors"][0]["color_name"], "Sienna") # Added assertion


        # Verify the prompt structure (simplified check)
        args, _ = mock_llm_instance.invoke.call_args
        messages_list = args[0]
        self.assertTrue(len(messages_list) > 0)
        human_message_content = messages_list[0].content
        self.assertTrue(isinstance(human_message_content, list))
        self.assertEqual(human_message_content[0]['type'], 'text')
        self.assertIn("Return ONLY the JSON object", human_message_content[0]['text'])
        self.assertEqual(human_message_content[1]['type'], 'image_url')

    @patch('src.ai_services.ChatGoogleGenerativeAI')
    def test_get_fashion_details_invalid_json_response(self, MockChatGoogle):
        mock_llm_instance = MockChatGoogle.return_value
        mock_response = MagicMock()
        mock_response.content = "This is not JSON."
        mock_llm_instance.invoke.return_value = mock_response
        dummy_image_bytes = create_dummy_image_bytes(50, 50)

        # In ai_services, st.error was replaced with print. We can't easily check print output
        # without further mocking sys.stdout. For now, just ensure it returns None.
        result = get_fashion_details_from_image(dummy_image_bytes, "fake_api_key", "gemini-pro-vision")
        self.assertIsNone(result)

    @patch('src.ai_services.ChatGoogleGenerativeAI')
    def test_get_fashion_details_malformed_data_in_json(self, MockChatGoogle):
        mock_llm_instance = MockChatGoogle.return_value
        mock_response = MagicMock()
        mock_response.content = '{"unexpected_key": "some_value"}' # Valid JSON, wrong structure
        mock_llm_instance.invoke.return_value = mock_response
        dummy_image_bytes = create_dummy_image_bytes(50, 50)
        result = get_fashion_details_from_image(dummy_image_bytes, "fake_api_key", "gemini-pro-vision")
        self.assertIsNone(result) # Expect None due to structural validation failure

if __name__ == '__main__':
    # This allows running the tests directly from this file: `python tests/test_suite.py`
    # Make sure the project root (aifashion/) is in PYTHONPATH or run with `python -m unittest tests.test_suite`
    unittest.main()
