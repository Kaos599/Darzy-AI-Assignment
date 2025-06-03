# test_app.py
import unittest
from unittest.mock import patch, MagicMock
import io
from PIL import Image, ImageDraw, ImageFont # Added ImageDraw, ImageFont for potential future use or if app uses them globally

# Assuming app.py contains the functions to be tested.
# We need to ensure app.py can be imported without running the Streamlit app.
# For example, Streamlit UI code should be in a main_app() function.
import app as fashion_app # Using an alias

# Helper to create a dummy image
def create_dummy_image_bytes(width, height, img_format="PNG", mode="RGB"):
    img = Image.new(mode, (width, height), color="blue")
    if mode == "RGBA" and img_format == "JPEG": # Simulate conversion scenario
        img = img.convert("RGB")

    img_byte_arr = io.BytesIO()
    try:
        if img_format == "JPEG":
            img.save(img_byte_arr, format=img_format, quality=90)
        else:
            img.save(img_byte_arr, format=img_format)
    except Exception as e:
        # Fallback for formats that might not support certain modes directly in save
        if mode != 'RGB':
            img.convert('RGB').save(img_byte_arr, format=img_format)
        else:
            raise e

    img_byte_arr = img_byte_arr.getvalue()
    return img_byte_arr

class TestImageProcessing(unittest.TestCase):
    def test_process_image_valid_png_no_resize(self):
        dummy_png_bytes = create_dummy_image_bytes(100, 100, "PNG")
        processed_bytes, new_filename, msg = fashion_app.process_image(dummy_png_bytes, "test.png")
        self.assertIsNotNone(processed_bytes)
        self.assertTrue(new_filename.endswith(f".{fashion_app.TARGET_FORMAT.lower()}"))
        pil_image = Image.open(io.BytesIO(processed_bytes))
        self.assertEqual(pil_image.format, fashion_app.TARGET_FORMAT)
        self.assertEqual(pil_image.size, (100,100)) # No resize expected

    def test_process_image_needs_resize(self):
        large_img_bytes = create_dummy_image_bytes(fashion_app.MAX_IMAGE_SIZE[0] + 100, 100, "JPEG")
        processed_bytes, _, _ = fashion_app.process_image(large_img_bytes, "large.jpg")
        self.assertIsNotNone(processed_bytes)
        pil_image = Image.open(io.BytesIO(processed_bytes))
        self.assertTrue(pil_image.size[0] <= fashion_app.MAX_IMAGE_SIZE[0] or \
                        pil_image.size[1] <= fashion_app.MAX_IMAGE_SIZE[1])

    def test_process_image_rgba_to_rgb_for_jpeg(self):
        # Ensure TARGET_FORMAT is JPEG for this test
        original_target_format = fashion_app.TARGET_FORMAT
        fashion_app.TARGET_FORMAT = "JPEG"

        rgba_img_bytes = create_dummy_image_bytes(50, 50, "PNG", mode="RGBA") # Create as PNG RGBA
        processed_bytes, _, _ = fashion_app.process_image(rgba_img_bytes, "rgba_test.png")
        self.assertIsNotNone(processed_bytes)
        pil_image = Image.open(io.BytesIO(processed_bytes))
        self.assertEqual(pil_image.format, "JPEG")
        self.assertEqual(pil_image.mode, "RGB")

        fashion_app.TARGET_FORMAT = original_target_format # Reset

    def test_process_image_invalid_file(self):
        invalid_bytes = b"this is not an image"
        processed_bytes, _, msg = fashion_app.process_image(invalid_bytes, "invalid.txt")
        self.assertIsNone(processed_bytes)
        self.assertIn("Processing failed", msg)


class TestCSVConverters(unittest.TestCase):
    def test_convert_palette_to_csv(self):
        # This tests the old `convert_palette_to_csv`
        # Renamed in a previous subtask report, ensure app.py reflects that.
        # Assuming it's now `convert_palette_to_csv` in app.py
        sample_data = {
            "colors": [
                {"color_name": "Red", "hex_code": "#FF0000", "percentage": "50%"},
                {"color_name": "Blue", "hex_code": "#0000FF", "percentage": "50%"}
            ]
        }
        # Ensure the function exists in fashion_app before calling
        self.assertTrue(hasattr(fashion_app, 'convert_palette_to_csv'), "convert_palette_to_csv function missing in app.py")
        csv_string = fashion_app.convert_palette_to_csv(sample_data)
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
        csv_string = fashion_app.convert_fashion_details_to_csv(sample_data, max_colors_per_item=2)
        self.assertTrue(csv_string.startswith("item_name,category,bbox_ymin,bbox_xmin,bbox_ymax,bbox_xmax,color_1_name,color_1_hex,color_1_percentage,color_2_name,color_2_hex,color_2_percentage"))
        self.assertIn("Test Shirt,Top,0.1,0.1,0.5,0.5,Green,#00FF00,80%,White,#FFFFFF,20%", csv_string)
        self.assertIn("Test Pants,Bottom,0.5,0.1,0.9,0.5,Black,#000000,100%,,", csv_string) # N/A becomes empty

    def test_convert_empty_fashion_details_to_csv(self):
        csv_string = fashion_app.convert_fashion_details_to_csv({"fashion_items": []})
        self.assertEqual(csv_string, "")


class TestAIInteraction(unittest.TestCase):
    @patch('app.ChatGoogleGenerativeAI') # Patch where it's looked up (in app module)
    def test_get_fashion_details_from_image_parsing(self, MockChatGoogle):
        # Configure the mock LLM
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
        result = fashion_app.get_fashion_details_from_image(dummy_image_bytes, "fake_api_key", "gemini-pro-vision")

        self.assertIsNotNone(result)
        self.assertIn("fashion_items", result)
        self.assertEqual(len(result["fashion_items"]), 1)
        item = result["fashion_items"][0]
        self.assertEqual(item["item_name"], "Mock Scarf")
        self.assertEqual(item["category"], "Accessory")
        self.assertEqual(item["bounding_box"], [0.05, 0.05, 0.25, 0.95])
        self.assertEqual(len(item["dominant_colors"]), 1)
        self.assertEqual(item["dominant_colors"][0]["color_name"], "Sienna")

        args, _ = mock_llm_instance.invoke.call_args
        messages_list = args[0]
        self.assertTrue(len(messages_list) > 0)
        human_message_content = messages_list[0].content
        self.assertTrue(isinstance(human_message_content, list))
        self.assertEqual(human_message_content[0]['type'], 'text')
        self.assertIn("Return ONLY the JSON object", human_message_content[0]['text'])
        self.assertEqual(human_message_content[1]['type'], 'image_url')


    @patch('app.ChatGoogleGenerativeAI')
    def test_get_fashion_details_invalid_json_response(self, MockChatGoogle):
        mock_llm_instance = MockChatGoogle.return_value
        mock_response = MagicMock()
        mock_response.content = "This is not JSON, just some text."
        mock_llm_instance.invoke.return_value = mock_response

        dummy_image_bytes = create_dummy_image_bytes(50, 50)
        with patch('app.st') as mock_st:
            result = fashion_app.get_fashion_details_from_image(dummy_image_bytes, "fake_api_key", "gemini-pro-vision")
            self.assertIsNone(result)
            mock_st.error.assert_any_call(unittest.mock.ANY)

    @patch('app.ChatGoogleGenerativeAI')
    def test_get_fashion_details_malformed_data_in_json(self, MockChatGoogle):
        mock_llm_instance = MockChatGoogle.return_value
        mock_response = MagicMock()
        mock_response.content = '''
        ```json
        {
          "detected_objects": [
            {"name": "Scarf"}
          ]
        }
        ```
        '''
        mock_llm_instance.invoke.return_value = mock_response
        dummy_image_bytes = create_dummy_image_bytes(50, 50)
        with patch('app.st') as mock_st:
            result = fashion_app.get_fashion_details_from_image(dummy_image_bytes, "fake_api_key", "gemini-pro-vision")
            self.assertIsNone(result)
            mock_st.error.assert_any_call(unittest.mock.ANY)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
