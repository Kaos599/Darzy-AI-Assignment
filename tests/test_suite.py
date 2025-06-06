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
import src.database_manager as db_manager # Import the module directly

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

class TestDatabaseManager(unittest.TestCase):

    @patch('src.database_manager.MongoClient')
    @patch('src.database_manager.st') # Patch streamlit
    @patch('src.database_manager.datetime') # Patch datetime
    @patch('src.database_manager.uuid') # Patch uuid
    def setUp(self, mock_uuid, mock_datetime, mock_st, mock_MongoClient):
        # Directly patch module-level variables for database_manager
        self.original_mongo_uri = db_manager.MONGO_URI
        self.original_mongo_db_name = db_manager.MONGO_DB_NAME
        self.original_image_collection = db_manager.MONGO_IMAGE_COLLECTION
        self.original_analysis_collection = db_manager.MONGO_ANALYSIS_COLLECTION
        self.original_recommendation_collection = db_manager.MONGO_RECOMMENDATION_COLLECTION
        self.original_size_collection = db_manager.MONGO_SIZE_COLLECTION
        self.original_copy_collection = db_manager.MONGO_COPY_COLLECTION

        db_manager.MONGO_URI = "mongodb://mock_uri"
        db_manager.MONGO_DB_NAME = "test_db"
        db_manager.MONGO_IMAGE_COLLECTION = "test_images"
        db_manager.MONGO_ANALYSIS_COLLECTION = "test_analysis"
        db_manager.MONGO_RECOMMENDATION_COLLECTION = "test_recommendations"
        db_manager.MONGO_SIZE_COLLECTION = "test_size"
        db_manager.MONGO_COPY_COLLECTION = "test_copy"

        # Mock datetime.now()
        self.mock_now = MagicMock()
        self.mock_now.return_value = "2023-01-01T12:00:00Z"
        mock_datetime.now.return_value = self.mock_now.return_value

        # Mock uuid.uuid4()
        mock_uuid.uuid4.return_value = "mock_uuid_value"

        # Mock MongoClient and its methods
        self.mock_client_instance = mock_MongoClient.return_value
        # Patch admin.command directly on the mock_client_instance
        self.mock_client_instance.admin.command = MagicMock()

        self.mock_db = self.mock_client_instance["test_db"]

        self.mock_images_collection = self.mock_db["test_images"]
        self.mock_analysis_collection = self.mock_db["test_analysis"]
        self.mock_recommendation_collection = self.mock_db["test_recommendations"]
        self.mock_size_collection = self.mock_db["test_size"]
        self.mock_copy_collection = self.mock_db["test_copy"]

        # Ensure _client in database_manager is None before each test
        self.original_db_manager_client = None
        if hasattr(db_manager, '_client'):
            self.original_db_manager_client = db_manager._client
        db_manager._client = None # Force re-initialization of MongoClient

    @patch('src.database_manager.MongoClient')
    @patch('src.database_manager.st') # Patch streamlit
    def tearDown(self, mock_st, mock_MongoClient):
        # Restore original module-level variables
        db_manager.MONGO_URI = self.original_mongo_uri
        db_manager.MONGO_DB_NAME = self.original_mongo_db_name
        db_manager.MONGO_IMAGE_COLLECTION = self.original_image_collection
        db_manager.MONGO_ANALYSIS_COLLECTION = self.original_analysis_collection
        db_manager.MONGO_RECOMMENDATION_COLLECTION = self.original_recommendation_collection
        db_manager.MONGO_SIZE_COLLECTION = self.original_size_collection
        db_manager.MONGO_COPY_COLLECTION = self.original_copy_collection

        # Restore original _client state in database_manager
        if self.original_db_manager_client is not None:
            db_manager._client = self.original_db_manager_client
        # Ensure the client is truly closed for the next test run if it was set up
        if db_manager._client:
            db_manager._client.close()
            db_manager._client = None

    @patch('src.database_manager.MongoClient')
    @patch('src.database_manager.st') # Patch streamlit
    def test_get_database_connection(self, mock_st, mock_MongoClient):
        # MONGO_URI is now directly patched in setUp
        db = db_manager.get_database()
        self.assertIsNotNone(db)
        mock_MongoClient.assert_called_once_with(db_manager.MONGO_URI)
        # Assert that admin.command was called on the instance returned by MongoClient
        self.mock_client_instance.admin.command.assert_called_once_with('ismaster')
        self.assertEqual(db.name, db_manager.MONGO_DB_NAME)

    @patch('src.database_manager.MongoClient')
    @patch('src.database_manager.st') # Patch streamlit
    @patch('src.database_manager.datetime') # Patch datetime
    def test_save_image_metadata(self, mock_datetime, mock_st, mock_MongoClient):
        # MONGO_URI, MONGO_DB_NAME, etc. are now directly patched in setUp
        # Configure mock_datetime.now()
        mock_datetime.now.return_value = self.mock_now.return_value

        # Mock the collection methods
        mock_client = mock_MongoClient.return_value
        mock_db = mock_client[db_manager.MONGO_DB_NAME]
        mock_images_collection = mock_db[db_manager.MONGO_IMAGE_COLLECTION]
        mock_images_collection.replace_one.return_value = MagicMock(matched_count=0, upserted_id="new_id")

        image_id = "img123"
        file_name = "test_image.jpg"
        image_b64 = "base64string"

        db_manager.save_image_metadata(image_id, file_name, image_b64)

        mock_images_collection.replace_one.assert_called_once_with(
            {"_id": image_id},
            {
                "_id": image_id,
                "original_filename": file_name,
                "processed_image_b64": image_b64,
                "timestamp": self.mock_now.return_value # Use the mocked datetime
            },
            upsert=True
        )

    @patch('src.database_manager.MongoClient')
    @patch('src.database_manager.st') # Patch streamlit
    def test_load_image_metadata(self, mock_st, mock_MongoClient):
        # MONGO_URI, MONGO_DB_NAME, etc. are now directly patched in setUp
        mock_client = mock_MongoClient.return_value
        mock_db = mock_client[db_manager.MONGO_DB_NAME]
        mock_images_collection = mock_db[db_manager.MONGO_IMAGE_COLLECTION]
        expected_data = {"_id": "img123", "original_filename": "test.jpg", "processed_image_b64": "abc", "timestamp": "2023-01-01T12:00:00Z"}
        mock_images_collection.find_one.return_value = expected_data

        result = db_manager.load_image_metadata("img123")

        mock_images_collection.find_one.assert_called_once_with({"_id": "img123"})
        self.assertEqual(result, expected_data)

    @patch('src.database_manager.MongoClient')
    @patch('src.database_manager.st') # Patch streamlit
    def test_load_image_metadata_not_found(self, mock_st, mock_MongoClient):
        # MONGO_URI, MONGO_DB_NAME, etc. are now directly patched in setUp
        mock_client = mock_MongoClient.return_value
        mock_db = mock_client[db_manager.MONGO_DB_NAME]
        mock_images_collection = mock_db[db_manager.MONGO_IMAGE_COLLECTION]
        mock_images_collection.find_one.return_value = None

        result = db_manager.load_image_metadata("non_existent_id")

        self.assertIsNone(result)

    @patch('src.database_manager.MongoClient')
    @patch('src.database_manager.st') # Patch streamlit
    @patch('src.database_manager.datetime') # Patch datetime
    @patch('src.database_manager.uuid') # Patch uuid
    def test_save_analysis_results(self, mock_uuid, mock_datetime, mock_st, mock_MongoClient):
        # MONGO_URI, MONGO_DB_NAME, etc. are now directly patched in setUp
        # Configure mock_datetime.now()
        mock_datetime.now.return_value = self.mock_now.return_value

        # Configure mock_uuid.uuid4()
        mock_uuid.uuid4.return_value = "mock_uuid_value"

        mock_client = mock_MongoClient.return_value
        mock_db = mock_client[db_manager.MONGO_DB_NAME]

        # Test saving fashion_details
        analysis_data_fd = {"fashion_items": [{"item": "shirt"}]}
        mock_db[db_manager.MONGO_ANALYSIS_COLLECTION].replace_one.return_value = MagicMock(matched_count=0, upserted_id="new_id_fd")
        db_manager.save_analysis_results("img1", analysis_data_fd, "fashion_details")
        mock_db[db_manager.MONGO_ANALYSIS_COLLECTION].replace_one.assert_called_once_with(
            {"_id": "img1"},
            {
                "_id": "img1",
                "fashion_items": [{"item": "shirt"}],
                "timestamp": self.mock_now.return_value,
                "image_id": "img1"
            },
            upsert=True
        )
        mock_db[db_manager.MONGO_ANALYSIS_COLLECTION].replace_one.reset_mock() # Reset mock for next assertion

        # Test saving size_estimation
        analysis_data_se = {"size_estimations": [{"item": "pants"}]}
        mock_db[db_manager.MONGO_SIZE_COLLECTION].replace_one.return_value = MagicMock(matched_count=0, upserted_id="new_id_se")
        db_manager.save_analysis_results("img1", analysis_data_se, "size_estimation")
        mock_db[db_manager.MONGO_SIZE_COLLECTION].replace_one.assert_called_once_with(
            {"_id": "img1"},
            {
                "_id": "img1",
                "size_estimations": [{"item": "pants"}],
                "timestamp": self.mock_now.return_value,
                "image_id": "img1"
            },
            upsert=True
        )
        mock_db[db_manager.MONGO_SIZE_COLLECTION].replace_one.reset_mock()

        # Test saving fashion_copy
        analysis_data_fc = {"product_description": "nice shirt"}
        mock_db[db_manager.MONGO_COPY_COLLECTION].replace_one.return_value = MagicMock(matched_count=0, upserted_id="new_id_fc")
        db_manager.save_analysis_results("img1", analysis_data_fc, "fashion_copy")
        mock_db[db_manager.MONGO_COPY_COLLECTION].replace_one.assert_called_once_with(
            {"_id": "img1"},
            {
                "_id": "img1",
                "product_description": "nice shirt",
                "timestamp": self.mock_now.return_value,
                "image_id": "img1"
            },
            upsert=True
        )
        mock_db[db_manager.MONGO_COPY_COLLECTION].replace_one.reset_mock()

        # Test saving smart_recommendations
        analysis_data_sr = {"complementary_suggestions": "shoes"}
        mock_db[db_manager.MONGO_RECOMMENDATION_COLLECTION].replace_one.return_value = MagicMock(matched_count=0, upserted_id="new_id_sr")
        db_manager.save_analysis_results("img1", analysis_data_sr, "smart_recommendations")
        mock_db[db_manager.MONGO_RECOMMENDATION_COLLECTION].replace_one.assert_called_once_with(
            {"_id": "img1"},
            {
                "_id": "img1",
                "complementary_suggestions": "shoes",
                "timestamp": self.mock_now.return_value,
                "image_id": "img1"
            },
            upsert=True
        )
        mock_db[db_manager.MONGO_RECOMMENDATION_COLLECTION].replace_one.reset_mock()

    @patch('src.database_manager.MongoClient')
    @patch('src.database_manager.st') # Patch streamlit
    def test_load_analysis_results(self, mock_st, mock_MongoClient):
        # MONGO_URI, MONGO_DB_NAME, etc. are now directly patched in setUp
        mock_client = mock_MongoClient.return_value
        mock_db = mock_client[db_manager.MONGO_DB_NAME]

        # Test loading fashion_details
        expected_fd_data = {"fashion_items": [{"item": "shirt"}]}
        mock_db[db_manager.MONGO_ANALYSIS_COLLECTION].find_one.return_value = {"_id": "img1", "image_id": "img1", "type": "fashion_details", "data": expected_fd_data}
        result_fd = db_manager.load_analysis_results("img1", "fashion_details")
        self.assertEqual(result_fd, expected_fd_data)
        mock_db[db_manager.MONGO_ANALYSIS_COLLECTION].find_one.assert_called_once_with({"image_id": "img1"})
        mock_db[db_manager.MONGO_ANALYSIS_COLLECTION].find_one.reset_mock()

        # Test loading size_estimation
        expected_se_data = {"size_estimations": [{"item": "pants"}]}
        mock_db[db_manager.MONGO_SIZE_COLLECTION].find_one.return_value = {"_id": "img1", "image_id": "img1", "type": "size_estimation", "data": expected_se_data}
        result_se = db_manager.load_analysis_results("img1", "size_estimation")
        self.assertEqual(result_se, expected_se_data)
        mock_db[db_manager.MONGO_SIZE_COLLECTION].find_one.assert_called_once_with({"image_id": "img1"})
        mock_db[db_manager.MONGO_SIZE_COLLECTION].find_one.reset_mock()

        # Test loading fashion_copy
        expected_fc_data = {"product_description": "nice shirt"}
        mock_db[db_manager.MONGO_COPY_COLLECTION].find_one.return_value = {"_id": "img1", "image_id": "img1", "type": "fashion_copy", "data": expected_fc_data}
        result_fc = db_manager.load_analysis_results("img1", "fashion_copy")
        self.assertEqual(result_fc, expected_fc_data)
        mock_db[db_manager.MONGO_COPY_COLLECTION].find_one.assert_called_once_with({"image_id": "img1"})
        mock_db[db_manager.MONGO_COPY_COLLECTION].find_one.reset_mock()

        # Test loading smart_recommendations
        expected_sr_data = {"complementary_suggestions": "shoes"}
        mock_db[db_manager.MONGO_RECOMMENDATION_COLLECTION].find_one.return_value = {"_id": "img1", "image_id": "img1", "type": "smart_recommendations", "data": expected_sr_data}
        result_sr = db_manager.load_analysis_results("img1", "smart_recommendations")
        self.assertEqual(result_sr, expected_sr_data)
        mock_db[db_manager.MONGO_RECOMMENDATION_COLLECTION].find_one.assert_called_once_with({"image_id": "img1"})
        mock_db[db_manager.MONGO_RECOMMENDATION_COLLECTION].find_one.reset_mock()

    @patch('src.database_manager.MongoClient')
    @patch('src.database_manager.st') # Patch streamlit
    def test_load_analysis_results_not_found(self, mock_st, mock_MongoClient):
        # MONGO_URI, MONGO_DB_NAME, etc. are now directly patched in setUp
        mock_client = mock_MongoClient.return_value
        mock_db = mock_client[db_manager.MONGO_DB_NAME]
        mock_db[db_manager.MONGO_ANALYSIS_COLLECTION].find_one.return_value = None

        result = db_manager.load_analysis_results("non_existent_id", "fashion_details")

        self.assertIsNone(result)
