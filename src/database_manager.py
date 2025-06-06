import os
from dotenv import load_dotenv
from pymongo import MongoClient, ASCENDING
from pymongo.errors import ConnectionFailure, PyMongoError
import streamlit as st

# Load environment variables
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME")
MONGO_IMAGE_COLLECTION = os.getenv("MONGO_IMAGE_COLLECTION")
MONGO_ANALYSIS_COLLECTION = os.getenv("MONGO_ANALYSIS_COLLECTION")
MONGO_RECOMMENDATION_COLLECTION = os.getenv("MONGO_RECOMMENDATION_COLLECTION")
MONGO_SIZE_COLLECTION = os.getenv("MONGO_SIZE_COLLECTION")
MONGO_COPY_COLLECTION = os.getenv("MONGO_COPY_COLLECTION", "fashion_copy") # Add a default if not in .env

_client = None

def get_database():
    """Returns the MongoDB database client."""
    global _client
    if _client is None:
        if not MONGO_URI:
            st.error("MongoDB URI is not set in environment variables.")
            return None
        try:
            _client = MongoClient(MONGO_URI)
            # The ismaster command is cheap and does not require auth.
            _client.admin.command('ismaster')
            st.success("Successfully connected to MongoDB!")
        except ConnectionFailure as e:
            st.error(f"Could not connect to MongoDB: {e}")
            _client = None # Ensure client is None if connection fails
            return None
        except Exception as e:
            st.error(f"An unexpected error occurred while connecting to MongoDB: {e}")
            _client = None
            return None

    db = _client[MONGO_DB_NAME]
    if db is not None: # Changed from 'if db:' to 'if db is not None:'
        return db
    else:
        st.error("MongoDB database object is not initialized.")
        return None

def get_collection(collection_name: str):
    """Returns a specific MongoDB collection."""
    db = get_database()
    if db is not None:
        return db[collection_name]
    return None

def save_image_metadata(image_id: str, filename: str, processed_bytes_b64: str) -> bool:
    """Saves image metadata to the images collection."""
    collection = get_collection(MONGO_IMAGE_COLLECTION)
    if collection is not None:
        try:
            image_doc = {
                "_id": image_id,
                "original_filename": filename,
                "processed_image_b64": processed_bytes_b64,
                "timestamp": datetime.now().isoformat()
            }
            collection.replace_one({"_id": image_id}, image_doc, upsert=True)
            return True
        except PyMongoError as e:
            st.error(f"Error saving image metadata to DB: {e}")
            return False
    return False

def load_image_metadata(image_id: str) -> dict | None:
    """Loads image metadata from the images collection."""
    collection = get_collection(MONGO_IMAGE_COLLECTION)
    if collection is not None:
        try:
            return collection.find_one({"_id": image_id})
        except PyMongoError as e:
            st.error(f"Error loading image metadata from DB: {e}")
            return None
    return None

def save_analysis_results(image_id: str, data: dict, analysis_type: str) -> bool:
    """Saves AI analysis results to the appropriate collection."""
    collection_map = {
        "fashion_details": MONGO_ANALYSIS_COLLECTION,
        "size_estimation": MONGO_SIZE_COLLECTION,
        "fashion_copy": MONGO_COPY_COLLECTION,
        "smart_recommendations": MONGO_RECOMMENDATION_COLLECTION,
    }
    col_name = collection_map.get(analysis_type)
    if not col_name:
        st.error(f"Unknown analysis type: {analysis_type}")
        return False

    collection = get_collection(col_name)
    if collection is not None:
        try:
            # Ensure _id is always the image_id for consistency in analysis results
            data_to_save = data.copy() # Create a copy to avoid modifying the original dict passed in
            data_to_save["_id"] = image_id
            data_to_save["timestamp"] = datetime.now().isoformat()
            data_to_save["image_id"] = image_id # Redundant but ensures field exists

            collection.replace_one({"_id": image_id}, data_to_save, upsert=True)
            return True
        except PyMongoError as e:
            st.error(f"Error saving {analysis_type} results to DB: {e}")
            return False
    return False

def load_analysis_results(image_id: str, analysis_type: str) -> dict | None:
    """Loads AI analysis results from the appropriate collection."""
    collection_map = {
        "fashion_details": MONGO_ANALYSIS_COLLECTION,
        "size_estimation": MONGO_SIZE_COLLECTION,
        "fashion_copy": MONGO_COPY_COLLECTION,
        "smart_recommendations": MONGO_RECOMMENDATION_COLLECTION,
    }
    col_name = collection_map.get(analysis_type)
    if not col_name:
        st.error(f"Unknown analysis type: {analysis_type}")
        return None

    collection = get_collection(col_name)
    if collection is not None:
        try:
            # Find document by image_id, assuming one result per image for each analysis type
            result = collection.find_one({"image_id": image_id})
            # Return the entire document, as the saved 'data' is the top-level dictionary
            return result
        except PyMongoError as e:
            st.error(f"Error loading {analysis_type} results from DB: {e}")
            return None
    return None

def close_db_connection():
    """Closes the MongoDB connection."""
    global _client
    if _client:
        _client.close()
        _client = None
        st.info("MongoDB connection closed.")

from datetime import datetime
import uuid # For generating unique IDs if needed 