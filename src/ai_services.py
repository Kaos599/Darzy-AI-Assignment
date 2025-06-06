# src/ai_services.py

import json
import base64
import io
import traceback
from PIL import Image
from io import BytesIO

from pydantic import BaseModel, Field

LANGCHAIN_AVAILABLE = False
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.messages import HumanMessage
    from langchain_core.output_parsers import JsonOutputParser
    LANGCHAIN_AVAILABLE = True
except ImportError:
    print("ERROR: langchain_google_genai or its dependencies not found. AI services will not be available.")

from .constants import TARGET_FORMAT


# 1. Fashion Item Detection Model
class DominantColor(BaseModel):
    hex_code: str = Field(..., description="The HEX code of the color (e.g., #RRGGBB).")
    color_name: str = Field(..., description="A human-readable name for the color (e.g., 'Red', 'Forest Green').")
    percentage: str = Field(..., description="The percentage of this color on the item (e.g., '70%').")

class FashionItem(BaseModel):
    item_name: str = Field(..., description="A concise descriptive name for the fashion item (e.g., 'Blue Denim Shirt').")
    category: str = Field(..., description="A general classification of the item (e.g., 'Topwear', 'Footwear', 'Accessory').")
    bounding_box: list[float] = Field(..., description="Normalized coordinates [ymin, xmin, ymax, xmax] of the item's bounding box.")
    dominant_colors: list[DominantColor] = Field(..., description="A list of dominant colors for this item.")

class FashionDetailsResponse(BaseModel):
    fashion_items: list[FashionItem] = Field(..., description="A list of detected fashion items with their details.")

# 2. Size Estimation Model
class SizeEstimation(BaseModel):
    item_description: str = Field(..., description="A brief description of the garment (e.g., 'red floral dress').")
    estimated_size: str = Field(..., description="The suggested size (e.g., 'M / US 6-8', 'Relaxed Fit Medium').")
    reasoning: str = Field(..., description="A brief explanation for the size recommendation.")

class SizeEstimationResponse(BaseModel):
    size_estimations: list[SizeEstimation] = Field(..., description="A list of size estimations for detected garments.")

# 3. Fashion Copywriter Model
class FashionCopyResponse(BaseModel):
    product_description: str = Field(..., description="An engaging product description for e-commerce.")
    styling_suggestions: str = Field(..., description="2-3 brief styling tips or outfit recommendations.")
    social_media_caption: str = Field(..., description="A short, catchy social media caption with hashtags (max ~280 characters).")

# 4. Smart Recommendations Model
class SmartRecommendationsResponse(BaseModel):
    complementary_suggestions: str = Field(..., description="Ideas for complementary items or colors.")
    similar_styles: str = Field(..., description="Recommendations for similar styles the user might like.")
    seasonal_advice: str = Field(..., description="Brief tips on how the item could be styled for a particular season.")

# --- Helper function for AI calls ---
def _call_gemini_api(api_key: str, model_name: str, prompt_text: str, image_bytes: bytes = None, temperature: float = 0.3):
    if not LANGCHAIN_AVAILABLE:
        print(f"ERROR: _call_gemini_api called but Langchain is not available.")
        return None
    try:
        llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key, temperature=temperature)

        messages_content = [{"type": "text", "text": prompt_text}]
        if image_bytes:
            encoded_image_base64 = base64.b64encode(image_bytes).decode('utf-8')
            image_mime_type = f"image/{TARGET_FORMAT.lower()}"
            messages_content.append({"type": "image_url", "image_url": f"data:{image_mime_type};base64,{encoded_image_base64}"})

        message = HumanMessage(content=messages_content)

        print(f"INFO: Sending request to Gemini API for {model_name}...")
        response = llm.invoke([message])
        print(f"INFO: Received response from Gemini API for {model_name}.")

        return response.content.strip()
    except Exception as e:
        print(f"ERROR: Failed to call Gemini API: {e}")
        print(traceback.format_exc())
        return None

def get_fashion_details_from_image(image_bytes: bytes, api_key: str, model_name: str) -> dict | None:
    if not LANGCHAIN_AVAILABLE or not api_key:
        print("ERROR: Langchain not available or API key not provided.")
        return None

    parser = JsonOutputParser(pydantic_object=FashionDetailsResponse)

    prompt_text = f"""Analyze the provided image and identify all distinct fashion items.
For each detected fashion item, provide the following details in JSON format:
1.  **item_name**: A concise descriptive name (e.g., "Blue Denim Shirt", "Leather Ankle Boots").
2.  **category**: A general classification (e.g., "Topwear", "Footwear", "Accessory").
3.  **bounding_box**: Normalized coordinates `[ymin, xmin, ymax, xmax]` defining the rectangular region where the item is located. Values must be between 0.0 and 1.0.
4.  **dominant_colors**: A list of 1 to 3 objects, each describing a dominant color for *that specific item*. Each color object must contain:
    *   **hex_code**: The color represented as a standard hexadecimal value (e.g., "#FF0000").
    *   **color_name**: A common, human-readable name for the color (e.g., "Crimson Red", "Forest Green").
    *   **percentage**: An estimated percentage representing how much of that color is present on that specific item (e.g., "70%").

Return ONLY the JSON object conforming to the following Pydantic schema:
{parser.get_format_instructions()}
Ensure the bounding box coordinates are always valid (ymin < ymax, xmin < xmax) and within [0.0, 1.0].
If no fashion items are detected, return an empty list for "fashion_items".
"""
    response_content = _call_gemini_api(api_key, model_name, prompt_text, image_bytes)

    if not response_content:
        return None

    try:
        parsed_response = parser.parse(response_content)
        return parsed_response
    except Exception as e:
        print(f"ERROR: Failed to parse or validate fashion details response: {e}")
        print(traceback.format_exc())
        return None

def get_size_estimation_for_items(image_bytes: bytes, api_key: str, model_name: str, detected_items: list = None) -> dict | None:
    if not LANGCHAIN_AVAILABLE or not api_key:
        print("ERROR: Langchain not available or API key not provided.")
        return None

    parser = JsonOutputParser(pydantic_object=SizeEstimationResponse)

    item_context_prompt = ""
    if detected_items and isinstance(detected_items, list) and len(detected_items) > 0:
        item_names = [item.get("item_name", "unknown item") for item in detected_items]
        item_context_prompt = f"The following items have already been detected in the image: {', '.join(item_names)}. Focus your size estimation on these items if possible, or other clearly visible garments."
    else:
        item_context_prompt = "Focus your size estimation on clearly visible garments in the image."

    prompt_text = f"""Analyze the provided image to estimate sizes for wearable fashion garments.
{item_context_prompt}

Return ONLY the JSON object conforming to the following Pydantic schema:
{parser.get_format_instructions()}
If you cannot confidently estimate the size for any item, or if no suitable garments are visible, return an empty list for "size_estimations".
Analyze garment proportions and any contextual clues visible. If a person is visible wearing the item, consider general human proportions but avoid making assumptions about the person's specific body measurements.
"""
    response_content = _call_gemini_api(api_key, model_name, prompt_text, image_bytes, temperature=0.3)

    if not response_content:
        return None

    try:
        parsed_response = parser.parse(response_content)
        return parsed_response
    except Exception as e:
        print(f"ERROR: Failed to parse or validate size estimation response: {e}")
        print(traceback.format_exc())
        return None

def generate_fashion_copy(image_bytes: bytes, api_key: str, model_name: str, detected_items: list = None) -> dict | None:
    if not LANGCHAIN_AVAILABLE or not api_key:
        print("ERROR: Langchain not available or API key not provided.")
        return None

    parser = JsonOutputParser(pydantic_object=FashionCopyResponse)

    item_context_prompt = "Analyze the fashion item(s) visible in the provided image."
    if detected_items and isinstance(detected_items, list) and len(detected_items) > 0:
        item_descriptions = []
        for item in detected_items:
            name = item.get("item_name", "item")
            category = item.get("category", "")
            colors = ", ".join([c.get("color_name", "") for c in item.get("dominant_colors", []) if c.get("color_name")])
            desc = f"{name} (Category: {category})"
            if colors:
                desc += f", Dominant Colors: {colors}"
            item_descriptions.append(desc)

        if item_descriptions:
            item_context_prompt = f"The image contains the following detected fashion items: {'; '.join(item_descriptions)}. Generate content primarily focusing on these items or the overall look."

    prompt_text = f"""
{item_context_prompt}

Return ONLY the JSON object conforming to the following Pydantic schema:
{parser.get_format_instructions()}

Ensure the generated text is creative, relevant to the visual information, and positive in tone.
If the image does not clearly show fashion items or is inappropriate, return empty strings for all fields.
"""
    response_content = _call_gemini_api(api_key, model_name, prompt_text, image_bytes, temperature=0.7)

    if not response_content:
        return None

    try:
        parsed_response = parser.parse(response_content)
        return parsed_response
    except Exception as e:
        print(f"ERROR: Failed to parse or validate fashion copy response: {e}")
        print(traceback.format_exc())
        return None

def get_smart_recommendations(api_key: str, model_name: str, detected_items: list) -> dict | None:
    if not LANGCHAIN_AVAILABLE or not api_key:
        print("ERROR: Langchain not available or API key not provided.")
        return None

    if not detected_items or not isinstance(detected_items, list) or len(detected_items) == 0:
        print("INFO: get_smart_recommendations called without detected items to analyze.")
        return {
            "complementary_suggestions": "No items were provided for recommendations.",
            "similar_styles": "No items were provided for recommendations.",
            "seasonal_advice": "No items were provided for recommendations."
        }

    parser = JsonOutputParser(pydantic_object=SmartRecommendationsResponse)

    item_descriptions = []
    for item in detected_items:
        name = item.get("item_name", "item")
        category = item.get("category", "")
        colors = ", ".join([c.get("color_name", "") for c in item.get("dominant_colors", []) if c.get("color_name")])
        desc = f"{name} (Category: {category})"
        if colors:
            desc += f", Dominant Colors: {colors}"
        item_descriptions.append(desc)

    items_context = "; ".join(item_descriptions)

    prompt_text = f"""
Based on the following detected fashion items: {items_context}, generate the following smart recommendations:

Return ONLY the JSON object conforming to the following Pydantic schema:
{parser.get_format_instructions()}

Ensure the recommendations are creative, relevant to the fashion items, and helpful.
"""
    response_content = _call_gemini_api(api_key, model_name, prompt_text, temperature=0.5)

    if not response_content:
        return None

    try:
        parsed_response = parser.parse(response_content)
        return parsed_response
    except Exception as e:
        print(f"ERROR: Failed to parse or validate smart recommendations response: {e}")
        print(traceback.format_exc())
        return None
