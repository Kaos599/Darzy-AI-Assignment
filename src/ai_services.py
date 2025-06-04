# src/ai_services.py

import json
import base64
import io
import traceback

LANGCHAIN_AVAILABLE = False
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.messages import HumanMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    print("ERROR: langchain_google_genai or its dependencies not found. AI services will not be available.")

from .constants import TARGET_FORMAT

def get_fashion_details_from_image(image_bytes: bytes, api_key: str, model_name: str) -> dict | None:
    # Analyzes an image with Gemini to detect fashion items, categories, bounding boxes, and colors.
    if not LANGCHAIN_AVAILABLE:
        print("ERROR: get_fashion_details_from_image called but Langchain is not available.")
        return None
    if not api_key:
        print("ERROR: GEMINI_API_KEY is not provided to get_fashion_details_from_image.")
        return None

    try:
        llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key, temperature=0.25)

        encoded_image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        image_mime_type = f"image/{TARGET_FORMAT.lower()}"

        prompt_text = f"""Analyze the provided image to identify all distinct fashion items. For EACH detected fashion item, provide the following details:
1.  "item_name": A concise name for the item (e.g., "T-shirt", "Blue Jeans", "Leather Handbag", "Sneakers").
2.  "category": A general category for the item (e.g., "Topwear", "Bottomwear", "Footwear", "Accessory", "Outerwear", "Full Body").
3.  "bounding_box": Normalized bounding box coordinates [ymin, xmin, ymax, xmax], where values are between 0.0 and 1.0. Ensure ymin < ymax and xmin < xmax.
4.  "dominant_colors": A list of 1 to 3 dominant colors for THIS specific item. Each color should be an object with:
    *   "hex_code": The HEX code (e.g., "#RRGGBB").
    *   "color_name": A common name for the color (e.g., "Navy Blue").
    *   "percentage": Estimated percentage of this color ON THIS ITEM (e.g., "70%").
5.  "fabric_type": The most likely fabric type (e.g., "Cotton", "Denim", "Wool", "Silk", "Polyester", "Leather", "Blend", "Unknown").
6.  "fabric_confidence_score": A confidence score for the fabric type, between 0.0 (low confidence) and 1.0 (high confidence).

Additionally, provide an overall "styling_recommendation" for the detected fashion items. This recommendation should be a general textual advice on how to style the detected items, suggesting complementary colors or general outfit suggestions, keeping in mind the detected fabric types and colors.

Return ONLY the JSON object as your response, starting with {{ and ending with }}.
The JSON object must have a top-level key "fashion_items", which is a list of objects, and a top-level key "styling_recommendation", which is a string.
Each object in the "fashion_items" list represents a detected fashion item and must contain all keys: "item_name", "category", "bounding_box", "dominant_colors", "fabric_type", and "fabric_confidence_score".

Example of the expected JSON structure:
{{
  "fashion_items": [
    {{
      "item_name": "Red Cotton T-shirt",
      "category": "Topwear",
      "bounding_box": [0.1, 0.2, 0.5, 0.8],
      "dominant_colors": [
        {{ "hex_code": "#FF0000", "color_name": "Red", "percentage": "90%" }},
        {{ "hex_code": "#FFFFFF", "color_name": "White", "percentage": "10%" }}
      ],
      "fabric_type": "Cotton",
      "fabric_confidence_score": 0.95
    }}
  ],
  "styling_recommendation": "This red cotton t-shirt would pair well with dark wash denim jeans and white sneakers for a casual look. For a slightly more dressed-up feel, consider layering with a light gray cardigan."
}}
If no fashion items are detected, return an empty list for "fashion_items": {{"fashion_items": []}} and an empty string for "styling_recommendation": "".
Focus only on actual clothing, shoes, and significant accessories. Ignore background elements.
Be precise with bounding boxes and analyze the image thoroughly for fabric details and provide relevant styling advice.
"""
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": f"data:{image_mime_type};base64,{encoded_image_base64}"},
            ]
        )

        print("INFO: Sending request to Gemini API for fashion details...")
        response = llm.invoke([message])
        print("INFO: Received response from Gemini API.")

        response_content = response.content.strip()

        json_str = ""
        if response_content.startswith("```json") and response_content.endswith("```"):
            json_str = response_content[7:-3].strip()
        elif response_content.startswith("```") and response_content.endswith("```"):
            json_str = response_content[3:-3].strip()
        elif response_content.startswith("{") and response_content.endswith("}"):
            json_str = response_content
        else:
            # Try to import re for the fallback search
            import re
            match = re.search(r'\{[\s\S]*\}', response_content)
            if match:
                json_str = match.group(0)
                print("WARNING: AI response was not clean JSON. Attempted to extract JSON.")
            else:
                print(f"ERROR: Could not find any JSON block in AI response. Snippet: {response_content[:500]}")
                return None

        try:
            analysis_data = json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"ERROR: Failed to parse JSON from AI response. Error: {e}. Content processed: {json_str[:500]}")
            return None

        if not isinstance(analysis_data, dict) or "fashion_items" not in analysis_data or \
           not isinstance(analysis_data["fashion_items"], list) or "styling_recommendation" not in analysis_data or \
           not isinstance(analysis_data["styling_recommendation"], str):
            print(f"ERROR: AI response JSON structure is invalid. Data: {analysis_data}")
            return None

        validated_items = []
        for item_idx, item in enumerate(analysis_data.get("fashion_items", [])):
            if not isinstance(item, dict) or not all(k in item for k in ["item_name", "category", "bounding_box", "dominant_colors", "fabric_type", "fabric_confidence_score"]):
                print(f"WARNING: Item {item_idx} missing required fields. Skipping. Item data: {item}")
                continue

            bbox = item["bounding_box"]
            if not (isinstance(bbox, list) and len(bbox) == 4 and \
                    all(isinstance(n, (float, int)) and 0.0 <= n <= 1.0 for n in bbox) and \
                    bbox[0] < bbox[2] and bbox[1] < bbox[3]):
                print(f"WARNING: Item '{item.get('item_name', 'Unknown')}' has invalid bbox {bbox}. Skipping.")
                continue

            if not isinstance(item["dominant_colors"], list):
                print(f"WARNING: Item '{item.get('item_name', 'Unknown')}' dominant_colors is not a list. Assigning empty list.")
                item["dominant_colors"] = []

            valid_colors_for_item = []
            for color_idx, color_detail in enumerate(item.get("dominant_colors", [])):
                 if isinstance(color_detail, dict) and all(k_color in color_detail for k_color in ["hex_code", "color_name", "percentage"]):
                      valid_colors_for_item.append(color_detail)
                 else:
                      print(f"WARNING: Invalid color detail (item: '{item.get('item_name', 'Unknown')}', color_idx: {color_idx}). Skipping color. Detail: {color_detail}")
            item["dominant_colors"] = valid_colors_for_item
            validated_items.append(item)

        analysis_data["fashion_items"] = validated_items
        return analysis_data

    except Exception as e:
        print(f"ERROR: An unexpected error occurred in get_fashion_details_from_image: {e}")
        print(traceback.format_exc())
        return None
