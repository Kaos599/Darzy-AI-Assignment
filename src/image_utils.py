# src/image_utils.py

import io
from PIL import Image, UnidentifiedImageError, ImageDraw, ImageFont
from .constants import MAX_IMAGE_SIZE, JPEG_QUALITY, TARGET_FORMAT, DEFAULT_FONT_SIZE

def process_image(image_bytes: bytes, filename: str) -> tuple[bytes | None, str | None, str]:
    # Processes an uploaded image: validates, resizes, converts, and compresses.
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.load()

        original_mode = img.mode
        if TARGET_FORMAT == 'JPEG' and original_mode not in ['RGB', 'L']:
            if original_mode in ['RGBA', 'P', 'LA']:
                img = img.convert('RGB')
            elif original_mode != 'RGB':
                 img = img.convert('RGB')

        resized = False
        if img.size[0] > MAX_IMAGE_SIZE[0] or img.size[1] > MAX_IMAGE_SIZE[1]:
            img.thumbnail(MAX_IMAGE_SIZE, Image.Resampling.LANCZOS)
            resized = True

        processed_image_io = io.BytesIO()
        name_parts = filename.rsplit('.', 1)
        base_name = name_parts[0] if len(name_parts) > 1 else filename
        new_filename = f"{base_name}_processed.{TARGET_FORMAT.lower()}"

        save_params = {}
        if TARGET_FORMAT == "JPEG":
            save_params['format'] = "JPEG"
            save_params['quality'] = JPEG_QUALITY
        elif TARGET_FORMAT == "PNG":
            save_params['format'] = "PNG"
            save_params['optimize'] = True
        else:
            return None, None, f"Unsupported TARGET_FORMAT: {TARGET_FORMAT}"

        img.save(processed_image_io, **save_params)
        processed_image_bytes = processed_image_io.getvalue()

        message = f"Image processed as {TARGET_FORMAT}"
        if resized: message += f" (resized to fit {MAX_IMAGE_SIZE[0]}x{MAX_IMAGE_SIZE[1]}px)"
        if TARGET_FORMAT == "JPEG": message += f" (quality: {JPEG_QUALITY}%)."
        else: message += "."
        return processed_image_bytes, new_filename, message

    except UnidentifiedImageError:
        return None, None, "Processing failed: The file is not a valid image or its format is not supported."
    except IOError as e:
        return None, None, f"Processing failed: Could not read or process image file. Error: {e}"
    except Exception as e:
        return None, None, f"Processing failed: An unexpected error during image processing: {e}"

def draw_detections_on_image(image_bytes: bytes, fashion_items_data: dict) -> bytes | None:
    # Draws bounding boxes and labels on an image based on fashion item detections.
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        draw = ImageDraw.Draw(img)

        try:
            font = ImageFont.truetype("arial.ttf", DEFAULT_FONT_SIZE)
        except IOError:
            font = ImageFont.load_default()

        img_width, img_height = img.size

        for item in fashion_items_data.get("fashion_items", []):
            bbox = item.get("bounding_box")
            if not (isinstance(bbox, list) and len(bbox) == 4 and all(isinstance(n, (float, int)) for n in bbox)):
                print(f"Skipping item due to invalid bounding box: {item.get('item_name')}") # Essential log
                continue

            ymin, xmin, ymax, xmax = bbox
            left = xmin * img_width; right = xmax * img_width
            top = ymin * img_height; bottom = ymax * img_height

            item_colors = item.get("dominant_colors", [])
            box_color_hex = item_colors[0].get("hex_code", "#FF0000") if item_colors and isinstance(item_colors[0], dict) else "#FF0000"
            if not (isinstance(box_color_hex, str) and box_color_hex.startswith('#') and len(box_color_hex) in [4, 7]):
                box_color_hex = "#FF0000"

            draw.rectangle([(left, top), (right, bottom)], outline=box_color_hex, width=3)

            label = f"{item.get('item_name', 'Unknown')} ({item.get('category', 'N/A')})"

            try:
                text_bbox = draw.textbbox((left, top), label, font=font) # Use (left,top) for reference, actual position adjusted
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                text_anchor_y = top - text_height - 4
                if text_anchor_y < 2:
                    text_anchor_y = top + 2
            except AttributeError:
                text_width, text_height = draw.textsize(label, font=font)
                text_anchor_y = top - text_height - 4
                if text_anchor_y < 2: text_anchor_y = top + 2

            draw.rectangle(
                [(left, text_anchor_y), (left + text_width + 4, text_anchor_y + text_height + 2)],
                fill=box_color_hex
            )
            draw.text((left + 2, text_anchor_y), label, fill="white", font=font)

        annotated_image_io = io.BytesIO()
        img.save(annotated_image_io, format="PNG")
        return annotated_image_io.getvalue()
    except Exception as e:
        print(f"Error drawing detections on image: {e}")
        return None
