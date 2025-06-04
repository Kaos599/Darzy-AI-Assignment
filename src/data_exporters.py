# src/data_exporters.py

import io
import csv
import json

def _flatten_colors_for_csv(colors_list: list, max_colors: int = 3) -> dict:
    # Helper to flatten a list of color dictionaries for CSV output.
    flat_colors = {}
    for i in range(max_colors):
        color_data = {}
        if i < len(colors_list) and isinstance(colors_list[i], dict):
            color_data = colors_list[i]

        flat_colors[f"color_{i+1}_name"] = color_data.get("color_name", "")
        flat_colors[f"color_{i+1}_hex"] = color_data.get("hex_code", "")
        flat_colors[f"color_{i+1}_percentage"] = color_data.get("percentage", "")
    return flat_colors

def convert_fashion_details_to_csv(analysis_data: dict, max_colors_per_item: int = 3) -> str:
    # Converts fashion item analysis data into a CSV formatted string.
    if not analysis_data or "fashion_items" not in analysis_data or not analysis_data.get("fashion_items"):
        return ""

    output = io.StringIO()
    base_fieldnames = ["item_name", "category", "fabric_type", "fabric_confidence_score", "bbox_ymin", "bbox_xmin", "bbox_ymax", "bbox_xmax"]
    color_fieldnames = []
    for i in range(max_colors_per_item):
        color_fieldnames.extend([f"color_{i+1}_name", f"color_{i+1}_hex", f"color_{i+1}_percentage"])
    full_fieldnames = base_fieldnames + color_fieldnames

    writer = csv.DictWriter(output, fieldnames=full_fieldnames, lineterminator='\n')
    writer.writeheader()

    for item in analysis_data["fashion_items"]:
        if not isinstance(item, dict):
            print(f"Skipping an item with unexpected format: {item}") # Essential log
            continue

        bbox_list = item.get("bounding_box", [])
        row = {
            "item_name": item.get("item_name", ""),
            "category": item.get("category", ""),
            "fabric_type": item.get("fabric_type", ""),
            "fabric_confidence_score": item.get("fabric_confidence_score", ""),
            "bbox_ymin": bbox_list[0] if len(bbox_list) == 4 else "",
            "bbox_xmin": bbox_list[1] if len(bbox_list) == 4 else "",
            "bbox_ymax": bbox_list[2] if len(bbox_list) == 4 else "",
            "bbox_xmax": bbox_list[3] if len(bbox_list) == 4 else "",
        }

        item_colors = item.get("dominant_colors", [])
        flattened_color_data = _flatten_colors_for_csv(item_colors, max_colors_per_item)
        row.update(flattened_color_data)
        writer.writerow(row)
    return output.getvalue()

def convert_palette_to_csv(analysis_data: dict) -> str:
    # DEPRECATED: Converts overall color palette data (old format) to a CSV string.
    if not analysis_data or "colors" not in analysis_data or not analysis_data.get("colors"):
        return ""
    output = io.StringIO()
    fieldnames = ["color_name", "hex_code", "percentage"]
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator='\n')
    writer.writeheader()
    for color_entry in analysis_data["colors"]:
        if not isinstance(color_entry, dict):
            print(f"Skipping a color entry with unexpected format: {color_entry}") # Essential log
            continue
        writer.writerow({
            "color_name": color_entry.get("color_name", ""),
            "hex_code": color_entry.get("hex_code", ""),
            "percentage": color_entry.get("percentage", "")
        })
    return output.getvalue()

def export_to_json_string(data: dict) -> str:
    # Converts a Python dictionary to a pretty-printed JSON string.
    try:
        return json.dumps(data, indent=2)
    except TypeError as e:
        print(f"Error serializing data to JSON: {e}") # Essential log
        return ""
