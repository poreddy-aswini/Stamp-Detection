import os
import re
import cv2
import numpy as np
import pandas as pd
import pymupdf  # Replaced 'import fitz' to resolve deprecation warning
import pytesseract

# Set Tesseract binary path if you get a TesseractNotFoundError on Windows:
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# =====================================================================
# STEP 4 CORE ENGINE: STAMP VERIFICATION FUNCTION
# # =====================================================================
# def verify_cell_stamp(cell_crop):
#     """
#     Evaluates a cropped cell image for a wet stamp.
#     Returns True (GREEN) if official stamp is found, else False (RED).
#     """
#     if cell_crop is None or cell_crop.size == 0:
#         return False

#     # 1. Color Masking (Filter Blue / Purple / Red stamp ink)
#     hsv = cv2.cvtColor(cell_crop, cv2.COLOR_BGR2HSV)

#     # Blue/Purple Ink Range
#     lower_blue = np.array([90, 30, 40])
#     upper_blue = np.array([160, 255, 255])

#     # Red Ink Range
#     lower_red1 = np.array([0, 40, 40])
#     upper_red1 = np.array([10, 255, 255])
#     lower_red2 = np.array([170, 40, 40])
#     upper_red2 = np.array([180, 255, 255])

#     mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
#     mask_red = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
#     combined_mask = mask_blue | mask_red

#     ink_pixels = cv2.countNonZero(combined_mask)
    
#     # Fail fast if there's no colored ink (e.g., blank, black ink signature, or plain text)
#     if ink_pixels < 120:
#         return False

#     # 2. Shape Analysis (Check for circular/oval stamp geometry)
#     contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     has_curved_boundary = False
#     for cnt in contours:
#         if cv2.contourArea(cnt) > 80:
#             perimeter = cv2.arcLength(cnt, True)
#             if perimeter > 0:
#                 approx = cv2.approxPolyDP(cnt, 0.03 * perimeter, True)
#                 if len(approx) >= 5:  # Stamp shapes evaluate to >= 5 vertices
#                     has_curved_boundary = True
#                     break

#     # 3. OCR Keyword & Regex Verification
#     try:
#         rgb = cv2.cvtColor(cell_crop, cv2.COLOR_BGR2RGB)
#         text = pytesseract.image_to_string(rgb, config="--psm 6")
#         stamp_pattern = r"(?i)(MOCK|INSPECTION|CHECK|TASL|VERIFIED|APPROVED|PASS|\d{2}-\d+)"
#         has_regex_match = bool(re.search(stamp_pattern, text))
#     except Exception:
#         # Fallback to color/shape check if Tesseract is not installed locally
#         has_regex_match = False

#     return has_curved_boundary or has_regex_match


def verify_cell_stamp(cell_crop):
    """
    Evaluates a cropped cell image for an official wet stamp.
    
    STRICT RULE ENGINE:
    - Empty Cell                -> Returns False (RED)
    - Any Text / Signature Only  -> Returns False (RED)
    - Checkmark / Scribble Only -> Returns False (RED)
    - Valid Official Stamp      -> Returns True  (GREEN)
    """
    # Rule 1: Check for Empty / Invalid Crop
    if cell_crop is None or cell_crop.size == 0:
        return False  # Mark RED

    # Rule 2: Convert to Gray to check total content (Empty Box Check)
    gray = cv2.cvtColor(cell_crop, cv2.COLOR_BGR2GRAY)
    # Thresholding to isolate black/dark marks (signatures, text, box lines)
    _, binary_inv = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    total_dark_pixels = cv2.countNonZero(binary_inv)

    # If the box is completely blank/empty (no text, no ink, no marks at all)
    if total_dark_pixels < 50:
        return False  # Mark RED (Empty)

    # -----------------------------------------------------------------
    # Rule 3: Isolate Colored Stamp Ink (Blue / Red / Purple)
    # -----------------------------------------------------------------
    hsv = cv2.cvtColor(cell_crop, cv2.COLOR_BGR2HSV)

    # Blue/Purple Ink Range
    lower_blue = np.array([90, 30, 40])
    upper_blue = np.array([160, 255, 255])

    # Red Ink Range
    lower_red1 = np.array([0, 40, 40])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 40, 40])
    upper_red2 = np.array([180, 255, 255])

    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    mask_red = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
    stamp_ink_mask = mask_blue | mask_red

    stamp_ink_pixels = cv2.countNonZero(stamp_ink_mask)

    # If there is text or signature, but NO stamp color ink -> Mark RED
    if stamp_ink_pixels < 120:
        return False  # Mark RED (Plain text, black signature, or blank)

    # -----------------------------------------------------------------
    # Rule 4: Shape Geometry Check (Circular / Oval Stamp Outline)
    # -----------------------------------------------------------------
    contours, _ = cv2.findContours(stamp_ink_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    has_stamp_shape = False

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 100:  # Ignore tiny color specs/noise
            perimeter = cv2.arcLength(cnt, True)
            if perimeter > 0:
                approx = cv2.approxPolyDP(cnt, 0.03 * perimeter, True)
                # Official circular/elliptical stamps approximate to >= 5 vertices
                if len(approx) >= 5:
                    has_stamp_shape = True
                    break

    # -----------------------------------------------------------------
    # Rule 5: OCR Stamp Text Keyword Matching
    # -----------------------------------------------------------------
    has_stamp_text = False
    try:
        rgb = cv2.cvtColor(cell_crop, cv2.COLOR_BGR2RGB)
        ocr_text = pytesseract.image_to_string(rgb, config="--psm 6")
        
        # Regex matching specifically for official approval stamp text
        stamp_pattern = r"(?i)(MOCK|INSPECTION|CHECK|TASL|VERIFIED|APPROVED|PASS|\d{2}-\d+)"
        has_stamp_text = bool(re.search(stamp_pattern, ocr_text))
    except Exception:
        pass

    # ONLY mark GREEN if both colored stamp ink is present AND it has either a stamp shape or stamp text
    if stamp_ink_pixels >= 120 and (has_stamp_shape or has_stamp_text):
        return True  # Mark GREEN
    
    return False  # Mark RED for everything else

# =====================================================================
# PDF / IMAGE PIPELINE (STEPS 1 TO FINAL)
# =====================================================================
def process_visual_document(input_path, output_path, target_cells=None):
    """
    Processes PDF or Image files using PyMuPDF (no Poppler required),
    checks target cells, draws Red/Green boxes, and returns the annotated document.
    """
    pages = []

    # Step 2: Convert PDF pages directly using PyMuPDF
    if input_path.lower().endswith(".pdf"):
        doc = pymupdf.open(input_path)
        for page in doc:
            pix = page.get_pixmap(dpi=300)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
            # PyMuPDF yields RGB -> Convert to BGR for OpenCV
            cv_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            pages.append(cv_img)
        doc.close()
    else:
        pages = [cv2.imread(input_path)]

    # Default grid boxes if none passed: [(x, y, width, height)]
    if not target_cells:
        target_cells = [
            (450, 300, 220, 130),  # Row 1
            (450, 450, 220, 130),  # Row 2
            (450, 600, 220, 130)   # Row 3
        ]

    annotated_images = []

    for cv_img in pages:
        # Step 3 & 4: Loop through cell objects and evaluate
        for (x, y, w, h) in target_cells:
            cell_crop = cv_img[y:y+h, x:x+w]
            is_stamp = verify_cell_stamp(cell_crop)

            # Assign Status
            status = "GREEN" if is_stamp else "RED"
            color = (0, 255, 0) if is_stamp else (0, 0, 255)  # BGR

            # Step 5: Draw Green / Red Box & Badge
            cv2.rectangle(cv_img, (x, y), (x + w, y + h), color, thickness=5)
            cv2.putText(
                cv_img, status, (x, max(y - 12, 30)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, thickness=3
            )

        annotated_images.append(cv_img)

    # Step 6 / Final Step: Reconstruct into output PDF / Image file
    if input_path.lower().endswith(".pdf"):
        pdf_out = pymupdf.open()
        for img in annotated_images:
            _, encoded = cv2.imencode(".png", img)
            img_bytes = encoded.tobytes()
            doc_page = pdf_out.new_page(width=img.shape[1], height=img.shape[0])
            doc_page.insert_image(doc_page.rect, stream=img_bytes)
        pdf_out.save(output_path)
        pdf_out.close()
    else:
        cv2.imwrite(output_path, annotated_images[0])

    return output_path


# =====================================================================
# CSV PIPELINE (STEPS 1 TO FINAL)
# =====================================================================
def process_csv_document(input_path, output_path):
    df = pd.read_csv(input_path)
    stamp_pattern = r"(?i)(STAMP|STAMPED|VERIFIED|APPROVED|PASS|MOCK)"

    results = []
    for _, row in df.iterrows():
        row_str = " ".join(row.astype(str).tolist())
        if re.search(stamp_pattern, row_str):
            results.append("GREEN")
        else:
            results.append("RED")

    df["VERIFICATION_STATUS"] = results
    df.to_csv(output_path, index=False)
    return output_path


# =====================================================================
# STEP 1: MAIN ROUTER ENTRY POINT
# =====================================================================
def process_incoming_file(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")

    filename, ext = os.path.splitext(file_path)
    output_path = f"{filename}_PROCESSED{ext}"

    print(f"📥 Received file: {file_path}")
    print("🔄 Processing document through verification engine...")

    if ext.lower() in ['.pdf', '.png', '.jpg', '.jpeg']:
        final_output = process_visual_document(file_path, output_path)
    elif ext.lower() == '.csv':
        final_output = process_csv_document(file_path, output_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    print(f"✅ Processing complete! Returning output file: {final_output}")
    return final_output


# =====================================================================
# EXECUTION TEST
# =====================================================================
if __name__ == "__main__":
    # 1. Input file path
    input_file = r"C:\Users\techl\OneDrive\Desktop\project\test_document.pdf"

    # 2. Define the exact cell bounding boxes [(x, y, w, h)]
    my_target_cells = [
        (450, 220, 350, 280),  # Row 1: MOCK INSPECTION cell
        (450, 500, 350, 280),  # Row 2: MOCK CHECK DATA cell
        (450, 780, 350, 280),  # Row 3: MOCK DATA ENTRY cell
    ]

    # 3. Run execution
    filename, ext = os.path.splitext(input_file)
    output_path = f"{filename}_PROCESSED{ext}"

    output_file = process_visual_document(input_file, output_path, target_cells=my_target_cells)

    print(f"🎉 Result saved to: {output_file}")