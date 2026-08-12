import csv
import math
import os
import uuid

import cv2
import numpy as np
import pymupdf
from flask import Flask, jsonify, render_template, request, send_from_directory
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Border, Side
from openpyxl.utils import get_column_letter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "outputs")
PREVIEW_FOLDER = os.path.join(BASE_DIR, "previews")

ALLOWED_EXTS = {".xlsx", ".xlsm", ".csv", ".pdf"}

GREEN = Side(style="thick", color="00B050")
RED = Side(style="thick", color="FF0000")
GREEN_BORDER = Border(left=GREEN, right=GREEN, top=GREEN, bottom=GREEN)
RED_BORDER = Border(left=RED, right=RED, top=RED, bottom=RED)

PDF_GREEN = (0, 0.69, 0.31)
PDF_RED = (1, 0, 0)
PDF_BORDER_WIDTH = 2.2

MAX_PREVIEW = 500
IMAGE_OVERLAP_THRESHOLD = 0.3

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(PREVIEW_FOLDER, exist_ok=True)


# Why: clean up a value into a standard lower-case form so two different
# strings (e.g. "PAID" and " Paid ") can be compared as equal.
def _normalize(value):
    return str(value).strip().lower()


# Why: check whether a cell's value counts as "stamp present", by seeing if
# it matches any of the user-typed words (like paid, received).
def _is_present(value, present_set):
    if value is None:
        return False
    norm = _normalize(value)
    return norm != "" and norm in present_set


# ---------------------------------------------------------------------------
# XLSX / CSV helpers
# ---------------------------------------------------------------------------

# Why: read an Excel file and return its contents as a list of rows, so the
# rest of the app can work with plain data.
def _read_rows_xlsx(filepath):
    wb = load_workbook(filepath, read_only=True, data_only=True)
    ws = wb.active
    rows = [list(r) for r in ws.iter_rows(values_only=True)]
    wb.close()
    return rows


# Why: read a CSV file and return its contents as a list of rows (same output
# as the Excel reader, so the code below can ignore the file type).
def _read_rows_csv(filepath):
    rows = []
    with open(filepath, "r", newline="", encoding="utf-8-sig", errors="replace") as fh:
        reader = csv.reader(fh)
        for row in reader:
            rows.append(row)
    return rows


# Why: pick the right reader based on the file extension, so callers can just
# ask for "the rows" without worrying about Excel vs CSV.
def _read_rows(filepath, ext):
    if ext == ".csv":
        return _read_rows_csv(filepath)
    return _read_rows_xlsx(filepath)


# Why: pull out the header row from the file, so we know the column names and
# can let the user choose which column holds the stamp.
def _header_at(rows, header_row):
    header_row = max(1, header_row)
    if header_row > len(rows):
        return None
    return [str(v) if v is not None else "" for v in rows[header_row - 1]]


# ---------------------------------------------------------------------------
# PDF helpers
# ---------------------------------------------------------------------------

# Why: locate the main table on a PDF page. We need it to know exactly where
# every row and cell sits, so we can inspect and border them later.
def _pdf_main_table(page):
    """Return the largest table found on the page, or None."""
    tabs = page.find_tables()
    if not tabs.tables:
        return None
    return max(tabs.tables, key=lambda t: pymupdf.Rect(t.bbox).get_area())


# Why: get the position (rectangle) of every image on the page. This tells us
# which cells contain any kind of image at all.
def _pdf_image_rects(page):
    rects = []
    for img in page.get_images(full=True):
        rects += page.get_image_rects(img[0])
    return rects


# Why: keep only the image rectangles that could be real stamps. Anything that
# covers most of the page is a scan/background, not a stamp, so we drop those.
def _pdf_stamp_images(page):
    """Image rects that are likely real stamp images (not a full-page scan)."""
    page_area = max(page.rect.get_area(), 1)
    rects = []
    for r in _pdf_image_rects(page):
        if r.get_area() < 0.6 * page_area:
            rects.append(r)
    return rects


# Why: check whether a stamp image sits inside this cell, by measuring how
# much the cell overlaps the image. Needed to decide if a row has a stamp.
def _cell_has_image(cell, image_rects):
    best = 0.0
    for ir in image_rects:
        inter = cell & ir
        if inter.get_area() > 0:
            best = max(best, inter.get_area() / max(ir.get_area(), 1e-6))
    return best > IMAGE_OVERLAP_THRESHOLD


# Why: the core stamp detector. It renders the cell as an image and uses
# OpenCV to verify it really is a rubber stamp (ink color + fine detail),
# instead of trusting any random image. Returns True/False.
def _stamp_detected(page, cell):
    """Render the cell region and detect a stamp impression.

    Stamps are printed ink with a ring / text strokes (lots of fine edges
    inside their silhouette). Solid filled objects such as photographs
    (e.g. an apple) have few interior edges and are rejected.
    """
    # Inset the clip so table grid borders (which sit on the cell edge) are
    # not mistaken for stamp content.
    inset = min(3.0, cell.width / 4, cell.height / 4)
    clip = pymupdf.Rect(cell.x0 + inset, cell.y0 + inset, cell.x1 - inset, cell.y1 - inset)
    if clip.width <= 2 or clip.height <= 2:
        return False
    pix = page.get_pixmap(clip=clip, dpi=150, colorspace=pymupdf.csRGB)
    if pix.width < 8 or pix.height < 8:
        return False

    rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    m_red = cv2.inRange(hsv, (0, 60, 50), (10, 255, 255)) | \
            cv2.inRange(hsv, (170, 60, 50), (180, 255, 255))
    m_blue = cv2.inRange(hsv, (95, 60, 40), (140, 255, 255))
    m_purple = cv2.inRange(hsv, (125, 60, 40), (165, 255, 255))
    color = m_red | m_blue | m_purple
    dark = (gray < 170).astype(np.uint8) * 255
    ink = cv2.bitwise_or(color, dark)

    area = pix.width * pix.height
    if cv2.countNonZero(ink) / max(area, 1) < 0.008:
        return False

    contours = cv2.findContours(ink, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    if not contours:
        return False
    # Dilate the ink so faint / fragmented stamp strokes (broken rings,
    # thin text) join into one coherent object before measuring its shape.
    ink_dilated = cv2.dilate(ink, np.ones((3, 3), np.uint8), iterations=2)
    contours = cv2.findContours(ink_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    if not contours:
        return False
    obj = np.zeros((pix.height, pix.width), np.uint8)
    cv2.drawContours(obj, [max(contours, key=cv2.contourArea)], -1, 255, -1)
    obj_area = cv2.countNonZero(obj)
    if obj_area / max(area, 1) < 0.004:
        return False

    edges = cv2.Canny(cv2.medianBlur(gray, 5), 70, 190)
    edge_density = cv2.countNonZero(cv2.bitwise_and(edges, obj)) / max(obj_area, 1)

    # White strokes / holes inside the silhouette: typical of solid colored
    # stamps that carry white text (e.g. a filled "PAID" disc).
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    obj_eroded = cv2.erode(obj, np.ones((9, 9), np.uint8))
    white = ((gray > 200) & (sat < 80) & (val > 190)).astype(np.uint8)
    holes = cv2.countNonZero(cv2.bitwise_and(white, obj_eroded)) / max(obj_area, 1)

    return edge_density >= 0.09 or holes >= 0.05


# Why: draw a coloured border just inside each cell, so the original black
# grid line stays visible around it instead of being covered up.
def _draw_unique_borders(page, cells):
    """Draw an inset coloured border inside each cell."""
    for cell, color in cells:
        inset = min(3.0, cell.width / 4, cell.height / 4)
        r = pymupdf.Rect(cell.x0 + inset, cell.y0 + inset, cell.x1 - inset, cell.y1 - inset)
        if r.width <= 0 or r.height <= 0:
            continue
        page.draw_line((r.x0, r.y0), (r.x1, r.y0), color=color,
                       width=PDF_BORDER_WIDTH)
        page.draw_line((r.x0, r.y0), (r.x0, r.y1), color=color,
                       width=PDF_BORDER_WIDTH)
        page.draw_line((r.x1, r.y0), (r.x1, r.y1), color=color,
                       width=PDF_BORDER_WIDTH)
        page.draw_line((r.x0, r.y1), (r.x1, r.y1), color=color,
                       width=PDF_BORDER_WIDTH)


# Why: tidy up cell text (collapse extra spaces) so header/text matching is
# consistent and works even with messy spacing.
def _clean_cell_text(value):
    return " ".join(str(value).split())


# Why: read the column header names from a PDF, so the user can pick which
# column to check for stamps (same purpose as reading an Excel header).
def _pdf_columns(filepath):
    doc = pymupdf.open(filepath)
    try:
        for page in doc:
            tab = _pdf_main_table(page)
            if tab is None:
                continue
            extracted = tab.extract()
            if not extracted or not extracted[0]:
                continue
            return [_clean_cell_text(v) for v in extracted[0]]
    finally:
        doc.close()
    return None


# Why: make a small PNG thumbnail of each PDF page so the user can see the
# uploaded document in the browser before processing.
def _pdf_page_previews(filepath, stored_name):
    """Render each page of the PDF as a PNG thumbnail for display on upload."""
    doc = pymupdf.open(filepath)
    urls = []
    base = os.path.splitext(stored_name)[0]
    try:
        for i, page in enumerate(doc, start=1):
            fname = f"{base}_page{i}.png"
            pix = page.get_pixmap(dpi=110)
            pix.save(os.path.join(PREVIEW_FOLDER, fname))
            urls.append(f"/preview/{fname}")
    finally:
        doc.close()
    return urls


# Why: the main PDF processing routine. It walks every page, finds the selected
# columns, checks each row (image + text) in every selected column, draws
# green/red borders, saves the new PDF, and builds the results shown on the
# web page.
def _pdf_process(filepath, columns, present_set, output_path):
    """Detect stamps in the given columns across all pages and draw borders."""
    doc = pymupdf.open(filepath)
    results = []
    stats = {"present": 0, "absent": 0, "empty": 0}
    col_names = [_normalize(c) for c in columns if str(c).strip()]
    headers = None
    base_name = os.path.splitext(os.path.basename(output_path))[0]

    try:
        for page_no, page in enumerate(doc, start=1):
            tab = _pdf_main_table(page)
            if tab is None:
                continue
            extracted = tab.extract()
            if not extracted:
                continue
            first_row = [_clean_cell_text(v) for v in extracted[0]]

            if headers is None:
                headers = first_row
                stamp_idxs = [i for i, h in enumerate(headers) if _normalize(h) in col_names]
                if not stamp_idxs:
                    continue
                start_row = 1
            else:
                repeat_idxs = [i for i, h in enumerate(first_row) if _normalize(h) in col_names]
                if repeat_idxs and len(extracted) > 1:
                    stamp_idxs = repeat_idxs
                    start_row = 1
                else:
                    start_row = 0

            image_rects = _pdf_stamp_images(page)
            has_any_image = len(image_rects) > 0

            # Pass 1: analyse every row, but do NOT touch the page yet. If a
            # border were drawn here, the next row's preview (rendered later)
            # would show that coloured line bleeding over its black grid edge.
            page_borders = []
            page_thumbs = []
            page_results = []
            for r in range(start_row, tab.row_count):
                row_values = []
                if r < len(extracted):
                    row_values = ["" if v is None else _clean_cell_text(v) for v in extracted[r]]

                images = {}
                col_present = {}
                for idx in stamp_idxs:
                    text = row_values[idx] if idx < len(row_values) else ""
                    cell = pymupdf.Rect(tab.rows[r].cells[idx])

                    has_img = has_any_image and _cell_has_image(cell, image_rects)
                    text_present = bool(text) and _is_present(text, present_set)
                    stamp_detected = has_img and _stamp_detected(page, cell)
                    present = stamp_detected or text_present

                    if present:
                        stats["present"] += 1
                    else:
                        stats["absent"] += 1
                        if not has_img and _normalize(text) == "":
                            stats["empty"] += 1

                    image_url = None
                    if has_img or stamp_detected:
                        pad = 3
                        clip = pymupdf.Rect(cell.x0 - pad, cell.y0 - pad, cell.x1 + pad, cell.y1 + pad)
                        fname = f"{base_name}_p{page_no}_r{r}_c{idx}.png"
                        page_thumbs.append((clip, fname))
                        image_url = f"/preview/{fname}"

                    page_borders.append((cell, PDF_GREEN if present else PDF_RED))
                    images[str(idx)] = image_url
                    col_present[str(idx)] = present

                present = any(col_present.values())

                first_text = row_values[stamp_idxs[0]] if stamp_idxs[0] < len(row_values) else ""
                page_results.append({
                    "row": r,
                    "cells": row_values,
                    "value": first_text if first_text else ("stamp image" if present else ""),
                    "images": images,
                    "col_present": col_present,
                    "present": present,
                    "page": page_no,
                })

            # Pass 2: render the previews while the page is still clean, so the
            # black grid border around each image stays clean and no neighbour's
            # border colour leaks into it.
            for clip, fname in page_thumbs:
                pix = page.get_pixmap(clip=clip, dpi=150)
                pix.save(os.path.join(PREVIEW_FOLDER, fname))

            # Pass 3: draw the borders now, one edge at a time so shared lines
            # between adjacent cells are drawn once and keep a single colour.
            _draw_unique_borders(page, page_borders)

            results.extend(page_results)
    finally:
        doc.save(output_path)
        doc.close()

    return headers, results, stats


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
# Why: show the main web page (the UI) when the user opens the site.
def index():
    return render_template("index.html")


@app.route("/api/columns", methods=["POST"])
# Why: handles the first step — the user uploads a file and this returns the
# list of column headers (and a preview) so they can choose the stamp column.
def api_columns():
    file = request.files.get("file")
    if file is None or file.filename == "":
        return jsonify({"error": "No file uploaded."}), 400

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTS:
        return jsonify({"error": "Only .xlsx, .xlsm, .csv and .pdf files are supported."}), 400

    header_row = int(request.form.get("header_row", 1) or 1)

    stored_name = str(uuid.uuid4()) + ext
    stored_path = os.path.join(UPLOAD_FOLDER, stored_name)
    file.save(stored_path)

    if ext == ".pdf":
        columns = _pdf_columns(stored_path)
        if not columns:
            return jsonify({"error": "No table detected in the PDF. Make sure it has a visible grid."}), 400
        previews = _pdf_page_previews(stored_path, stored_name)
        return jsonify({"file": stored_name, "columns": columns, "preview": previews})

    try:
        rows = _read_rows(stored_path, ext)
    except Exception:
        return jsonify({"error": "Could not read the file. Please upload a valid table."}), 400

    if not rows:
        return jsonify({"error": "The file is empty."}), 400

    columns = _header_at(rows, header_row)
    if not columns:
        return jsonify({"error": "Header row is beyond the file content."}), 400

    return jsonify({
        "file": stored_name,
        "columns": columns,
        "preview": rows[:5],
    })


@app.route("/api/process", methods=["POST"])
# Why: the main action — the user picks a column (and optional values) and this
# processes the file, marks green/red borders, saves the output, and returns
# the stats + results table to the browser.
def api_process():
    data = request.get_json(force=True)
    filename = data.get("file", "")
    raw_columns = data.get("columns", [])
    if isinstance(raw_columns, str):
        raw_columns = [raw_columns]
    columns = [c for c in raw_columns if str(c).strip()]
    if not columns:
        return jsonify({"error": "Select at least one column to apply borders."}), 400
    header_row = int(data.get("header_row", 1) or 1)

    raw_present = data.get("present_values", "") or ""
    present_set = {
        _normalize(v) for v in str(raw_present).replace("，", ",").split(",") if _normalize(v)
    }

    stored_path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(stored_path):
        return jsonify({"error": "Uploaded file not found. Please upload again."}), 400

    ext = os.path.splitext(stored_path)[1].lower()
    output_name = os.path.splitext(filename)[0] + "_stamped" + (".pdf" if ext == ".pdf" else ".xlsx")
    output_path = os.path.join(OUTPUT_FOLDER, output_name)

    if ext == ".pdf":
        headers, results, stats = _pdf_process(stored_path, columns, present_set, output_path)
        if not results:
            return jsonify({"error": "No table / stamp column found in the PDF."}), 400
        stamp_indexes = []
        if headers:
            stamp_indexes = [i for i, h in enumerate(headers) if _normalize(h) in {_normalize(c) for c in columns}]
        return jsonify({
            "download": f"/download/{output_name}",
            "stats": stats,
            "headers": headers or [],
            "stamp_indexes": stamp_indexes,
            "results": results[:MAX_PREVIEW],
            "total": len(results),
        })

    rows = _read_rows(stored_path, ext)
    if not rows:
        return jsonify({"error": "The file is empty."}), 400

    headers = _header_at(rows, header_row)
    if not headers:
        return jsonify({"error": "Header row is beyond the file content."}), 400
    col_indexes = []
    for c in columns:
        for i, name in enumerate(headers):
            if _normalize(name) == _normalize(c):
                col_indexes.append(i)
                break
    if not col_indexes:
        return jsonify({"error": f"Column '{columns[0]}' not found in the header."}), 400

    ws_header_row = 1 if ext == ".csv" else header_row
    data_start = ws_header_row + 1
    ws_max_row = max(data_start, len(rows))

    if ext == ".csv":
        wb = Workbook()
        ws = wb.active
        for r in rows:
            ws.append(r)
    else:
        wb = load_workbook(stored_path, data_only=True)
        ws = wb.active

    stats = {"present": 0, "absent": 0, "empty": 0}
    results = []
    for offset in range(data_start, ws_max_row + 1):
        cells = []
        for c in range(len(headers)):
            v = ws.cell(row=offset, column=c + 1).value
            cells.append("" if v is None else str(v))

        images = {}
        col_present = {}
        for ci in col_indexes:
            cell = ws.cell(row=offset, column=ci + 1)
            present = _is_present(cell.value, present_set)
            cell.border = GREEN_BORDER if present else RED_BORDER
            if present:
                stats["present"] += 1
            else:
                stats["absent"] += 1
                if cell.value is None or _normalize(cell.value) == "":
                    stats["empty"] += 1
            col_present[str(ci)] = present
            images[str(ci)] = None

        present = any(col_present.values())

        first_text = ws.cell(row=offset, column=col_indexes[0] + 1).value
        results.append({
            "row": offset,
            "cells": cells,
            "value": "" if first_text is None else first_text,
            "images": images,
            "col_present": col_present,
            "present": present,
        })

    for ci in col_indexes:
        col_letter = get_column_letter(ci + 1)
        ws.column_dimensions[col_letter].width = max(ws.column_dimensions[col_letter].width or 10, 14)

    wb.save(output_path)

    return jsonify({
        "download": f"/download/{output_name}",
        "stats": stats,
        "headers": headers,
        "stamp_indexes": col_indexes,
        "results": results[:MAX_PREVIEW],
        "total": len(results),
    })


@app.route("/preview/<path:filename>")
# Why: serve the stamp-cell thumbnails and page previews so the browser can
# display them in the results table.
def preview(filename):
    return send_from_directory(PREVIEW_FOLDER, filename)


@app.route("/download/<path:filename>")
# Why: let the user download the final processed (green/red bordered) file.
def download(filename):
    return send_from_directory(OUTPUT_FOLDER, filename, as_attachment=True)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
