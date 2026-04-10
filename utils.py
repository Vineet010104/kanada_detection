import cv2
import numpy as np
import pytesseract
import os

# PATHS
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
os.environ['TESSDATA_PREFIX'] = r"C:\Program Files\Tesseract-OCR\tessdata"


# 🔹 Kannada check
def is_kannada(text):
    return any('\u0C80' <= ch <= '\u0CFF' for ch in text)


# 🔹 MAIN FUNCTION (OCR-based detection)
def detect_kannada_text(image):
    h, w, _ = image.shape

    boxes = pytesseract.image_to_data(
        image,
        lang='kan+eng',
        output_type=pytesseract.Output.DICT
    )

    results = []

    for i in range(len(boxes['text'])):
        text = boxes['text'][i].strip()

        if text == "":
            continue

        x = boxes['left'][i]
        y = boxes['top'][i]
        bw = boxes['width'][i]
        bh = boxes['height'][i]

        if is_kannada(text):
            results.append((x, y, bw, bh, text))

    return results


# 🔹 Draw boxes
def draw_boxes(image, results):
    for (x, y, w, h, text) in results:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, "Kannada", (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image