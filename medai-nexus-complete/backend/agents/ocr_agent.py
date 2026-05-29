"""
MedAI Nexus — OCR Agent
Extracts text from medical documents:
  PDF → PyMuPDF (text layer) → Tesseract fallback for scanned pages
  Images → OpenCV preprocessing → Tesseract OCR
"""
from __future__ import annotations
import asyncio, io, logging, os, re
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
import pytesseract
import fitz  # PyMuPDF

logger = logging.getLogger("medai.ocr_agent")


class OCRAgent:
    """
    Autonomous OCR agent that:
    1. Detects file type (PDF / image)
    2. Applies optimal preprocessing (deskew, denoise, threshold)
    3. Extracts text with confidence scoring
    4. Cleans and normalises medical text
    """

    MEDICAL_PATTERNS = {
        "hemoglobin":   r"(?:hgb|hemoglobin|haemoglobin)\s*[:=]?\s*([\d.]+)\s*(g/dl|g%)?",
        "wbc":          r"(?:wbc|white blood cell|leucocyte)\s*[:=]?\s*([\d,]+)\s*(/[μu]l|/mm3)?",
        "blood_sugar":  r"(?:glucose|blood sugar|fbs|ppbs|rbs)\s*[:=]?\s*([\d.]+)\s*(mg/dl)?",
        "blood_pressure": r"(?:bp|blood pressure)\s*[:=]?\s*([\d]+)\s*/\s*([\d]+)\s*(mmhg)?",
        "creatinine":   r"(?:creatinine|creat)\s*[:=]?\s*([\d.]+)\s*(mg/dl)?",
        "cholesterol":  r"(?:total cholesterol|cholesterol)\s*[:=]?\s*([\d.]+)\s*(mg/dl)?",
    }

    async def extract(self, file_path: str) -> Dict:
        """Main entry point — auto-routes to PDF or image pipeline."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = path.suffix.lower()
        if ext == ".pdf":
            result = await asyncio.to_thread(self._extract_pdf, str(path))
        else:
            result = await asyncio.to_thread(self._extract_image, str(path))

        result["parsed_values"] = self._parse_medical_values(result["text"])
        logger.info(f"[OCR] Extracted {len(result['text'])} chars | confidence={result['confidence']:.2f}")
        return result

    # ── PDF Pipeline ───────────────────────────
    def _extract_pdf(self, path: str) -> Dict:
        doc = fitz.open(path)
        full_text, confidences = [], []

        for page_num, page in enumerate(doc):
            text = page.get_text("text").strip()
            if len(text) > 50:
                # Native text layer available
                full_text.append(text)
                confidences.append(0.98)
            else:
                # Scanned page — rasterize and OCR
                mat  = fitz.Matrix(2.5, 2.5)    # 2.5× zoom for clarity
                pix  = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY)
                img  = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w)
                img  = self._preprocess_image(img)
                data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT,
                                                 config="--psm 6 --oem 3")
                page_text, conf_vals = self._extract_from_data(data)
                full_text.append(page_text)
                if conf_vals:
                    confidences.append(sum(conf_vals) / len(conf_vals) / 100)

        doc.close()
        return {
            "text": self._clean_text("\n\n".join(full_text)),
            "confidence": sum(confidences) / len(confidences) if confidences else 0.0,
            "pages": len(full_text),
        }

    # ── Image Pipeline ─────────────────────────
    def _extract_image(self, path: str) -> Dict:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not read image: {path}")
        img = self._preprocess_image(img)
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT,
                                         config="--psm 6 --oem 3")
        text, conf_vals = self._extract_from_data(data)
        conf = sum(conf_vals) / len(conf_vals) / 100 if conf_vals else 0.0
        return {
            "text": self._clean_text(text),
            "confidence": conf,
            "pages": 1,
        }

    # ── Image Preprocessing ────────────────────
    def _preprocess_image(self, img: np.ndarray) -> np.ndarray:
        # 1. Denoise
        img = cv2.fastNlMeansDenoising(img, h=10)
        # 2. Deskew
        img = self._deskew(img)
        # 3. Adaptive threshold (handles variable lighting)
        img = cv2.adaptiveThreshold(img, 255,
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)
        # 4. Dilate to connect broken characters
        kernel = np.ones((1, 1), np.uint8)
        img = cv2.dilate(img, kernel, iterations=1)
        return img

    def _deskew(self, img: np.ndarray) -> np.ndarray:
        coords = np.column_stack(np.where(img < 128))
        if len(coords) < 5:
            return img
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = 90 + angle
        if abs(angle) < 0.5:
            return img
        h, w = img.shape
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)

    # ── Text Extraction Helpers ────────────────
    def _extract_from_data(self, data: Dict):
        text_parts, conf_vals = [], []
        for i, word in enumerate(data["text"]):
            if word.strip():
                text_parts.append(word)
                c = data["conf"][i]
                if isinstance(c, int) and c > 0:
                    conf_vals.append(c)
        return " ".join(text_parts), conf_vals

    def _clean_text(self, text: str) -> str:
        text = re.sub(r"\s{3,}", "\n", text)
        text = re.sub(r"[^\w\s\.\,\:\;\-\/\(\)\+\%\@]", " ", text)
        return text.strip()

    def _parse_medical_values(self, text: str) -> Dict:
        found = {}
        lower = text.lower()
        for key, pattern in self.MEDICAL_PATTERNS.items():
            match = re.search(pattern, lower)
            if match:
                found[key] = match.group(1)
        return found
