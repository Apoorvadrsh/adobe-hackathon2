import fitz  # PyMuPDF
import json
import time
import cProfile
import pstats
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
from collections import Counter
import re
import os
import logging
from typing import List, Dict, Optional

# Import the ML classifier
from .ml_heading_classifier import HeadingClassifier

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OutlineExtractorML:
    """
    ML-Enhanced Outline Extractor with machine learning-based heading classification.
    
    This version uses a trained ML model to classify text blocks as H1, H2, H3, or body text
    based on features like font size, bold/italic presence, indentation, vertical spacing, etc.
    Falls back to heuristic-based classification if ML model is not available.
    """
    
    def __init__(self, enable_profiling=False, ml_model_path=None, use_ml=True):
        self.enable_profiling = enable_profiling
        self.profiler = None
        self.use_ml = use_ml
        self.ml_classifier = None
        
        if enable_profiling:
            self.profiler = cProfile.Profile()
        
        # Initialize ML classifier
        if use_ml:
            self._initialize_ml_classifier(ml_model_path)

    def _initialize_ml_classifier(self, model_path=None):
        """Initialize the ML classifier."""
        try:
            if model_path and os.path.exists(model_path):
                logger.info(f"Loading ML model from {model_path}")
                self.ml_classifier = HeadingClassifier(model_path)
            else:
                # Try to find a pre-trained model in the same directory
                default_model_path = os.path.join(
                    os.path.dirname(__file__), 
                    "heading_classifier_model.joblib"
                )
                
                if os.path.exists(default_model_path):
                    logger.info(f"Loading default ML model from {default_model_path}")
                    self.ml_classifier = HeadingClassifier(default_model_path)
                else:
                    logger.info("No pre-trained model found. Training new model...")
                    self.ml_classifier = HeadingClassifier()
                    self.ml_classifier.train(use_synthetic=True)
                    
                    # Save the trained model
                    self.ml_classifier.save_model(default_model_path)
                    logger.info(f"New model trained and saved to {default_model_path}")
                    
        except Exception as e:
            logger.warning(f"Failed to initialize ML classifier: {e}")
            logger.warning("Falling back to heuristic-based classification")
            self.ml_classifier = None
            self.use_ml = False

    def extract_outline(self, pdf_path):
        """Enhanced outline extraction with ML-based heading classification."""
        if self.enable_profiling:
            self.profiler.enable()
        
        start_time = time.time()
        
        doc = fitz.open(pdf_path)
        
        # Detect document language
        language = self._detect_language(doc)
        
        # Extract components with ML-enhanced processing
        title = self._extract_title_enhanced(doc, language)
        headings = self._extract_headings_ml(doc, language)
        full_text_by_page = self._extract_full_text_by_page(doc)
        
        doc.close()
        
        processing_time = time.time() - start_time
        
        if self.enable_profiling:
            self.profiler.disable()
        
        result = {
            "title": title,
            "headings": headings,
            "full_text_by_page": full_text_by_page,
            "metadata": {
                "processing_time_seconds": processing_time,
                "detected_language": language,
                "total_pages": len(full_text_by_page),
                "total_headings": len(headings),
                "ml_classification_used": self.use_ml and self.ml_classifier is not None,
                "classification_method": "ML" if (self.use_ml and self.ml_classifier is not None) else "Heuristic"
            }
        }
        
        return result

    def _detect_language(self, doc):
        """Detect document language using first few pages."""
        sample_text = ""
        max_pages_to_sample = min(3, doc.page_count)
        
        for page_num in range(max_pages_to_sample):
            page = doc[page_num]
            page_text = page.get_text("text")
            sample_text += page_text[:1000]  # First 1000 chars per page
            if len(sample_text) > 2000:  # Enough sample
                break
        
        try:
            # Clean text for better detection
            clean_text = re.sub(r'[^\w\s]', ' ', sample_text)
            clean_text = re.sub(r'\s+', ' ', clean_text).strip()
            
            if len(clean_text) > 50:  # Minimum text for reliable detection
                return detect(clean_text)
        except (LangDetectException, Exception):
            pass
        
        return "en"  # Default to English

    def _extract_title_enhanced(self, doc, language):
        """Enhanced title extraction with better heuristics."""
        if doc.page_count == 0:
            return ""
        
        page = doc[0]
        blocks = page.get_text("dict")["blocks"]
        
        title_candidates = []
        
        for b in blocks:
            if b["type"] == 0:  # text block
                for line in b["lines"]:
                    for span in line["spans"]:
                        text = span["text"].strip()
                        if not text:
                            continue
                        
                        # Enhanced criteria for title detection
                        font_size = span["size"]
                        is_bold = "bold" in span["font"].lower()
                        is_large = font_size > 16
                        is_short = len(text) < 200  # Titles are usually short
                        is_upper_area = span["bbox"][1] < page.rect.height * 0.3  # Upper 30% of page
                        
                        # Language-specific adjustments
                        word_count = len(text.split())
                        is_reasonable_length = 2 <= word_count <= 20
                        
                        if is_large and is_short and is_upper_area and is_reasonable_length:
                            score = font_size
                            if is_bold:
                                score += 5
                            if span["bbox"][1] < page.rect.height * 0.15:  # Very top
                                score += 3
                            
                            title_candidates.append({
                                "text": text,
                                "score": score,
                                "font_size": font_size,
                                "y_position": span["bbox"][1]
                            })
        
        if title_candidates:
            # Sort by score and return the best candidate
            title_candidates.sort(key=lambda x: x["score"], reverse=True)
            return title_candidates[0]["text"]
        
        return ""

    def _extract_headings_ml(self, doc, language):
        """ML-enhanced heading extraction."""
        extracted_headings = []
        
        # Collect all text blocks from all pages
        all_text_blocks = []
        
        for page_num in range(doc.page_count):
            page = doc[page_num]
            blocks = page.get_text("dict")["blocks"]
            
            for b in blocks:
                if b["type"] == 0:  # text block
                    for line in b["lines"]:
                        for span in line["spans"]:
                            text = span["text"].strip()
                            if text:
                                all_text_blocks.append({
                                    "text": text,
                                    "font_size": span["size"],
                                    "font": span["font"],
                                    "bbox": span["bbox"],
                                    "page": page_num + 1,
                                    "is_bold": "bold" in span["font"].lower(),
                                    "is_italic": "italic" in span["font"].lower()
                                })
        
        if not all_text_blocks:
            return extracted_headings
        
        # Use ML classification if available
        if self.use_ml and self.ml_classifier is not None:
            try:
                predictions = self.ml_classifier.predict(all_text_blocks)
                probabilities = self.ml_classifier.predict_proba(all_text_blocks)
                
                for i, (block, prediction, proba) in enumerate(zip(all_text_blocks, predictions, probabilities)):
                    if prediction != 'body':  # It's a heading
                        # Get confidence score
                        confidence = max(proba)
                        
                        # Filter out low-confidence predictions
                        if confidence > 0.6:  # Confidence threshold
                            extracted_headings.append({
                                "text": block["text"],
                                "level": prediction,
                                "page": block["page"],
                                "font_size": block["font_size"],
                                "is_bold": block["is_bold"],
                                "confidence": float(confidence),
                                "classification_method": "ML"
                            })
                
                logger.info(f"ML classification: {len(extracted_headings)} headings found")
                
            except Exception as e:
                logger.warning(f"ML classification failed: {e}")
                logger.warning("Falling back to heuristic classification")
                return self._extract_headings_heuristic(all_text_blocks, language)
        
        else:
            # Fall back to heuristic classification
            return self._extract_headings_heuristic(all_text_blocks, language)
        
        return extracted_headings

    def _extract_headings_heuristic(self, all_text_blocks, language):
        """Heuristic-based heading extraction (fallback method)."""
        extracted_headings = []
        
        # Analyze font size distribution
        font_sizes = [block["font_size"] for block in all_text_blocks]
        font_size_counter = Counter(font_sizes)
        
        # Determine body text size (most common)
        body_text_size = font_size_counter.most_common(1)[0][0]
        
        # Identify heading sizes (larger than body text)
        unique_sizes = sorted(set(font_sizes), reverse=True)
        heading_sizes = [size for size in unique_sizes if size > body_text_size + 1]
        
        # Process each text block
        for i, block in enumerate(all_text_blocks):
            if self._is_heading_candidate_heuristic(block, heading_sizes, body_text_size, all_text_blocks, i, language):
                level = self._determine_heading_level_heuristic(block, heading_sizes)
                
                extracted_headings.append({
                    "text": block["text"],
                    "level": level,
                    "page": block["page"],
                    "font_size": block["font_size"],
                    "is_bold": block["is_bold"],
                    "confidence": 0.8,  # Default confidence for heuristic
                    "classification_method": "Heuristic"
                })
        
        logger.info(f"Heuristic classification: {len(extracted_headings)} headings found")
        return extracted_headings

    def _is_heading_candidate_heuristic(self, block, heading_sizes, body_text_size, all_blocks, block_index, language):
        """Determine if a text block is likely a heading using heuristics."""
        text = block["text"]
        font_size = block["font_size"]
        
        # Basic criteria
        is_large_enough = font_size > body_text_size + 0.5
        is_reasonable_length = 3 <= len(text) <= 150
        is_not_too_long = len(text.split()) <= 15
        
        # Font-based criteria
        is_bold = block["is_bold"]
        is_heading_size = font_size in heading_sizes
        
        # Position-based criteria
        has_vertical_space = self._has_adequate_vertical_spacing_heuristic(block, all_blocks, block_index)
        
        # Content-based criteria
        is_not_number_only = not text.replace(".", "").replace(",", "").isdigit()
        has_letters = any(c.isalpha() for c in text)
        
        # Language-specific adjustments
        if language in ["zh", "ja", "ko"]:  # CJK languages
            is_reasonable_length = 2 <= len(text) <= 100
        
        return (is_large_enough or is_bold) and is_reasonable_length and is_not_too_long and \
               has_vertical_space and is_not_number_only and has_letters and \
               (is_heading_size or is_bold)

    def _has_adequate_vertical_spacing_heuristic(self, block, all_blocks, block_index):
        """Check if block has adequate vertical spacing to be a heading."""
        if block_index == 0:
            return True
        
        current_y = block["bbox"][1]
        
        # Check spacing from previous block
        if block_index > 0:
            prev_block = all_blocks[block_index - 1]
            prev_y_bottom = prev_block["bbox"][3]
            gap_before = current_y - prev_y_bottom
            
            if gap_before < 3:  # Too close to previous text
                return False
        
        return True

    def _determine_heading_level_heuristic(self, block, heading_sizes):
        """Determine heading level based on font size and other factors."""
        font_size = block["font_size"]
        
        if not heading_sizes:
            return "H3"
        
        # Sort heading sizes in descending order
        sorted_sizes = sorted(heading_sizes, reverse=True)
        
        if font_size >= sorted_sizes[0] * 0.95:
            return "H1"
        elif len(sorted_sizes) > 1 and font_size >= sorted_sizes[1] * 0.95:
            return "H2"
        elif len(sorted_sizes) > 2 and font_size >= sorted_sizes[2] * 0.95:
            return "H3"
        else:
            return "H4"

    def _extract_full_text_by_page(self, doc):
        """Extract full text by page (unchanged from original)."""
        full_text_by_page = {}
        for page_num in range(doc.page_count):
            page = doc[page_num]
            full_text_by_page[page_num + 1] = page.get_text("text")
        return full_text_by_page

    def get_performance_stats(self):
        """Get performance profiling statistics."""
        if self.profiler:
            stats = pstats.Stats(self.profiler)
            return stats
        return None

    def validate_against_ground_truth(self, extracted_headings, ground_truth_headings):
        """Calculate precision and recall against ground truth."""
        if not ground_truth_headings:
            return {"precision": 0, "recall": 0, "f1": 0}
        
        # Simple text-based matching (can be enhanced)
        extracted_texts = set(h["text"].lower().strip() for h in extracted_headings)
        ground_truth_texts = set(h["text"].lower().strip() for h in ground_truth_headings)
        
        true_positives = len(extracted_texts.intersection(ground_truth_texts))
        false_positives = len(extracted_texts - ground_truth_texts)
        false_negatives = len(ground_truth_texts - extracted_texts)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives
        }

    def get_ml_model_info(self):
        """Get information about the ML model."""
        if self.ml_classifier is None:
            return {"status": "No ML model loaded"}
        
        info = {
            "status": "ML model loaded",
            "feature_count": len(self.ml_classifier.feature_names),
            "features": self.ml_classifier.feature_names,
            "classes": list(self.ml_classifier.label_mapping.keys())
        }
        
        # Get feature importance if available
        try:
            importance = self.ml_classifier.get_feature_importance()
            info["feature_importance"] = importance
        except:
            pass
        
        return info

if __name__ == "__main__":
    print("ML-Enhanced Outline Extractor with machine learning-based heading classification.")
    print("To test, ensure you have a PDF file and call OutlineExtractorML.extract_outline(pdf_path).")
    
    # Example usage
    extractor = OutlineExtractorML(use_ml=True)
    print(f"ML Model Info: {extractor.get_ml_model_info()}")

