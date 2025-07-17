import fitz  # PyMuPDF
import json

class OutlineExtractor:
    def __init__(self):
        pass

    def extract_outline(self, pdf_path):
        doc = fitz.open(pdf_path)
        title = self._extract_title(doc)
        headings = self._extract_headings(doc)
        full_text_by_page = self._extract_full_text_by_page(doc)
        doc.close()
        return {"title": title, "headings": headings, "full_text_by_page": full_text_by_page}

    def _extract_title(self, doc):
        # Heuristic: Find the largest text on the first page as a potential title
        if doc.page_count == 0: return ""
        page = doc[0]
        blocks = page.get_text("dict")["blocks"]
        
        max_font_size = 0
        title_text = ""

        for b in blocks:
            if b["type"] == 0:  # text block
                for line in b["lines"]:
                    for span in line["spans"]:
                        # Consider text that is significantly larger than typical body text
                        if span["size"] > max_font_size and span["size"] > 18: # Assuming body text is usually < 12-14
                            max_font_size = span["size"]
                            title_text = span["text"]
                        elif span["size"] == max_font_size and len(span["text"]) > len(title_text):
                            title_text = span["text"]
        return title_text.strip()

    def _extract_headings(self, doc):
        extracted_headings = []
        for page_num in range(doc.page_count):
            page = doc[page_num]
            blocks = page.get_text("dict")["blocks"]
            
            # Collect all spans with their properties
            spans_on_page = []
            for b in blocks:
                if b["type"] == 0: # text block
                    for line in b["lines"]:
                        for span in line["spans"]:
                            spans_on_page.append({
                                "text": span["text"].strip(),
                                "size": span["size"],
                                "font": span["font"],
                                "bbox": span["bbox"],
                                "origin": span["origin"],
                                "page": page_num + 1
                            })
            
            # Sort spans by their vertical position to process them in order
            spans_on_page.sort(key=lambda x: x["bbox"][1]) # Sort by y0 coordinate

            # Heuristic-based heading detection
            # This is a more refined heuristic, but still a simplification
            # A real solution would involve more complex analysis of visual cues and spatial relationships
            
            # Simple approach: identify distinct font sizes and bold text
            unique_font_sizes = sorted(list(set([s["size"] for s in spans_on_page])), reverse=True)
            
            # Assuming the largest few font sizes are for headings
            # This needs to be adaptive based on document structure
            heading_sizes = []
            if len(unique_font_sizes) > 0: heading_sizes.append(unique_font_sizes[0])
            if len(unique_font_sizes) > 1: heading_sizes.append(unique_font_sizes[1])
            if len(unique_font_sizes) > 2: heading_sizes.append(unique_font_sizes[2])

            for i, span in enumerate(spans_on_page):
                is_bold = "bold" in span["font"].lower()
                is_large_enough = span["size"] > 12 # General threshold
                is_potential_heading = False

                if span["size"] in heading_sizes and is_large_enough:
                    is_potential_heading = True
                elif is_bold and is_large_enough:
                    is_potential_heading = True
                
                # Further refine: check for short lines, distinct vertical spacing
                if is_potential_heading and len(span["text"]) < 100: # Not too long
                    # Check vertical spacing (simplified: just look at previous line's y1)
                    if i > 0:
                        prev_span = spans_on_page[i-1]
                        vertical_gap = span["bbox"][1] - prev_span["bbox"][3] # current y0 - previous y1
                        if vertical_gap < 5: # If too close, might be part of a paragraph
                            is_potential_heading = False

                if is_potential_heading:
                    # Determine level based on font size relative to other headings
                    level = "H3" # Default to H3
                    if span["size"] >= unique_font_sizes[0] * 0.9: # Close to largest font
                        level = "H1"
                    elif len(unique_font_sizes) > 1 and span["size"] >= unique_font_sizes[1] * 0.9:
                        level = "H2"
                    
                    extracted_headings.append({
                        "text": span["text"].strip(),
                        "level": level,
                        "page": span["page"]
                    })
        return extracted_headings

    def _extract_full_text_by_page(self, doc):
        full_text_by_page = {}
        for page_num in range(doc.page_count):
            page = doc[page_num]
            full_text_by_page[page_num + 1] = page.get_text("text")
        return full_text_by_page

if __name__ == "__main__":
    print("This module is part of the project and should be run via main.py or integrated for testing.")
    print("To test, ensure you have a PDF file and call OutlineExtractor.extract_outline(pdf_path).")


