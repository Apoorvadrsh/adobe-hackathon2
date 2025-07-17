import os
import json
import argparse
from round1a.outline_extractor import OutlineExtractor
from round1b.document_analyst import DocumentAnalyst
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

def create_dummy_pdf(filename="dummy.pdf"):
    c = canvas.Canvas(filename, pagesize=letter)
    c.setFont("Helvetica-Bold", 24)
    c.drawString(100, 750, "My Awesome Document Title")
    c.setFont("Helvetica-Bold", 18)
    c.drawString(100, 700, "1. Introduction to AI")
    c.setFont("Helvetica", 12)
    c.drawString(120, 680, "This section provides an overview of Artificial Intelligence.")
    c.drawString(120, 660, "AI is transforming various industries.")
    c.setFont("Helvetica-Bold", 16)
    c.drawString(120, 630, "1.1 Machine Learning Basics")
    c.setFont("Helvetica", 12)
    c.drawString(140, 610, "Machine learning is a subset of AI that enables systems to learn from data.")
    c.setFont("Helvetica-Bold", 16)
    c.drawString(120, 580, "1.2 Deep Learning Advances")
    c.setFont("Helvetica", 12)
    c.drawString(140, 560, "Deep learning, a subfield of machine learning, uses neural networks.")

    c.showPage()
    c.setFont("Helvetica-Bold", 18)
    c.drawString(100, 750, "2. Applications of AI")
    c.setFont("Helvetica", 12)
    c.drawString(120, 730, "AI has numerous applications across different sectors.")
    c.setFont("Helvetica-Bold", 16)
    c.drawString(120, 700, "2.1 AI in Healthcare")
    c.setFont("Helvetica", 12)
    c.drawString(140, 680, "From diagnostics to drug discovery, AI is revolutionizing healthcare.")
    c.setFont("Helvetica-Bold", 16)
    c.drawString(120, 650, "2.2 AI in Finance")
    c.setFont("Helvetica", 12)
    c.drawString(140, 630, "AI helps in fraud detection, algorithmic trading, and risk assessment.")
    c.save()
    print(f"Dummy PDF created at {filename}")

def main():
    parser = argparse.ArgumentParser(description="Adobe India Hackathon Project")
    parser.add_argument("--input_dir", type=str, default="/app/input",
                        help="Input directory containing PDF files.")
    parser.add_argument("--output_dir", type=str, default="/app/output",
                        help="Output directory for JSON results.")
    parser.add_argument("--round", type=str, choices=["1A", "1B"], required=True,
                        help="Specify which round to execute (1A or 1B).")
    parser.add_argument("--persona_file", type=str, 
                        help="Path to persona definition file for Round 1B.")
    parser.add_argument("--job_to_be_done", type=str, 
                        help="Job-to-be-done description for Round 1B.")
    parser.add_argument("--create_dummy_pdf", action="store_true",
                        help="Create a dummy PDF for testing purposes.")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.input_dir, exist_ok=True)

    if args.create_dummy_pdf:
        create_dummy_pdf(os.path.join(args.input_dir, "dummy.pdf"))
        return

    pdf_files = [f for f in os.listdir(args.input_dir) if f.lower().endswith(".pdf")]

    if args.round == "1A":
        print(f"Executing Round 1A: Outline Extraction for PDFs in {args.input_dir}")
        extractor = OutlineExtractor()
        for pdf_file in pdf_files:
            pdf_path = os.path.join(args.input_dir, pdf_file)
            output_filename = os.path.splitext(pdf_file)[0] + ".json"
            output_path = os.path.join(args.output_dir, output_filename)
            
            print(f"Processing {pdf_file}...")
            outline = extractor.extract_outline(pdf_path)
            with open(output_path, "w") as f:
                json.dump(outline, f, indent=4)
            print(f"Outline saved to {output_path}")

    elif args.round == "1B":
        if not args.persona_file or not args.job_to_be_done:
            print("Error: --persona_file and --job_to_be_done are required for Round 1B.")
            return

        print(f"Executing Round 1B: Persona-Driven Document Analysis for PDFs in {args.input_dir}")
        extractor = OutlineExtractor() # Round 1A extractor is used by Round 1B
        analyst = DocumentAnalyst(args.persona_file, args.job_to_be_done)
        
        pdf_paths_full = [os.path.join(args.input_dir, f) for f in pdf_files]
        results = analyst.analyze_documents(pdf_paths_full, extractor)
        
        output_path = os.path.join(args.output_dir, "round1b_analysis.json")
        with open(output_path, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Analysis saved to {output_path}")

if __name__ == "__main__":
    main()


