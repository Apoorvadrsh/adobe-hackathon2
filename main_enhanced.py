import os
import json
import argparse
import time
from round1a.outline_extractor import OutlineExtractor
from round1a.outline_extractor_enhanced import OutlineExtractorEnhanced
from round1a.outline_extractor_ml import OutlineExtractorML
from round1b.document_analyst import DocumentAnalyst
from round1b.document_analyst_enhanced import DocumentAnalystEnhanced
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

def create_dummy_pdf(filename="dummy.pdf"):
    """Create a dummy PDF for testing purposes."""
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

def create_multilingual_test_pdf(filename="multilingual_test.pdf"):
    """Create a multilingual test PDF."""
    c = canvas.Canvas(filename, pagesize=letter)
    
    # English content
    c.setFont("Helvetica-Bold", 20)
    c.drawString(100, 750, "Multilingual Document Test")
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, 720, "1. English Section")
    c.setFont("Helvetica", 12)
    c.drawString(120, 700, "This is an English paragraph about artificial intelligence.")
    
    # Spanish content
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, 670, "2. Sección en Español")
    c.setFont("Helvetica", 12)
    c.drawString(120, 650, "Esta es una sección en español sobre inteligencia artificial.")
    
    # French content
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, 620, "3. Section Française")
    c.setFont("Helvetica", 12)
    c.drawString(120, 600, "Ceci est une section française sur l'intelligence artificielle.")
    
    c.save()
    print(f"Multilingual test PDF created at {filename}")

def benchmark_performance(pdf_paths, extractor_class, rounds=3):
    """Benchmark the performance of an extractor."""
    print(f"Benchmarking {extractor_class.__name__}...")
    
    times = []
    for round_num in range(rounds):
        start_time = time.time()
        
        extractor = extractor_class()
        for pdf_path in pdf_paths:
            extractor.extract_outline(pdf_path)
        
        end_time = time.time()
        times.append(end_time - start_time)
        print(f"Round {round_num + 1}: {times[-1]:.2f} seconds")
    
    avg_time = sum(times) / len(times)
    print(f"Average time: {avg_time:.2f} seconds")
    print(f"Time per PDF: {avg_time / len(pdf_paths):.2f} seconds")
    
    return avg_time

def validate_heading_accuracy(pdf_path, extractor, ground_truth_file=None):
    """Validate heading detection accuracy against ground truth."""
    if not ground_truth_file or not os.path.exists(ground_truth_file):
        print("No ground truth file provided or file doesn't exist. Skipping validation.")
        return None
    
    with open(ground_truth_file, 'r') as f:
        ground_truth = json.load(f)
    
    extracted = extractor.extract_outline(pdf_path)
    
    if hasattr(extractor, 'validate_against_ground_truth'):
        metrics = extractor.validate_against_ground_truth(
            extracted["headings"], 
            ground_truth.get("headings", [])
        )
        print(f"Validation Results:")
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall: {metrics['recall']:.3f}")
        print(f"  F1 Score: {metrics['f1']:.3f}")
        return metrics
    
    return None

def main():
    parser = argparse.ArgumentParser(description="Adobe India Hackathon Project - Enhanced Version")
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
    parser.add_argument("--create_multilingual_pdf", action="store_true",
                        help="Create a multilingual test PDF.")
    parser.add_argument("--use_enhanced", action="store_true",
                        help="Use enhanced versions of extractors/analysts.")
    parser.add_argument("--use_ml", action="store_true",
                        help="Use ML-based heading classification (requires --use_enhanced).")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run performance benchmarks.")
    parser.add_argument("--validate", type=str,
                        help="Path to ground truth file for validation.")
    parser.add_argument("--enable_profiling", action="store_true",
                        help="Enable performance profiling.")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.input_dir, exist_ok=True)

    if args.create_dummy_pdf:
        create_dummy_pdf(os.path.join(args.input_dir, "dummy.pdf"))
        return

    if args.create_multilingual_pdf:
        create_multilingual_test_pdf(os.path.join(args.input_dir, "multilingual_test.pdf"))
        return

    pdf_files = [f for f in os.listdir(args.input_dir) if f.lower().endswith(".pdf")]
    
    if not pdf_files:
        print(f"No PDF files found in {args.input_dir}")
        return

    if args.round == "1A":
        print(f"Executing Round 1A: Outline Extraction for PDFs in {args.input_dir}")
        
        # Choose extractor based on enhanced and ML flags
        if args.use_enhanced and args.use_ml:
            extractor = OutlineExtractorML(enable_profiling=args.enable_profiling, use_ml=True)
            print("Using ML-Enhanced Outline Extractor")
        elif args.use_enhanced:
            extractor = OutlineExtractorEnhanced(enable_profiling=args.enable_profiling)
            print("Using Enhanced Outline Extractor")
        else:
            extractor = OutlineExtractor()
            print("Using Original Outline Extractor")
        
        # Benchmark if requested
        if args.benchmark:
            pdf_paths = [os.path.join(args.input_dir, f) for f in pdf_files]
            if args.use_enhanced and args.use_ml:
                benchmark_performance(pdf_paths, OutlineExtractorML)
            elif args.use_enhanced:
                benchmark_performance(pdf_paths, OutlineExtractorEnhanced)
            else:
                benchmark_performance(pdf_paths, OutlineExtractor)
        
        # Process PDFs
        for pdf_file in pdf_files:
            pdf_path = os.path.join(args.input_dir, pdf_file)
            
            # Determine output filename based on version used
            if args.use_enhanced and args.use_ml:
                output_filename = os.path.splitext(pdf_file)[0] + "_ml.json"
            elif args.use_enhanced:
                output_filename = os.path.splitext(pdf_file)[0] + "_enhanced.json"
            else:
                output_filename = os.path.splitext(pdf_file)[0] + ".json"
            
            output_path = os.path.join(args.output_dir, output_filename)
            
            print(f"Processing {pdf_file}...")
            start_time = time.time()
            outline = extractor.extract_outline(pdf_path)
            processing_time = time.time() - start_time
            
            print(f"  Processing time: {processing_time:.2f} seconds")
            if args.use_enhanced and "metadata" in outline:
                print(f"  Detected language: {outline['metadata']['detected_language']}")
                print(f"  Total headings: {outline['metadata']['total_headings']}")
                if args.use_ml and "classification_method" in outline['metadata']:
                    print(f"  Classification method: {outline['metadata']['classification_method']}")
                    if hasattr(extractor, 'get_ml_model_info'):
                        ml_info = extractor.get_ml_model_info()
                        if ml_info.get('status') == 'ML model loaded':
                            print(f"  ML model features: {len(ml_info.get('features', []))}")
            
            with open(output_path, "w") as f:
                json.dump(outline, f, indent=4)
            print(f"  Outline saved to {output_path}")
            
            # Validate if ground truth provided
            if args.validate:
                validate_heading_accuracy(pdf_path, extractor, args.validate)

    elif args.round == "1B":
        if not args.persona_file or not args.job_to_be_done:
            print("Error: --persona_file and --job_to_be_done are required for Round 1B.")
            return

        print(f"Executing Round 1B: Persona-Driven Document Analysis for PDFs in {args.input_dir}")
        
        # Choose components based on enhanced and ML flags
        if args.use_enhanced and args.use_ml:
            extractor = OutlineExtractorML()
            analyst = DocumentAnalystEnhanced(args.persona_file, args.job_to_be_done)
            print("Using ML-Enhanced Outline Extractor with Enhanced Document Analyst")
        elif args.use_enhanced:
            extractor = OutlineExtractorEnhanced()
            analyst = DocumentAnalystEnhanced(args.persona_file, args.job_to_be_done)
            print("Using Enhanced Document Analyst")
        else:
            extractor = OutlineExtractor()
            analyst = DocumentAnalyst(args.persona_file, args.job_to_be_done)
            print("Using Original Document Analyst")
        
        pdf_paths_full = [os.path.join(args.input_dir, f) for f in pdf_files]
        
        print(f"Analyzing {len(pdf_paths_full)} documents...")
        start_time = time.time()
        results = analyst.analyze_documents(pdf_paths_full, extractor)
        total_time = time.time() - start_time
        
        print(f"Total analysis time: {total_time:.2f} seconds")
        
        if args.use_enhanced and args.use_ml:
            output_filename = "round1b_analysis_ml.json"
        elif args.use_enhanced:
            output_filename = "round1b_analysis_enhanced.json"
        else:
            output_filename = "round1b_analysis.json"
        
        output_path = os.path.join(args.output_dir, output_filename)
        
        with open(output_path, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Analysis saved to {output_path}")
        
        # Print summary if using enhanced version
        if args.use_enhanced and hasattr(analyst, 'get_analysis_summary'):
            summary = analyst.get_analysis_summary(results)
            print("\nAnalysis Summary:")
            print(f"  Total sections analyzed: {summary['total_sections_analyzed']}")
            print(f"  Processing time: {summary['processing_time']:.2f} seconds")
            print(f"  Persona role: {summary['persona_role']}")
            print(f"  Top 3 relevant sections:")
            for i, section in enumerate(summary['top_5_sections'][:3]):
                print(f"    {i+1}. {section['title']} (Score: {section['relevance_score']})")

if __name__ == "__main__":
    main()

