"""
Validation utilities for testing the enhanced Adobe Hackathon project.
"""

import json
import os
import time
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import numpy as np

class ValidationUtils:
    """Utilities for validating and testing the enhanced project."""
    
    @staticmethod
    def create_ground_truth_template(pdf_path: str, output_path: str):
        """Create a ground truth template for manual annotation."""
        template = {
            "pdf_file": os.path.basename(pdf_path),
            "title": "MANUALLY_ANNOTATE_TITLE_HERE",
            "headings": [
                {
                    "text": "MANUALLY_ANNOTATE_HEADING_TEXT",
                    "level": "H1_OR_H2_OR_H3",
                    "page": 1,
                    "notes": "Add any notes about this heading"
                }
            ],
            "notes": "This is a template for manual ground truth annotation. Replace the placeholder values with actual headings from the PDF."
        }
        
        with open(output_path, 'w') as f:
            json.dump(template, f, indent=4)
        
        print(f"Ground truth template created at {output_path}")
        print("Please manually annotate the headings and save the file.")

    @staticmethod
    def compare_extractors(pdf_paths: List[str], rounds: int = 3):
        """Compare performance between original and enhanced extractors."""
        from round1a.outline_extractor import OutlineExtractor
        from round1a.outline_extractor_enhanced import OutlineExtractorEnhanced
        
        results = {
            "original": {"times": [], "avg_headings": []},
            "enhanced": {"times": [], "avg_headings": []}
        }
        
        print("Comparing Original vs Enhanced Outline Extractors...")
        
        for round_num in range(rounds):
            print(f"\nRound {round_num + 1}/{rounds}")
            
            # Test original extractor
            start_time = time.time()
            original_extractor = OutlineExtractor()
            original_headings = []
            
            for pdf_path in pdf_paths:
                outline = original_extractor.extract_outline(pdf_path)
                original_headings.append(len(outline["headings"]))
            
            original_time = time.time() - start_time
            results["original"]["times"].append(original_time)
            results["original"]["avg_headings"].append(np.mean(original_headings))
            
            print(f"  Original: {original_time:.2f}s, Avg headings: {np.mean(original_headings):.1f}")
            
            # Test enhanced extractor
            start_time = time.time()
            enhanced_extractor = OutlineExtractorEnhanced()
            enhanced_headings = []
            
            for pdf_path in pdf_paths:
                outline = enhanced_extractor.extract_outline(pdf_path)
                enhanced_headings.append(len(outline["headings"]))
            
            enhanced_time = time.time() - start_time
            results["enhanced"]["times"].append(enhanced_time)
            results["enhanced"]["avg_headings"].append(np.mean(enhanced_headings))
            
            print(f"  Enhanced: {enhanced_time:.2f}s, Avg headings: {np.mean(enhanced_headings):.1f}")
        
        # Calculate averages
        orig_avg_time = np.mean(results["original"]["times"])
        enh_avg_time = np.mean(results["enhanced"]["times"])
        
        print(f"\nSummary:")
        print(f"  Original average time: {orig_avg_time:.2f}s")
        print(f"  Enhanced average time: {enh_avg_time:.2f}s")
        print(f"  Time difference: {((enh_avg_time - orig_avg_time) / orig_avg_time * 100):+.1f}%")
        
        return results

    @staticmethod
    def test_multilingual_support(test_pdf_path: str):
        """Test multilingual support of the enhanced extractor."""
        from round1a.outline_extractor_enhanced import OutlineExtractorEnhanced
        
        print("Testing multilingual support...")
        
        extractor = OutlineExtractorEnhanced()
        outline = extractor.extract_outline(test_pdf_path)
        
        if "metadata" in outline:
            detected_lang = outline["metadata"]["detected_language"]
            print(f"Detected language: {detected_lang}")
            print(f"Total headings: {outline['metadata']['total_headings']}")
            print(f"Processing time: {outline['metadata']['processing_time_seconds']:.2f}s")
            
            print("\nExtracted headings:")
            for heading in outline["headings"][:5]:  # Show first 5
                print(f"  {heading['level']}: {heading['text']}")
        
        return outline

    @staticmethod
    def test_persona_variations(pdf_paths: List[str], persona_file: str):
        """Test different job-to-be-done scenarios with the same persona."""
        from round1a.outline_extractor_enhanced import OutlineExtractorEnhanced
        from round1b.document_analyst_enhanced import DocumentAnalystEnhanced
        
        test_jobs = [
            "Find information about artificial intelligence applications",
            "Locate technical implementation details",
            "Identify business opportunities and market insights",
            "Extract research methodologies and findings",
            "Find regulatory and compliance information"
        ]
        
        extractor = OutlineExtractorEnhanced()
        results = {}
        
        print("Testing different job-to-be-done scenarios...")
        
        for job in test_jobs:
            print(f"\nTesting: {job}")
            
            analyst = DocumentAnalystEnhanced(persona_file, job)
            analysis = analyst.analyze_documents(pdf_paths, extractor)
            
            top_sections = analysis["extracted_sections"][:3]
            results[job] = {
                "processing_time": analysis["metadata"]["processing_time_seconds"],
                "top_sections": [
                    {
                        "title": section["title"],
                        "relevance_score": section.get("relevance_score", 0)
                    }
                    for section in top_sections
                ]
            }
            
            print(f"  Processing time: {results[job]['processing_time']:.2f}s")
            print(f"  Top section: {top_sections[0]['title'] if top_sections else 'None'}")
        
        return results

    @staticmethod
    def generate_performance_report(results: Dict[str, Any], output_path: str):
        """Generate a performance report with visualizations."""
        try:
            import matplotlib.pyplot as plt
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
            
            # Time comparison
            if "original" in results and "enhanced" in results:
                times_orig = results["original"]["times"]
                times_enh = results["enhanced"]["times"]
                
                ax1.boxplot([times_orig, times_enh], labels=["Original", "Enhanced"])
                ax1.set_title("Processing Time Comparison")
                ax1.set_ylabel("Time (seconds)")
                
                # Heading count comparison
                headings_orig = results["original"]["avg_headings"]
                headings_enh = results["enhanced"]["avg_headings"]
                
                ax2.boxplot([headings_orig, headings_enh], labels=["Original", "Enhanced"])
                ax2.set_title("Average Headings Detected")
                ax2.set_ylabel("Number of Headings")
            
            # Performance over rounds
            if "original" in results:
                rounds = range(1, len(results["original"]["times"]) + 1)
                ax3.plot(rounds, results["original"]["times"], 'o-', label="Original")
                ax3.plot(rounds, results["enhanced"]["times"], 's-', label="Enhanced")
                ax3.set_title("Performance Over Test Rounds")
                ax3.set_xlabel("Round")
                ax3.set_ylabel("Time (seconds)")
                ax3.legend()
            
            # Summary statistics
            ax4.axis('off')
            if "original" in results and "enhanced" in results:
                orig_avg = np.mean(results["original"]["times"])
                enh_avg = np.mean(results["enhanced"]["times"])
                improvement = ((orig_avg - enh_avg) / orig_avg * 100)
                
                summary_text = f"""
Performance Summary:

Original Extractor:
  Avg Time: {orig_avg:.2f}s
  Std Dev: {np.std(results["original"]["times"]):.2f}s

Enhanced Extractor:
  Avg Time: {enh_avg:.2f}s
  Std Dev: {np.std(results["enhanced"]["times"]):.2f}s

Performance Change: {improvement:+.1f}%
                """
                ax4.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center')
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Performance report saved to {output_path}")
            
        except ImportError:
            print("Matplotlib not available. Skipping visualization.")
            
            # Create text report instead
            text_output = output_path.replace('.png', '.txt')
            with open(text_output, 'w') as f:
                f.write("Performance Report\n")
                f.write("==================\n\n")
                
                if "original" in results and "enhanced" in results:
                    orig_avg = np.mean(results["original"]["times"])
                    enh_avg = np.mean(results["enhanced"]["times"])
                    
                    f.write(f"Original Extractor Average Time: {orig_avg:.2f}s\n")
                    f.write(f"Enhanced Extractor Average Time: {enh_avg:.2f}s\n")
                    f.write(f"Performance Change: {((orig_avg - enh_avg) / orig_avg * 100):+.1f}%\n")
            
            print(f"Text report saved to {text_output}")

    @staticmethod
    def run_comprehensive_test_suite(input_dir: str, output_dir: str, persona_file: str):
        """Run a comprehensive test suite for the enhanced project."""
        print("Running Comprehensive Test Suite")
        print("=" * 50)
        
        # Find PDF files
        pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".pdf")]
        pdf_paths = [os.path.join(input_dir, f) for f in pdf_files]
        
        if not pdf_paths:
            print("No PDF files found for testing.")
            return
        
        results = {}
        
        # Test 1: Extractor comparison
        print("\n1. Testing Extractor Performance...")
        results["extractor_comparison"] = ValidationUtils.compare_extractors(pdf_paths)
        
        # Test 2: Multilingual support (if multilingual PDF exists)
        multilingual_pdf = os.path.join(input_dir, "multilingual_test.pdf")
        if os.path.exists(multilingual_pdf):
            print("\n2. Testing Multilingual Support...")
            results["multilingual_test"] = ValidationUtils.test_multilingual_support(multilingual_pdf)
        
        # Test 3: Persona variations
        if os.path.exists(persona_file):
            print("\n3. Testing Persona Variations...")
            results["persona_variations"] = ValidationUtils.test_persona_variations(pdf_paths, persona_file)
        
        # Generate report
        report_path = os.path.join(output_dir, "performance_report.png")
        ValidationUtils.generate_performance_report(results["extractor_comparison"], report_path)
        
        # Save full results
        results_path = os.path.join(output_dir, "test_results.json")
        with open(results_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_results = {}
            for key, value in results.items():
                if key == "extractor_comparison":
                    json_results[key] = {
                        "original": {
                            "times": [float(t) for t in value["original"]["times"]],
                            "avg_headings": [float(h) for h in value["original"]["avg_headings"]]
                        },
                        "enhanced": {
                            "times": [float(t) for t in value["enhanced"]["times"]],
                            "avg_headings": [float(h) for h in value["enhanced"]["avg_headings"]]
                        }
                    }
                else:
                    json_results[key] = value
            
            json.dump(json_results, f, indent=4)
        
        print(f"\nTest results saved to {results_path}")
        print("Comprehensive test suite completed!")

if __name__ == "__main__":
    print("Validation utilities for the Adobe Hackathon project.")
    print("Use this module through main_enhanced.py or import specific functions.")

