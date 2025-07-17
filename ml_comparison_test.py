"""
Comprehensive comparison test for Original, Enhanced, and ML-Enhanced outline extractors.
"""

import time
import json
import os
from round1a.outline_extractor import OutlineExtractor
from round1a.outline_extractor_enhanced import OutlineExtractorEnhanced
from round1a.outline_extractor_ml import OutlineExtractorML

def compare_all_extractors(pdf_path, output_dir="./output"):
    """
    Compare all three extractor versions on a single PDF.
    
    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save comparison results
    """
    print(f"Comparing extractors on: {os.path.basename(pdf_path)}")
    print("=" * 60)
    
    results = {}
    
    # Test Original Extractor
    print("1. Testing Original Extractor...")
    start_time = time.time()
    original_extractor = OutlineExtractor()
    original_result = original_extractor.extract_outline(pdf_path)
    original_time = time.time() - start_time
    
    results['original'] = {
        'processing_time': original_time,
        'total_headings': len(original_result['headings']),
        'headings': original_result['headings'],
        'title': original_result['title']
    }
    
    print(f"   Time: {original_time:.3f}s")
    print(f"   Headings found: {len(original_result['headings'])}")
    print(f"   Title: {original_result['title']}")
    
    # Test Enhanced Extractor
    print("\n2. Testing Enhanced Extractor...")
    start_time = time.time()
    enhanced_extractor = OutlineExtractorEnhanced()
    enhanced_result = enhanced_extractor.extract_outline(pdf_path)
    enhanced_time = time.time() - start_time
    
    results['enhanced'] = {
        'processing_time': enhanced_time,
        'total_headings': len(enhanced_result['headings']),
        'headings': enhanced_result['headings'],
        'title': enhanced_result['title'],
        'detected_language': enhanced_result['metadata']['detected_language']
    }
    
    print(f"   Time: {enhanced_time:.3f}s")
    print(f"   Headings found: {len(enhanced_result['headings'])}")
    print(f"   Title: {enhanced_result['title']}")
    print(f"   Language: {enhanced_result['metadata']['detected_language']}")
    
    # Test ML-Enhanced Extractor
    print("\n3. Testing ML-Enhanced Extractor...")
    start_time = time.time()
    ml_extractor = OutlineExtractorML(use_ml=True)
    ml_result = ml_extractor.extract_outline(pdf_path)
    ml_time = time.time() - start_time
    
    results['ml_enhanced'] = {
        'processing_time': ml_time,
        'total_headings': len(ml_result['headings']),
        'headings': ml_result['headings'],
        'title': ml_result['title'],
        'detected_language': ml_result['metadata']['detected_language'],
        'classification_method': ml_result['metadata']['classification_method'],
        'ml_model_info': ml_extractor.get_ml_model_info()
    }
    
    print(f"   Time: {ml_time:.3f}s")
    print(f"   Headings found: {len(ml_result['headings'])}")
    print(f"   Title: {ml_result['title']}")
    print(f"   Language: {ml_result['metadata']['detected_language']}")
    print(f"   Classification: {ml_result['metadata']['classification_method']}")
    
    # Print detailed heading comparison
    print("\n4. Detailed Heading Comparison:")
    print("-" * 60)
    
    print("\nOriginal Extractor Headings:")
    for i, heading in enumerate(original_result['headings']):
        print(f"   {i+1}. [{heading['level']}] {heading['text']} (Page {heading['page']})")
    
    print("\nEnhanced Extractor Headings:")
    for i, heading in enumerate(enhanced_result['headings']):
        print(f"   {i+1}. [{heading['level']}] {heading['text']} (Page {heading['page']}, Size: {heading.get('font_size', 'N/A')})")
    
    print("\nML-Enhanced Extractor Headings:")
    for i, heading in enumerate(ml_result['headings']):
        confidence = heading.get('confidence', 0)
        print(f"   {i+1}. [{heading['level']}] {heading['text']} (Page {heading['page']}, Confidence: {confidence:.3f})")
    
    # Performance comparison
    print("\n5. Performance Comparison:")
    print("-" * 60)
    print(f"Original:     {original_time:.3f}s ({len(original_result['headings'])} headings)")
    print(f"Enhanced:     {enhanced_time:.3f}s ({len(enhanced_result['headings'])} headings)")
    print(f"ML-Enhanced:  {ml_time:.3f}s ({len(ml_result['headings'])} headings)")
    
    # Speed comparison
    fastest = min(original_time, enhanced_time, ml_time)
    print(f"\nSpeed comparison (relative to fastest):")
    print(f"Original:     {original_time/fastest:.2f}x")
    print(f"Enhanced:     {enhanced_time/fastest:.2f}x")
    print(f"ML-Enhanced:  {ml_time/fastest:.2f}x")
    
    # Save detailed results
    os.makedirs(output_dir, exist_ok=True)
    comparison_file = os.path.join(output_dir, f"extractor_comparison_{os.path.splitext(os.path.basename(pdf_path))[0]}.json")
    
    with open(comparison_file, 'w') as f:
        json.dump(results, f, indent=4, default=str)
    
    print(f"\nDetailed comparison saved to: {comparison_file}")
    
    return results

def run_comprehensive_ml_test():
    """Run comprehensive ML comparison test."""
    print("🤖 COMPREHENSIVE ML EXTRACTOR COMPARISON TEST")
    print("=" * 80)
    
    # Test files
    test_files = [
        "./input/dummy.pdf",
        "./input/multilingual_test.pdf"
    ]
    
    all_results = {}
    
    for pdf_file in test_files:
        if os.path.exists(pdf_file):
            print(f"\n📄 Testing: {os.path.basename(pdf_file)}")
            results = compare_all_extractors(pdf_file)
            all_results[os.path.basename(pdf_file)] = results
        else:
            print(f"❌ File not found: {pdf_file}")
    
    # Overall summary
    print("\n📊 OVERALL SUMMARY")
    print("=" * 80)
    
    for filename, results in all_results.items():
        print(f"\n{filename}:")
        for method, data in results.items():
            print(f"  {method.capitalize():12}: {data['processing_time']:.3f}s, {data['total_headings']} headings")
    
    # ML Model Information
    if all_results:
        sample_result = list(all_results.values())[0]
        if 'ml_enhanced' in sample_result:
            ml_info = sample_result['ml_enhanced']['ml_model_info']
            print(f"\n🧠 ML MODEL INFORMATION")
            print("-" * 40)
            print(f"Status: {ml_info.get('status', 'Unknown')}")
            print(f"Features: {ml_info.get('feature_count', 0)}")
            print(f"Classes: {', '.join(ml_info.get('classes', []))}")
            
            if 'feature_importance' in ml_info:
                print("\nTop 5 Most Important Features:")
                importance = ml_info['feature_importance']
                sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
                for feature, imp in sorted_features[:5]:
                    print(f"  {feature}: {imp:.3f}")
    
    print("\n✅ Comprehensive ML test completed!")
    return all_results

if __name__ == "__main__":
    run_comprehensive_ml_test()

