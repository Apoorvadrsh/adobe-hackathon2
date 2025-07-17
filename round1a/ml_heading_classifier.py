"""
Machine Learning-based Heading Classification for PDF Text Blocks

This module implements a lightweight ML model to classify text blocks as H1, H2, H3, or body text
based on features like font size, bold/italic presence, indentation, vertical spacing, and word count.
"""

import numpy as np
import pandas as pd
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import joblib
import json
from typing import List, Dict, Tuple, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HeadingClassifier:
    """
    Lightweight ML model for classifying text blocks as headings or body text.
    
    Features used:
    - Font size (normalized)
    - Bold presence (binary)
    - Italic presence (binary)
    - Indentation level (normalized)
    - Vertical spacing before (normalized)
    - Vertical spacing after (normalized)
    - Word count (normalized)
    - Character count (normalized)
    - Position on page (normalized)
    - Line length ratio
    - Capitalization ratio
    - Punctuation ratio
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = [
            'font_size_norm', 'is_bold', 'is_italic', 'indentation_norm',
            'vertical_spacing_before_norm', 'vertical_spacing_after_norm',
            'word_count_norm', 'char_count_norm', 'position_y_norm',
            'line_length_ratio', 'capitalization_ratio', 'punctuation_ratio'
        ]
        self.label_mapping = {
            'body': 0,
            'H3': 1,
            'H2': 2,
            'H1': 3
        }
        self.reverse_label_mapping = {v: k for k, v in self.label_mapping.items()}
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self._create_default_model()
    
    def _create_default_model(self):
        """Create a default lightweight model."""
        # Use RandomForest with limited parameters to keep model small
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(
                n_estimators=50,  # Limited trees for size
                max_depth=10,     # Limited depth
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ))
        ])
    
    def extract_features(self, text_blocks: List[Dict]) -> np.ndarray:
        """
        Extract features from text blocks for ML classification.
        
        Args:
            text_blocks: List of dictionaries containing text block information
            
        Returns:
            Feature matrix as numpy array
        """
        features = []
        
        # Calculate normalization factors
        font_sizes = [block.get('font_size', 12) for block in text_blocks]
        positions_y = [block.get('bbox', [0, 0, 0, 0])[1] for block in text_blocks]
        
        max_font_size = max(font_sizes) if font_sizes else 24
        min_font_size = min(font_sizes) if font_sizes else 8
        max_position_y = max(positions_y) if positions_y else 800
        min_position_y = min(positions_y) if positions_y else 0
        
        for i, block in enumerate(text_blocks):
            text = block.get('text', '')
            font_size = block.get('font_size', 12)
            font_name = block.get('font', '').lower()
            bbox = block.get('bbox', [0, 0, 0, 0])
            
            # Calculate features
            features_row = []
            
            # 1. Normalized font size
            font_size_norm = (font_size - min_font_size) / max(1, max_font_size - min_font_size)
            features_row.append(font_size_norm)
            
            # 2. Bold presence
            is_bold = 1 if 'bold' in font_name else 0
            features_row.append(is_bold)
            
            # 3. Italic presence
            is_italic = 1 if 'italic' in font_name else 0
            features_row.append(is_italic)
            
            # 4. Indentation (normalized by page width)
            page_width = 600  # Approximate page width
            indentation_norm = bbox[0] / page_width
            features_row.append(indentation_norm)
            
            # 5. Vertical spacing before (normalized)
            if i > 0:
                prev_bbox = text_blocks[i-1].get('bbox', [0, 0, 0, 0])
                vertical_spacing_before = bbox[1] - prev_bbox[3]
            else:
                vertical_spacing_before = 0
            vertical_spacing_before_norm = min(vertical_spacing_before / 50, 1.0)  # Normalize by typical line height
            features_row.append(vertical_spacing_before_norm)
            
            # 6. Vertical spacing after (normalized)
            if i < len(text_blocks) - 1:
                next_bbox = text_blocks[i+1].get('bbox', [0, 0, 0, 0])
                vertical_spacing_after = next_bbox[1] - bbox[3]
            else:
                vertical_spacing_after = 0
            vertical_spacing_after_norm = min(vertical_spacing_after / 50, 1.0)
            features_row.append(vertical_spacing_after_norm)
            
            # 7. Word count (normalized)
            word_count = len(text.split())
            word_count_norm = min(word_count / 20, 1.0)  # Normalize by typical heading length
            features_row.append(word_count_norm)
            
            # 8. Character count (normalized)
            char_count = len(text)
            char_count_norm = min(char_count / 100, 1.0)  # Normalize by typical heading length
            features_row.append(char_count_norm)
            
            # 9. Position on page (normalized)
            position_y_norm = (bbox[1] - min_position_y) / max(1, max_position_y - min_position_y)
            features_row.append(position_y_norm)
            
            # 10. Line length ratio (width vs typical line)
            line_width = bbox[2] - bbox[0]
            line_length_ratio = min(line_width / page_width, 1.0)
            features_row.append(line_length_ratio)
            
            # 11. Capitalization ratio
            if text:
                capitalization_ratio = sum(1 for c in text if c.isupper()) / len(text)
            else:
                capitalization_ratio = 0
            features_row.append(capitalization_ratio)
            
            # 12. Punctuation ratio
            if text:
                punctuation_count = sum(1 for c in text if c in '.,!?;:')
                punctuation_ratio = punctuation_count / len(text)
            else:
                punctuation_ratio = 0
            features_row.append(punctuation_ratio)
            
            features.append(features_row)
        
        return np.array(features)
    
    def generate_synthetic_training_data(self, num_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic training data for the heading classifier.
        This creates realistic feature combinations for different heading types.
        """
        np.random.seed(42)
        features = []
        labels = []
        
        for _ in range(num_samples):
            # Randomly choose a class
            class_type = np.random.choice(['body', 'H3', 'H2', 'H1'], p=[0.6, 0.2, 0.15, 0.05])
            
            if class_type == 'H1':
                # H1 characteristics: large font, bold, minimal indentation, good spacing
                font_size_norm = np.random.normal(0.9, 0.1)
                is_bold = np.random.choice([0, 1], p=[0.1, 0.9])
                is_italic = np.random.choice([0, 1], p=[0.8, 0.2])
                indentation_norm = np.random.normal(0.1, 0.05)
                vertical_spacing_before_norm = np.random.normal(0.8, 0.2)
                vertical_spacing_after_norm = np.random.normal(0.6, 0.2)
                word_count_norm = np.random.normal(0.3, 0.1)
                char_count_norm = np.random.normal(0.3, 0.1)
                position_y_norm = np.random.uniform(0, 1)
                line_length_ratio = np.random.normal(0.6, 0.2)
                capitalization_ratio = np.random.normal(0.3, 0.2)
                punctuation_ratio = np.random.normal(0.05, 0.03)
                
            elif class_type == 'H2':
                # H2 characteristics: medium-large font, often bold, some spacing
                font_size_norm = np.random.normal(0.7, 0.1)
                is_bold = np.random.choice([0, 1], p=[0.2, 0.8])
                is_italic = np.random.choice([0, 1], p=[0.8, 0.2])
                indentation_norm = np.random.normal(0.15, 0.1)
                vertical_spacing_before_norm = np.random.normal(0.6, 0.2)
                vertical_spacing_after_norm = np.random.normal(0.4, 0.2)
                word_count_norm = np.random.normal(0.4, 0.15)
                char_count_norm = np.random.normal(0.4, 0.15)
                position_y_norm = np.random.uniform(0, 1)
                line_length_ratio = np.random.normal(0.7, 0.2)
                capitalization_ratio = np.random.normal(0.2, 0.15)
                punctuation_ratio = np.random.normal(0.06, 0.04)
                
            elif class_type == 'H3':
                # H3 characteristics: medium font, sometimes bold, moderate spacing
                font_size_norm = np.random.normal(0.5, 0.1)
                is_bold = np.random.choice([0, 1], p=[0.4, 0.6])
                is_italic = np.random.choice([0, 1], p=[0.8, 0.2])
                indentation_norm = np.random.normal(0.2, 0.1)
                vertical_spacing_before_norm = np.random.normal(0.4, 0.2)
                vertical_spacing_after_norm = np.random.normal(0.3, 0.2)
                word_count_norm = np.random.normal(0.5, 0.2)
                char_count_norm = np.random.normal(0.5, 0.2)
                position_y_norm = np.random.uniform(0, 1)
                line_length_ratio = np.random.normal(0.8, 0.2)
                capitalization_ratio = np.random.normal(0.15, 0.1)
                punctuation_ratio = np.random.normal(0.08, 0.05)
                
            else:  # body text
                # Body text characteristics: normal font, no bold, full lines, minimal spacing
                font_size_norm = np.random.normal(0.3, 0.1)
                is_bold = np.random.choice([0, 1], p=[0.9, 0.1])
                is_italic = np.random.choice([0, 1], p=[0.9, 0.1])
                indentation_norm = np.random.normal(0.1, 0.05)
                vertical_spacing_before_norm = np.random.normal(0.1, 0.1)
                vertical_spacing_after_norm = np.random.normal(0.1, 0.1)
                word_count_norm = np.random.normal(0.8, 0.2)
                char_count_norm = np.random.normal(0.9, 0.1)
                position_y_norm = np.random.uniform(0, 1)
                line_length_ratio = np.random.normal(0.9, 0.1)
                capitalization_ratio = np.random.normal(0.05, 0.03)
                punctuation_ratio = np.random.normal(0.12, 0.05)
            
            # Clip values to valid ranges
            feature_row = [
                np.clip(font_size_norm, 0, 1),
                np.clip(is_bold, 0, 1),
                np.clip(is_italic, 0, 1),
                np.clip(indentation_norm, 0, 1),
                np.clip(vertical_spacing_before_norm, 0, 1),
                np.clip(vertical_spacing_after_norm, 0, 1),
                np.clip(word_count_norm, 0, 1),
                np.clip(char_count_norm, 0, 1),
                np.clip(position_y_norm, 0, 1),
                np.clip(line_length_ratio, 0, 1),
                np.clip(capitalization_ratio, 0, 1),
                np.clip(punctuation_ratio, 0, 1)
            ]
            
            features.append(feature_row)
            labels.append(self.label_mapping[class_type])
        
        return np.array(features), np.array(labels)
    
    def train(self, X: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None, 
              use_synthetic: bool = True) -> Dict:
        """
        Train the heading classifier.
        
        Args:
            X: Feature matrix (optional, will generate synthetic if not provided)
            y: Labels (optional, will generate synthetic if not provided)
            use_synthetic: Whether to use synthetic training data
            
        Returns:
            Training metrics dictionary
        """
        if X is None or y is None or use_synthetic:
            logger.info("Generating synthetic training data...")
            X, y = self.generate_synthetic_training_data(num_samples=2000)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info(f"Training with {len(X_train)} samples, testing with {len(X_test)} samples")
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evaluate
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5)
        
        # Predictions for detailed metrics
        y_pred = self.model.predict(X_test)
        
        metrics = {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'classification_report': classification_report(
                y_test, y_pred, 
                target_names=list(self.reverse_label_mapping.values()),
                output_dict=True
            ),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        logger.info(f"Training completed. Test accuracy: {test_score:.3f}")
        logger.info(f"Cross-validation: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        return metrics
    
    def predict(self, text_blocks: List[Dict]) -> List[str]:
        """
        Predict heading levels for text blocks.
        
        Args:
            text_blocks: List of text block dictionaries
            
        Returns:
            List of predicted labels ('H1', 'H2', 'H3', 'body')
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        if not text_blocks:
            return []
        
        features = self.extract_features(text_blocks)
        predictions = self.model.predict(features)
        
        return [self.reverse_label_mapping[pred] for pred in predictions]
    
    def predict_proba(self, text_blocks: List[Dict]) -> np.ndarray:
        """
        Predict class probabilities for text blocks.
        
        Args:
            text_blocks: List of text block dictionaries
            
        Returns:
            Probability matrix (n_samples, n_classes)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        if not text_blocks:
            return np.array([])
        
        features = self.extract_features(text_blocks)
        return self.model.predict_proba(features)
    
    def save_model(self, model_path: str):
        """Save the trained model to disk."""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'label_mapping': self.label_mapping,
            'reverse_label_mapping': self.reverse_label_mapping
        }
        
        joblib.dump(model_data, model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Check file size
        file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
        logger.info(f"Model file size: {file_size_mb:.2f} MB")
        
        if file_size_mb > 200:
            logger.warning(f"Model size ({file_size_mb:.2f} MB) exceeds 200MB limit!")
    
    def load_model(self, model_path: str):
        """Load a trained model from disk."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model_data = joblib.load(model_path)
        
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.label_mapping = model_data['label_mapping']
        self.reverse_label_mapping = model_data['reverse_label_mapping']
        
        logger.info(f"Model loaded from {model_path}")
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from the trained model."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Get feature importance from the RandomForest classifier
        classifier = self.model.named_steps['classifier']
        importance = classifier.feature_importances_
        
        return dict(zip(self.feature_names, importance))

def create_and_train_model(model_save_path: str = None) -> HeadingClassifier:
    """
    Create and train a new heading classifier model.
    
    Args:
        model_save_path: Path to save the trained model
        
    Returns:
        Trained HeadingClassifier instance
    """
    logger.info("Creating and training heading classifier...")
    
    classifier = HeadingClassifier()
    metrics = classifier.train(use_synthetic=True)
    
    # Print training results
    print("\n=== Heading Classifier Training Results ===")
    print(f"Test Accuracy: {metrics['test_accuracy']:.3f}")
    print(f"Cross-validation: {metrics['cv_mean']:.3f} (+/- {metrics['cv_std'] * 2:.3f})")
    
    print("\nFeature Importance:")
    importance = classifier.get_feature_importance()
    for feature, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
        print(f"  {feature}: {imp:.3f}")
    
    print("\nClassification Report:")
    report = metrics['classification_report']
    for class_name in ['body', 'H3', 'H2', 'H1']:
        if class_name in report:
            precision = report[class_name]['precision']
            recall = report[class_name]['recall']
            f1 = report[class_name]['f1-score']
            print(f"  {class_name}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")
    
    # Save model if path provided
    if model_save_path:
        classifier.save_model(model_save_path)
    
    return classifier

if __name__ == "__main__":
    # Train and save a new model
    model_path = "heading_classifier_model.joblib"
    classifier = create_and_train_model(model_path)
    
    # Example usage
    sample_blocks = [
        {
            'text': 'Introduction to Machine Learning',
            'font_size': 18,
            'font': 'Arial-Bold',
            'bbox': [50, 100, 400, 120]
        },
        {
            'text': 'This chapter provides an overview of machine learning concepts and applications.',
            'font_size': 12,
            'font': 'Arial',
            'bbox': [50, 140, 500, 160]
        }
    ]
    
    predictions = classifier.predict(sample_blocks)
    probabilities = classifier.predict_proba(sample_blocks)
    
    print(f"\nExample predictions: {predictions}")
    print(f"Example probabilities: {probabilities}")

