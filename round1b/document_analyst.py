import json
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from summa import summarizer, keywords
from rake_nltk import Rake
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load spaCy model (ensure it\\\\\\\\\\\\\'s downloaded in Dockerfile)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading en_core_web_sm model for spaCy...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Download NLTK data (ensure it\\\\\\\\\\\\\'s downloaded in Dockerfile)
try:
    nltk.data.find("corpora/stopwords")
except nltk.downloader.DownloadError:
    nltk.download("stopwords")
try:
    nltk.data.find("tokenizers/punkt")
except nltk.downloader.DownloadError:
    nltk.download("punkt")

stop_words = set(stopwords.words("english"))

class DocumentAnalyst:
    def __init__(self, persona_file, job_to_be_done):
        self.persona = self._load_persona(persona_file)
        self.job_to_be_done = job_to_be_done
        self.tfidf_vectorizer = TfidfVectorizer()

    def _load_persona(self, persona_file):
        with open(persona_file, "r") as f:
            return json.load(f)

    def analyze_documents(self, pdf_paths, outline_extractor_instance):
        all_document_data = []
        for pdf_path in pdf_paths:
            outline_data = outline_extractor_instance.extract_outline(pdf_path)
            
            full_text = " ".join([h["text"] for h in outline_data["headings"]]) + outline_data["title"]

            document_sections = self._create_document_sections(outline_data)
            all_document_data.extend(document_sections)

        ranked_sections = self._rank_sections(all_document_data)
        
        results = {
            "metadata": {
                "persona": self.persona,
                "job_to_be_done": self.job_to_be_done
            },
            "extracted_sections": ranked_sections
        }
        return results

    def _create_document_sections(self, outline_data):
        sections = []
        full_text_by_page = outline_data["full_text_by_page"]

        for heading in outline_data["headings"]:
            page_num = heading["page"]
            content_text = full_text_by_page.get(page_num, "")

            sections.append({
                "title": heading["text"],
                "level": heading["level"],
                "page": heading["page"],
                "full_text_content": content_text, 
                "importance_rank": 0 
            })
        return sections

    def _preprocess_text(self, text):
        tokens = word_tokenize(text.lower())
        filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
        return " ".join(filtered_tokens)

    def _extract_keywords(self, text):
        if not text.strip() or len(text.split()) < 5: 
            return []

        summa_keywords = []
        try:
            summa_keywords_raw = keywords.keywords(text, words=5, scores=True, split=True)
            summa_keywords = [kw[0] for kw in summa_keywords_raw] if summa_keywords_raw else []
        except IndexError: # Catch the specific IndexError from summa
            print(f"Warning: summa.keywords failed for text: {text[:100]}...")
            summa_keywords = []
        
        r = Rake()
        r.extract_keywords_from_text(text)
        rake_keywords = r.get_ranked_phrases()
        
        if not rake_keywords:
            rake_keywords = []

        return list(set(summa_keywords + rake_keywords))

    def _perform_ner(self, text):
        if not text.strip():
            return []
        doc = nlp(text)
        return [(ent.text, ent.label_) for ent in doc.ents]

    def _summarize_text(self, text):
        if not text.strip():
            return ""
        return summarizer.summarize(text, words=50) 

    def _rank_sections(self, sections):
        if not sections: return []

        query_text = self._preprocess_text(self.persona.get("description", "") + " " + self.job_to_be_done)
        query_keywords = self._extract_keywords(query_text)
        query_entities = self._perform_ner(query_text)

        section_texts = [self._preprocess_text(s["full_text_content"]) for s in sections]
        
        if not any(section_texts):
            for section in sections:
                section["importance_rank"] = 0
                section["refined_text"] = self._summarize_text(section["full_text_content"])
            return sections

        non_empty_section_texts = [text for text in section_texts if text.strip()]
        if not non_empty_section_texts:
            for section in sections:
                section["importance_rank"] = 0
                section["refined_text"] = self._summarize_text(section["full_text_content"])
            return sections

        tfidf_matrix = self.tfidf_vectorizer.fit_transform([query_text] + non_empty_section_texts)
        query_tfidf = tfidf_matrix[0:1]
        section_tfidfs = tfidf_matrix[1:]

        cosine_similarities = cosine_similarity(query_tfidf, section_tfidfs).flatten()

        ranked_sections = []
        section_idx = 0
        for i, section in enumerate(sections):
            section_score = 0.0
            if section_texts[i].strip(): 
                section_score += cosine_similarities[section_idx] * 0.6

                section_keywords = self._extract_keywords(section["full_text_content"])
                section_entities = self._perform_ner(section["full_text_content"])
                
                keyword_overlap = len(set(query_keywords).intersection(set(section_keywords)))
                entity_overlap = len(set([e[0] for e in query_entities]).intersection(set([e[0] for e in section_entities])))
                
                section_score += (keyword_overlap + entity_overlap) * 0.2

                level_map = {"H1": 3, "H2": 2, "H3": 1}
                section_score += level_map.get(section["level"], 0) * 0.1

                section_score += min(len(section["full_text_content"]) / 500, 1) * 0.1
                section_idx += 1

            section["importance_rank"] = section_score
            section["refined_text"] = self._summarize_text(section["full_text_content"])
            ranked_sections.append(section)
        
        ranked_sections.sort(key=lambda x: x["importance_rank"], reverse=True)
        
        for i, section in enumerate(ranked_sections):
            section["importance_rank"] = i + 1

        return ranked_sections

if __name__ == "__main__":
    print("This module is part of the project and should be run via main.py or integrated for testing.")
    print("To test, ensure you have a PDF file and call DocumentAnalyst.analyze_documents(pdf_paths, outline_extractor_instance).")


