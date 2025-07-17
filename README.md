# Adobe Hackathon Project: Persona-Driven Document Analysis

This project is a solution for the Adobe Hackathon, focusing on intelligent document analysis tailored to user personas and specific tasks. It comprises two main components:

- **Round 1A: Outline Extractor**: A module that parses PDF documents to extract a structured outline, including headings and titles.
- **Round 1B: Persona-Driven Document Analyst**: A module that leverages the extracted outline and full-text content to rank and summarize document sections based on a user persona and a "job-to-be-done."

## Project Structure

```
project_adobe_hackathon/
├── Dockerfile
├── README.md
├── input/
│   └── dummy.pdf
├── main.py
├── output/
│   ├── dummy.json
│   └── round1b_analysis.json
├── persona.json
├── requirements.txt
├── round1a/
│   ├── __init__.py
│   └── outline_extractor.py
└── round1b/
    ├── __init__.py
    └── document_analyst.py
```

## Getting Started

### Prerequisites

- Docker

### Installation and Setup

1.  **Build the Docker image:**

    ```bash
    docker build -t adobe-hackathon-project .
    ```

### Running the Project

The project can be run via the `main.py` script within the Docker container. You can specify which round to execute (1A or 1B) and provide necessary inputs.

1.  **Create a dummy PDF for testing:**

    ```bash
    docker run --rm -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output adobe-hackathon-project python main.py --create_dummy_pdf
    ```

2.  **Run Round 1A (Outline Extraction):**

    This will process all PDFs in the `input` directory and save the extracted outlines as JSON files in the `output` directory.

    ```bash
    docker run --rm -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output adobe-hackathon-project python main.py --round 1A
    ```

3.  **Run Round 1B (Persona-Driven Analysis):**

    This will perform a persona-driven analysis on the PDFs in the `input` directory, using the specified persona and job-to-be-done.

    ```bash
    docker run --rm -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output -v $(pwd)/persona.json:/app/persona.json adobe-hackathon-project python main.py --round 1B --persona_file /app/persona.json --job_to_be_done "Find relevant sections about AI applications."
    ```

## Implementation Details

### Round 1A: Outline Extractor

-   **PDF Parsing**: Uses the `PyMuPDF` library to open and read PDF documents.
-   **Title and Heading Extraction**: Employs a set of heuristics to identify the document title and headings based on font size, style (bold), and layout.
-   **Full Text Extraction**: Extracts the full text content from each page of the PDF.
-   **JSON Output**: The extracted outline, including the title, headings (with levels and page numbers), and full text per page, is saved in a structured JSON format.

### Round 1B: Persona-Driven Document Analyst

-   **Persona and Job-to-be-Done**: Takes a user persona (defined in a JSON file) and a "job-to-be-done" (a natural language description of the user's goal) as input.
-   **Lightweight NLP**: Utilizes a combination of NLP techniques to analyze the document content:
    -   **Keyword Extraction**: Uses `rake-nltk` and `summa` to identify important keywords.
    -   **Named Entity Recognition (NER)**: Employs `spaCy` to recognize and categorize named entities (e.g., persons, organizations, locations).
    -   **Extractive Summarization**: Leverages `summa` to generate concise summaries of document sections.
-   **Relevance Ranking**: Ranks document sections based on their relevance to the user's persona and job-to-be-done. The ranking algorithm considers:
    -   **TF-IDF Cosine Similarity**: Measures the similarity between the user's query (persona + job-to-be-done) and each document section.
    -   **Keyword and Entity Overlap**: Considers the overlap of keywords and named entities between the query and each section.
    -   **Heading Level**: Assigns higher importance to higher-level headings (e.g., H1, H2).
-   **JSON Output**: The final output is a JSON file containing the ranked and summarized document sections, along with the persona and job-to-be-done for context.


