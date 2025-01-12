# NYD Hackathon 2024

## Overview

**NYD Hackathon** is a project created during the New Year's Day Hackathon, showcasing a sophisticated Retrieval-Augmented Generation (RAG) pipeline designed for spiritual texts. This pipeline, named **MARSA** (Modified Adaptive RAG for Spiritual Assistance), provides accurate and contextually rich responses to user queries related to the Bhagavad Gita and Patanjali Yoga Sutras. It integrates advanced NLP techniques with multiple specialized agents to ensure high-quality outputs.

---

## Key Features

### MARSA Pipeline Components

1. **Router Agent**: Routes queries to either the vectorstore (for internal documents) or the web search agent.
2. **Web Search Agent**: Uses Tavily search to fetch real-time data from the web.
3. **Retrieval Agent**: Enhances retrieval accuracy using a reranker and a context compression retriever.
4. **Verse Extractor Agent**: Extracts chapter and verse references from the Bhagavad Gita and Patanjali Yoga Sutras.
5. **Grader Agent**: Validates the relevance of retrieved documents.
6. **Question Rewriter Agent**: Refines ambiguous queries for improved processing.
7. **Generator Agent**: Generates final responses with contextual memory.
8. **Hallucination Grader Agent**: Detects and minimizes hallucinated facts.
9. **Introspective Agent**: Refines responses for accuracy and reduces toxicity.
10. **Joiner Agent**: Combines the response with chapter and verse references for a coherent output.

### Evaluation Metrics
The pipeline is evaluated using:
- **BERT Score**
- **BLEU Score**
- **ROUGE-L**
- **Verse Accuracy**
- **Chapter Accuracy**
- **Faithfulness**
- **Contextual Recall**

---

## Repository Structure

```
.
├── adaptive_rag                 # Contains the main RAG pipeline
├── Chain-of-Abstraction         # Chain of abstraction agent
├── Chunking_TextSplitter        # Tools for chunking text
├── Combined_Data                # Aggregated data for training and testing
├── Data                         # Raw data files
├── Data_Augmentation            # Scripts for data augmentation
├── Data_Connector               # Connectors for external data sources
├── Embedder                     # Embedding utilities
├── LLMCompiler                  # Tools for LLM integration
├── Query_Transformations        # Query transformation utilities
├── ReactAgent                   # ReAct agent implementation
├── ReRanker                     # Reranking tools
├── Retriever                    # Retrieval mechanisms
├── VectorStore                  # Vector store utilities
├── __pycache__                  # Compiled Python files
├── app.py                       # Streamlit application entry point
├── adaptive_class_main.py       # Main script for running the pipeline
├── eval.py                      # Evaluation script
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

---

## Getting Started

### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- pip

### Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/Agniva2004/NYD_Hackathon.git
   cd NYD_Hackathon
   ```

2. **Create a Virtual Environment** (optional but recommended):

   ```bash
   python -m venv env
   source env/bin/activate   # On Windows: env\Scripts\activate
   ```

3. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

1. **Streamlit Application**:

   Navigate to the `adaptive_rag` folder:

   ```bash
   cd adaptive_rag
   streamlit run app.py
   ```

2. **Query Processing**:

   Run the pipeline for user queries:

   ```bash
   python adaptive_class_main.py
   ```

3. **Evaluation**:

   Evaluate the pipeline using `eval.py`:

   ```bash
   python eval.py
   ```

---

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a new branch for your feature or bug fix:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m "Description of changes"
   ```
4. Push to your branch:
   ```bash
   git push origin feature-name
   ```
5. Open a pull request.

---


