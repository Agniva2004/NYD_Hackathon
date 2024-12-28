from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from PyPDF2 import PdfReader
from langchain.text_splitter import TokenTextSplitter
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
import numpy as np
from typing import List, Dict

class SemanticChunkerWrapper:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", device="cpu"):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.chunker = SemanticChunker(self.embeddings)

    def split_text(self, text):
        return self.chunker.split_text(text)
    
class TokenChunker:
    def __init__(self, 
                 chunk_size: int = 100,
                 chunk_overlap: int = 20,
                 encoding_name: str = "cl100k_base"):
        """
        Initialize the token-based chunker
        
        Args:
            chunk_size: Number of tokens per chunk
            chunk_overlap: Number of overlapping tokens between chunks
            encoding_name: Name of the tokenizer encoding to use
        """
        self.splitter = TokenTextSplitter(
            encoding_name=encoding_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def split_text(self, text: str) -> List[str]:
        """Split text into chunks using token-based splitting"""
        return self.splitter.split_text(text)   
    
    def split_documents(self, documents: List[str]) -> List[str]:
        """Split multiple documents into chunks"""
        all_chunks = []
        for doc in documents:
            chunks = self.split_text(doc)
            all_chunks.extend(chunks)
        return all_chunks
    
class MarkdownChunker:
    def __init__(self, headers_to_split_on: List[tuple] = None):
        """
        Initialize the markdown chunker
        
        Args:
            headers_to_split_on: List of tuples containing (header_symbol, header_name)
                               If None, uses default headers (H1, H2, H3)
        """
        if headers_to_split_on is None:
            self.headers_to_split_on = [
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
            ]
        else:
            self.headers_to_split_on = headers_to_split_on
            
        self.splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.headers_to_split_on
        )   
        
    def split_text(self, text: str) -> List[Dict]:
        """
        Split markdown text into chunks based on headers
        
        Args:
            text: Markdown formatted text
            
        Returns:
            List of dictionaries containing:
                - page_content: The text content
                - metadata: Header level information
        """
        return self.splitter.split_text(text)
    
    def extract_sections(self, chunks: List[Dict]) -> Dict[str, str]:
        """
        Organize chunks by their header levels
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            Dictionary mapping header paths to content
        """
        sections = {}
        for chunk in chunks:
            
            header_path = []
            for header_type in ["Header 1", "Header 2", "Header 3"]:
                if header_type in chunk.metadata:
                    header_path.append(chunk.metadata[header_type])
            
            
            section_key = " > ".join(header_path)
            sections[section_key] = chunk.page_content
            
        return sections     
    
     

class HybridMarkdownChunker:
    def __init__(self, 
                 headers_to_split_on: List[tuple] = None,
                 chunk_size: int = 250,
                 chunk_overlap: int = 30,
                 strip_headers: bool = False):
        """
        Initialize the hybrid markdown chunker
        
        Args:
            headers_to_split_on: List of tuples containing (header_symbol, header_name)
            chunk_size: Size of character-level chunks
            chunk_overlap: Overlap between character-level chunks
            strip_headers: Whether to strip headers from content
        """
       
        if headers_to_split_on is None:
            self.headers_to_split_on = [
                ("#", "Header 1"),
                ("##", "Header 2"),
            ]
        else:
            self.headers_to_split_on = headers_to_split_on
            
        
        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.headers_to_split_on,
            strip_headers=strip_headers
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def split_text(self, text: str) -> List[Dict]:
        """
        Perform two-stage splitting: first by headers, then by character chunks
        
        Args:
            text: Markdown formatted text
            
        Returns:
            List of chunks with metadata
        """
        
        md_splits = self.markdown_splitter.split_text(text)
        
       
        final_splits = self.text_splitter.split_documents(md_splits)
        
        return final_splits
    
    def extract_section_info(self, chunks: List[Dict]) -> List[Dict]:
        """
        Extract and format section information from chunks
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            List of dictionaries with formatted section information
        """
        formatted_chunks = []
        for chunk in chunks:
            section_info = {
                'content': chunk.page_content,
                'headers': {},
                'section_path': []
            }
            
            
            for header_type in ["Header 1", "Header 2"]:
                if header_type in chunk.metadata:
                    section_info['headers'][header_type] = chunk.metadata[header_type]
                    section_info['section_path'].append(chunk.metadata[header_type])
            
            section_info['section_path'] = " > ".join(section_info['section_path'])
            formatted_chunks.append(section_info)
            
        return formatted_chunks

def get_chunker_strategy(strategy: str, **kwargs):
    """
    Factory function to return the appropriate chunker based on strategy
    
    Args:
        strategy: One of 'semantic', 'token', 'markdown', or 'hybrid'
        kwargs: Additional arguments for the chunker initialization
    """
    strategies = {
        'semantic': SemanticChunkerWrapper,
        'token': TokenChunker,
        'markdown': MarkdownChunker,
        'hybrid': HybridMarkdownChunker
    }
    
    if strategy not in strategies:
        raise ValueError(f"Invalid strategy. Choose from: {', '.join(strategies.keys())}")
    
    return strategies[strategy](**kwargs)

def main():
    try:
        
        print("\nAvailable chunking strategies:")
        print("1. semantic  - Uses sentence-transformers for semantic chunking")
        print("2. token    - Uses token-based chunking")
        print("3. markdown - Uses markdown header-based chunking")
        print("4. hybrid   - Uses combined markdown and character-based chunking")
        
        strategy = input("\nEnter chunking strategy (semantic/token/markdown/hybrid): ").lower()
        
        
        if strategy == 'semantic':
            chunker = get_chunker_strategy(strategy, 
                                        model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        elif strategy == 'token':
            chunker = get_chunker_strategy(strategy,
                                        chunk_size=300,
                                        chunk_overlap=50,
                                        encoding_name="gpt2")
        
        elif strategy == 'markdown':
            chunker = get_chunker_strategy(strategy,
                                        headers_to_split_on=[
                                            ("#", "Header 1"),
                                            ("##", "Header 2"),
                                            ("###", "Header 3")
                                        ])
        
        elif strategy == 'hybrid':
            chunker = get_chunker_strategy(strategy,
                                        chunk_size=250,
                                        chunk_overlap=30,
                                        strip_headers=False)
        
        
        pdf_path = 'pdf_folder\PS_Appian_AI_Challenge_2025.pdf' 
        pdf_reader = PdfReader(pdf_path)
        text = ""
        
        
        for page in pdf_reader.pages:
            text += page.extract_text()
        
       
        if strategy in ['semantic', 'token']:
            chunks = chunker.split_text(text)
            print(f"\nCreated {len(chunks)} chunks")
            print("\nFirst 3 chunks:")
            for i, chunk in enumerate(chunks[:3]):
                print(f"\nChunk {i+1}:")
                print(chunk[:100] + "...")
                
        elif strategy == 'markdown':
            chunks = chunker.split_text(text)
            sections = chunker.extract_sections(chunks)
            print("\nDocument Structure:")
            for header_path, content in sections.items():
                print(f"\n{header_path}:")
                print(f"{content[:100]}...")
                
        elif strategy == 'hybrid':
            chunks = chunker.split_text(text)
            formatted_chunks = chunker.extract_section_info(chunks)
            print(f"\nCreated {len(chunks)} chunks")
            print("\nDocument Structure:")
            for i, chunk in enumerate(formatted_chunks[:3]):
                print(f"\nChunk {i+1}:")
                print(f"Section: {chunk['section_path']}")
                print(f"Content: {chunk['content'][:100]}...")
                
    except FileNotFoundError:
        print("File not found. Please provide the correct file path.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()