from llama_index.readers.web import SimpleWebPageReader
from llama_index.core import SimpleDirectoryReader
import os

class DataConnector:
    def __init__(self, urls=None, base_folder_path=None):
        """
        Initializes the DataConnector class with a list of URLs and/or a base folder path.
        :param urls: List of URLs to fetch content from (optional).
        :param base_folder_path: Path to the folder containing subfolders with files (optional).
        """
        self.urls = urls if urls else []
        self.base_folder_path = base_folder_path

    def fetch_web_pages_with_llamaindex(self):
        """
        Fetches content from the given list of URLs using SimpleWebPageReader.
        Converts HTML content to text if html_to_text=True.
        Returns:
            List of document objects from web pages.
        """
        if not self.urls:
            print("No URLs provided.")
            return []

        reader = SimpleWebPageReader(html_to_text=True)
        documents = reader.load_data(self.urls)
        return documents

    def fetch_files_with_llamaindex(self, subfolder):
        """
        Fetches content from files in a specified subfolder.
        :param subfolder: Subfolder name within the base folder to fetch files from.
        Returns:
            List of document objects from files.
        """
        if not self.base_folder_path or not os.path.isdir(self.base_folder_path):
            print("Invalid or missing base folder path.")
            return []

        target_folder = os.path.join(self.base_folder_path, subfolder)

        if not os.path.isdir(target_folder):
            print(f"Subfolder '{subfolder}' does not exist.")
            return []

        reader = SimpleDirectoryReader(target_folder)
        documents = reader.load_data()
        return documents

if __name__ == "__main__":
    # Replace with actual folder path and URLs
    folder_path = r'C:\Users\Anushree\Desktop\NYD\NYD_Hackathon\Data'
    urls = [
        "http://paulgraham.com/worked.html",
        "https://wikipedia.org"
    ]

    connector = DataConnector(urls=urls, base_folder_path=folder_path)

    # Fetch and print web page documents
    web_documents = connector.fetch_web_pages_with_llamaindex()
    print("\nWeb Page Documents:")
    for doc in web_documents:
        print(f"Document URL: {doc.extra_info.get('url', 'N/A')}")
        print(f"Document Text (Preview): {doc.text[:500]}")

    # Fetch and print CSV documents from 'Patanjali_Yoga_Sutras'
    csv_documents = connector.fetch_files_with_llamaindex(subfolder="Patanjali_Yoga_Sutras")
    print("\nCSV Documents:")
    for doc in csv_documents:
        print(f"Document ID: {doc.doc_id}")
        print(f"Document Text (Preview): {doc.text[:500]}")

    # Fetch and print PDF documents from 'PDFs'
    pdf_documents = connector.fetch_files_with_llamaindex(subfolder="PDFs")
    print("\nPDF Documents:")
    for doc in pdf_documents:
        print(f"Document ID: {doc.doc_id}")
        print(f"Document Text (Preview): {doc.text[:500]}")
