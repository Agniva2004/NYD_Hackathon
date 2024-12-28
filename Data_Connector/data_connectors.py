from llama_index.readers.web import SimpleWebPageReader
from llama_index.core import SimpleDirectoryReader
import os

class DataConnector:
    def __init__(self, urls=None, base_folder_path=None):

        self.urls = urls if urls else []
        self.base_folder_path = base_folder_path

    def fetch_web_pages_with_llamaindex(self):

        if not self.urls:
            print("No URLs provided.")
            return []

        reader = SimpleWebPageReader(html_to_text=True)
        documents = reader.load_data(self.urls)
        return documents

    def fetch_files_with_llamaindex(self, subfolder):

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
    folder_path = r'..\Data'
    urls = [
        "http://paulgraham.com/worked.html",
        "https://wikipedia.org"
    ]

    connector = DataConnector(urls=urls, base_folder_path=folder_path)

    web_documents = connector.fetch_web_pages_with_llamaindex()
    print("\nWeb Page Documents:")
    for doc in web_documents:
        print(f"Document URL: {doc.extra_info.get('url', 'N/A')}")
        print(f"Document Text (Preview): {doc.text[:500]}")

    csv_documents = connector.fetch_files_with_llamaindex(subfolder="Patanjali_Yoga_Sutras")
    print("\nCSV Documents:")
    for doc in csv_documents:
        print(f"Document ID: {doc.doc_id}")
        print(f"Document Text (Preview): {doc.text[:500]}")

    pdf_documents = connector.fetch_files_with_llamaindex(subfolder="PDFs")
    print("\nPDF Documents:")
    for doc in pdf_documents:
        print(f"Document ID: {doc.doc_id}")
        print(f"Document Text (Preview): {doc.text[:500]}")
