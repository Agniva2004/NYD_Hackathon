from llama_index.core import SimpleDirectoryReader
import os

class DataConnector:
    def __init__(self, folder_path):
        """
        Initializes the DataConnector class with the path to the folder containing the files.
        """
        self.folder_path = folder_path

    def fetch_csv_files_with_llamaindex(self):
        """
        Fetches content from all CSV files in the folder using LlamaIndex.
        Returns:
            List of document objects from CSV files.
        """
        
        csv_files = [os.path.join(self.folder_path, file) 
                     for file in os.listdir(self.folder_path) 
                     if file.endswith('.csv')]

        if not csv_files:
            print("No CSV files found in the folder.")
            return []

    
        reader = SimpleDirectoryReader(input_dir=self.folder_path, required_exts=[".csv"])
        documents = reader.load_data()
        return documents

    def fetch_pdf_files_with_llamaindex(self):
        """
        Fetches content from all PDF files in the folder using LlamaIndex.
        Returns:
            List of document objects from PDF files.
        """

        pdf_files = [os.path.join(self.folder_path, file) 
                     for file in os.listdir(self.folder_path) 
                     if file.endswith('.pdf')]

        if not pdf_files:
            print("No PDF files found in the folder.")
            return []

        reader = SimpleDirectoryReader(input_dir=self.folder_path, required_exts=[".pdf"])
        documents = reader.load_data()
        return documents

if __name__ == "__main__":
    folder_path =r'C:\Users\Anushree\Desktop\NYD\NYD_Hackathon\check'  # Replace with your folder path
    connector = DataConnector(folder_path)

    csv_documents = connector.fetch_csv_files_with_llamaindex()
    print("\nCSV Documents:")
    for doc in csv_documents:
        print(f"Document ID: {doc.doc_id}")
        print(f"Document Text (Preview): {doc.text[:500]}")


    pdf_documents = connector.fetch_pdf_files_with_llamaindex()
    print("\nPDF Documents:")
    for doc in pdf_documents:
        print(f"Document ID: {doc.doc_id}")
        print(f"Document Text (Preview): {doc.text[:500]}")
