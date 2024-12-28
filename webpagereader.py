from llama_index.readers.web import SimpleWebPageReader

class DataConnector:
    def __init__(self, urls):
        """
        Initializes the DataConnector class with a list of URLs.
        :param urls: List of URLs to fetch content from.
        """
        if not urls or not isinstance(urls, list):
            raise ValueError("A list of URLs is required.")
        self.urls = urls

    def fetch_web_pages_with_llamaindex(self):
        """
        Fetches content from the given list of URLs using SimpleWebPageReader.
        Converts HTML content to text if html_to_text=True.
        Returns:
            List of document objects from web pages.
        """
        reader = SimpleWebPageReader(html_to_text=True)
        documents = reader.load_data(self.urls)
        return documents


# Example usage
if __name__ == "__main__":
    # Replace these URLs with the actual ones you want to fetch
    urls = [
        "http://paulgraham.com/worked.html",
        "https://wikipedia.org"
    ]

    connector = DataConnector(urls)

    # Fetch and print Web Page documents
    web_documents = connector.fetch_web_pages_with_llamaindex()
    print("\nWeb Page Documents:")
    for doc in web_documents:
        print(f"Document URL: {doc.extra_info.get('url', 'N/A')}")
        print(f"Document Text (Preview): {doc.text[:500]}")
