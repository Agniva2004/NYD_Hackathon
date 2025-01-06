from langchain_core.documents import Document
from langchain_community.tools.tavily_search import TavilySearchResults
class Search:
    def __init__(self, k):
        self.web_search_tool = TavilySearchResults(k=k)
    def web_search(self, question):
        docs = self.web_search_tool.invoke({"query": question})
        web_results = "\n".join([d["content"] for d in docs])
        web_results = Document(page_content=web_results)
        return web_results