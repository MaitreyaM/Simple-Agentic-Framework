import os
from brave import Brave
from duckduckgo_search import DDGS
from ..tool_pattern.tool import tool

@tool
def brave_search(query: str, num_results: int = 5) -> str:
    """
    Performs a web search using the Brave Search API.

    Args:
        query (str): The search query string.
        num_results (int): The desired number of search results (default is 5).

    Returns:
        str: A string representation of the search results, or an error message.
    """
    try:
        brave = Brave()
        search_results = brave.search(q=query, count=num_results)
        return str(search_results)
    except Exception as e:
        return f"Error during Brave web search for '{query}': {e}"

@tool
def duckduckgo_search(query: str, num_results: int = 5) -> str:
    """Performs a web search using the DuckDuckGo Search API."""
    try:
        with DDGS() as ddgs:
            results = ddgs.text(keywords=query, max_results=num_results)
            if results:
                results_str = f"DuckDuckGo search results for '{query}':\n"
                for i, result in enumerate(results):
                    title = result.get('title', 'No Title')
                    snippet = result.get('body', 'No Snippet')
                    url = result.get('href', 'No URL')
                    results_str += f"{i+1}. {title}\n   URL: {url}\n   Snippet: {snippet}\n\n"
                return results_str.strip()
            else:
                return f"No DuckDuckGo results found for '{query}'."
    except Exception as e:
        return f"Error during DuckDuckGo web search for '{query}': {e}"