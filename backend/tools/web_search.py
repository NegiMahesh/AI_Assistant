from ddgs import DDGS

def web_search(query: str, max_results: int = 5):
    query = query.strip()
    if not query:
        return {"success": False, "error": "Search query cannot be empty.", "results": []}
    try:
        results = []
        with DDGS() as ddgs:
            search_results = ddgs.text(query, max_results=max_results)
            for result in search_results:
                results.append({
                    "title": result.get("title", ""),
                    "url": result.get("href", ""),
                    "snippet": result.get("body", ""),
                })
        return {"success": True, "query": query, "results": results}
    except Exception as error:
        return {"success": False, "query": query, "error": str(error), "results": []}
