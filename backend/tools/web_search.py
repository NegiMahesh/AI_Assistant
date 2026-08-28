from ddgs import DDGS


def web_search(query: str, max_results: int = 5):
    """
    Search the web and return useful search results.
    """

    query = query.strip()

    if not query:
        return {
            "success": False,
            "error": "Search query cannot be empty."
        }

    try:
        results = []

        with DDGS() as ddgs:
            search_results = ddgs.text(
                query,
                max_results=max_results
            )

            for result in search_results:
                results.append({
                    "title": result.get("title", ""),
                    "url": result.get("href", ""),
                    "snippet": result.get("body", "")
                })

        return {
            "success": True,
            "query": query,
            "results": results
        }

    except Exception as error:
        return {
            "success": False,
            "query": query,
            "error": str(error),
            "results": []
        }


if __name__ == "__main__":

    print()
    print("=" * 60)
    print("WEB SEARCH TEST")
    print("=" * 60)

    query = input("Enter search query: ").strip()

    result = web_search(query)

    print()

    if not result["success"]:
        print("Search failed:")
        print(result["error"])

    else:
        print("Search successful!")
        print()
        print("Query:", result["query"])
        print()

        for index, item in enumerate(
            result["results"],
            start=1
        ):
            print(f"[{index}] {item['title']}")
            print("URL:", item["url"])
            print("INFO:", item["snippet"])
            print("-" * 60)

    print()
    print("=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)