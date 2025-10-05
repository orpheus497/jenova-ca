from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from duckduckgo_search import DDGS

def web_search(query: str, max_results: int = 3) -> dict:
    """
    Performs a web search using DuckDuckGo and extracts the full text content of the top search results.
    """
    results = []
    try:
        with DDGS() as ddgs:
            search_results = list(ddgs.text(query, max_results=max_results))

        if not search_results:
            return {"results": []}

        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")

        with webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=chrome_options) as driver:
            for result in search_results:
                try:
                    driver.get(result['href'])
                    # A simple way to get the main content of the page. This can be improved.
                    body = driver.find_element(By.TAG_NAME, 'body')
                    text_content = body.text
                    results.append({
                        "title": result['title'],
                        "link": result['href'],
                        "content": text_content
                    })
                except Exception as e:
                    print(f"Error processing {result['href']}: {e}")
                    continue
    except Exception as e:
        print(f"An error occurred during web search: {e}")
        return {"results": []}

    return {"results": results}