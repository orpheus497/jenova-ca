import datetime
import subprocess
import shlex
import os
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options
from webdriver_manager.firefox import GeckoDriverManager

def get_current_datetime() -> str:
    """
    Returns the current date and time in ISO 8601 format.
    """
    return datetime.datetime.now().isoformat()

def execute_shell_command(command: str) -> dict:
    """
    Executes a shell command and returns the result.
    """
    try:
        # Use shlex.split to safely parse the command string
        command_args = shlex.split(command)
        result = subprocess.run(command_args, capture_output=True, text=True, check=False)
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
            "error": result.returncode != 0
        }
    except Exception as e:
        return {
            "stdout": "",
            "stderr": str(e),
            "returncode": -1,
            "error": True
        }

def web_search(query: str) -> list[dict]:
    """
    Performs a web search using DuckDuckGo and returns the results.
    """
    options = Options()
    options.headless = True
    driver = webdriver.Firefox(executable_path=GeckoDriverManager().install(), options=options)
    
    try:
        driver.get(f"https://duckduckgo.com/html/?q={query}")
        results = []
        for result in driver.find_elements(By.CLASS_NAME, "result"):
            title = result.find_element(By.CLASS_NAME, "result__title").text
            link = result.find_element(By.CLASS_NAME, "result__url").get_attribute("href")
            snippet = result.find_element(By.CLASS_NAME, "result__snippet").text
            results.append({"title": title, "link": link, "summary": snippet})
        return results
    finally:
        driver.quit()

class FileTools:
    def __init__(self, sandbox_path: str):
        self.sandbox_path = os.path.expanduser(sandbox_path)
        if not os.path.exists(self.sandbox_path):
            os.makedirs(self.sandbox_path)

    def _get_safe_path(self, path: str) -> str | None:
        """
        Resolves a path to an absolute path within the sandbox.
        Returns None if the path is outside the sandbox.
        """
        safe_path = os.path.abspath(os.path.join(self.sandbox_path, path))
        if os.path.commonpath([self.sandbox_path, safe_path]) != self.sandbox_path:
            return None
        return safe_path

    def read_file(self, path: str) -> str:
        """
        Reads the content of a file within the sandbox.
        """
        safe_path = self._get_safe_path(path)
        if not safe_path:
            return "Error: Path is outside the sandbox."
        try:
            with open(safe_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            return f"Error reading file: {e}"

    def write_file(self, path: str, content: str) -> str:
        """
        Writes content to a file within the sandbox.
        """
        safe_path = self._get_safe_path(path)
        if not safe_path:
            return "Error: Path is outside the sandbox."
        try:
            with open(safe_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return f"File written successfully to {path}"
        except Exception as e:
            return f"Error writing file: {e}"

    def list_directory(self, path: str) -> list[str] | str:
        """
        Lists the contents of a directory within the sandbox.
        """
        safe_path = self._get_safe_path(path)
        if not safe_path:
            return "Error: Path is outside the sandbox."
        try:
            return os.listdir(safe_path)
        except Exception as e:
            return f"Error listing directory: {e}"
