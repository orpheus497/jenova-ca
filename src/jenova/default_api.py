# The JENOVA Cognitive Architecture
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

"""This module provides the default API for the JENOVA Cognitive Architecture.
"""

import datetime
import os
import shlex
import subprocess

from selenium import webdriver
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options


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
        result = subprocess.run(
            command_args, capture_output=True, text=True, check=False, timeout=30)
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
            "error": result.returncode != 0
        }
    except FileNotFoundError:
        return {
            "stdout": "",
            "stderr": f"Command not found: {command.split()[0]}",
            "returncode": -1,
            "error": True
        }
    except subprocess.TimeoutExpired:
        return {
            "stdout": "",
            "stderr": "Command timed out after 30 seconds.",
            "returncode": -1,
            "error": True
        }
    except Exception as e:
        return {
            "stdout": "",
            "stderr": str(e),
            "returncode": -1,
            "error": True
        }


def web_search(query: str) -> list[dict] | str:
    """
    Performs a web search using DuckDuckGo and returns the results.
    """
    # Preserve CUDA environment for subprocess (geckodriver/Firefox)
    # This prevents CUDA context conflicts when spawning the browser
    import os
    original_cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')

    options = Options()
    options.headless = True
    driver = None
    try:
        # Temporarily hide CUDA from the browser subprocess to prevent conflicts
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

        driver = webdriver.Firefox(options=options)
        driver.get(f"https://duckduckgo.com/html/?q={query}")
        results = []
        # Limit to top 5 results
        for result in driver.find_elements(By.CLASS_NAME, "result")[:5]:
            title = result.find_element(By.CLASS_NAME, "result__title").text
            link = result.find_element(
                By.CLASS_NAME, "result__url").get_attribute("href")
            snippet = result.find_element(
                By.CLASS_NAME, "result__snippet").text
            results.append({"title": title, "link": link, "summary": snippet})
        return results
    except WebDriverException as e:
        return f"Error: Web search failed. Could not initialize browser. Please ensure Firefox is installed. Details: {e}"
    except Exception as e:
        return f"Error: An unexpected error occurred during web search: {e}"
    finally:
        # Always restore original CUDA visibility setting
        os.environ['CUDA_VISIBLE_DEVICES'] = original_cuda_visible
        if driver:
            try:
                driver.quit()
            except Exception:
                # Suppress cleanup errors
                pass


class FileTools:
    def __init__(self, sandbox_path: str):
        self.sandbox_path = os.path.realpath(os.path.expanduser(sandbox_path))
        if not os.path.exists(self.sandbox_path):
            os.makedirs(self.sandbox_path)

    def _get_safe_path(self, path: str) -> str | None:
        """
        Resolves a path to a real, absolute path within the sandbox.
        Returns None if the path is outside the sandbox or is a symlink.
        """
        # Normalize the user-provided path by removing any relative path components
        normalized_path = os.path.normpath(path)
        # Prevent absolute paths from being treated as relative
        if os.path.isabs(normalized_path):
            return None

        # Join with the sandbox root
        prospective_path = os.path.join(self.sandbox_path, normalized_path)

        # Get the real, absolute path, resolving any symlinks
        real_path = os.path.realpath(prospective_path)

        # Check if the resolved path is within the sandbox directory
        if os.path.commonprefix([self.sandbox_path, real_path]) != self.sandbox_path:
            return None

        return real_path

    def read_file(self, path: str) -> str:
        """
        Reads the content of a file within the sandbox.
        """
        safe_path = self._get_safe_path(path)
        if not safe_path:
            return "Error: Path is outside the sandbox or is invalid."
        try:
            with open(safe_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            return f"Error: File not found at '{path}'."
        except Exception as e:
            return f"Error reading file: {e}"

    def write_file(self, path: str, content: str) -> str:
        """
        Writes content to a file within the sandbox.
        """
        safe_path = self._get_safe_path(path)
        if not safe_path:
            return "Error: Path is outside the sandbox or is invalid."
        try:
            # Ensure parent directory exists
            parent_dir = os.path.dirname(safe_path)
            os.makedirs(parent_dir, exist_ok=True)

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
            return "Error: Path is outside the sandbox or is invalid."
        try:
            return os.listdir(safe_path)
        except FileNotFoundError:
            return f"Error: Directory not found at '{path}'."
        except Exception as e:
            return f"Error listing directory: {e}"
