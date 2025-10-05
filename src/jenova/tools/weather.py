import requests
import os
import re

class WeatherTool:
    def __init__(self, config, ui_logger, file_logger):
        self.config = config
        self.ui_logger = ui_logger
        self.file_logger = file_logger

    def handle_tool_request(self, response: str) -> str:
        weather_pattern = re.compile(r'<TOOL:GET_WEATHER\(location="([^"]+)"\)>')
        weather_match = weather_pattern.search(response)
        if weather_match:
            location = weather_match.group(1)
            self.ui_logger.info(f'Tool request detected: GET_WEATHER(location="{location}")')
            weather_info = self.get_weather(location)
            return weather_pattern.sub(weather_info, response)
        return response

    def get_weather(self, location: str) -> str:
        try:
            url = f"http://wttr.in/{location}?format=%C+%t"
            response = requests.get(url)
            response.raise_for_status()
            weather_data = response.text.strip()
            return f"The weather in {location} is: {weather_data}"
        except requests.exceptions.RequestException as e:
            self.file_logger.log_error(f"Error getting weather for {location}: {e}")
            return f"Error: Could not connect to the weather service."
        except Exception as e:
            self.file_logger.log_error(f"An unexpected error occurred while getting weather for {location}: {e}")
            return "An unexpected error occurred while fetching the weather."
