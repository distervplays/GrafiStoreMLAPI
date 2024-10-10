from urllib.parse import urljoin
import requests

class PremoAPI:
    def __init__(self) -> None:
        """
        This class is used to interact with the Premo API
        """
        self.domain: str = "https://premo.gshub.nl/"
        self.api_url: str = urljoin(self.domain, "api/dinand/")
        self.learning_days = 7
    
    @property
    def play_data(self) -> dict:
        url = urljoin(self.api_url, "play/")
        json_data = requests.get(url).json()
        return json_data
    
    @property
    def learn_data(self) -> dict:
        url = urljoin(self.api_url, f"learn/{self.learning_days}")
        json_data = requests.get(url).json()
        return json_data