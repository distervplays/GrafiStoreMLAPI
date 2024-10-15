from urllib.parse import urljoin
import requests

class PremoAPI:
    def __init__(self) -> None:
        """
        Initializes the PremoAPI class with default settings for domain, API URL, and learning days.

        This constructor sets up the initial configuration for interacting with the Premo API. It defines the base domain, constructs the full API URL, and sets the default number of learning days.

        Attributes:
            domain (str): The base URL for the Premo API.
            api_url (str): The full URL for accessing the Dinand API endpoint.
            learning_days (int): The default number of days for learning purposes.
        """
        
        self.domain: str = "https://premo.gshub.nl/"
        self.api_url: str = urljoin(self.domain, "api/dinand/")
        self.learning_days = 7
    
    @property
    def play_data(self) -> dict:
        """
        Fetches and returns play data from the API.
        This function constructs the URL for the "play" endpoint by joining the base API URL with the "play/" path.
        It then sends a GET request to this URL and parses the response as JSON.
        
        Returns:
            dict: A dictionary containing the play data fetched from the API.
        """
        
        url = urljoin(self.api_url, "play/")
        json_data = requests.get(url).json()
        return json_data
    
    @property
    def learn_data(self) -> dict:
        """
        Fetches learning data from the API.
        This function constructs a URL using the base API URL and the specified number of learning days,
        sends a GET request to the constructed URL, and returns the JSON response.
        
        Returns:
            dict: A dictionary containing the learning data fetched from the API.
        """
        
        url = urljoin(self.api_url, f"learn/{self.learning_days}")
        json_data = requests.get(url).json()
        return json_data