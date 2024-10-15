from modules.PremoAPI import PremoAPI
from modules.database import Database
from modules.database.Orders import Orders
from modules.database.Products import Products
from modules.database.TaskCards import TaskCards
from modules.database.ProductOperations import Operations
from flask import Flask
from model import Model
import os

username = os.getenv("mongodb_user")
password = os.getenv("mongodb_password")
db = Database(username, password)
orders = Orders(username, password)
products = Products(username, password)
taskcards = TaskCards(username, password)
operations = Operations(username, password)
    
api = PremoAPI()
agent = Model()
agent.model_path = os.path.join(os.getcwd(), 'model_weights')

def create_app():
    """
    Creates and configures the Flask application instance.
        This function initializes a Flask application, sets its configuration, loads necessary weights for the agent,
        and registers the main and calendar routes. It is typically used to set up the application with all its routes
        and configurations before running the server.
        
        Features:
            - Initializes the Flask application.
            - Configures the application with a secret key from environment variables.
            - Loads weights for the agent.
            - Registers the main and calendar blueprints for routing.
            
        Returns:
            app (Flask): The configured Flask application instance.
    """
    
    app = Flask(__name__)
    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')
    agent.load_weights()
    from .MainRoute import views
    from .CalendarRoute import calendar
    
    app.register_blueprint(views)
    app.register_blueprint(calendar)
    
    return app