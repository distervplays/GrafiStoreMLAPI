from utils.UpdateDatabase import *
from dotenv import load_dotenv
from web import create_app
import os
load_dotenv()

debug = True if os.getenv("PRODUCTION") == "False" else False
host = os.getenv("HOST")
port = os.getenv("PORT")

app = create_app()

if __name__ == "__main__":
    app.run(debug=debug, host=host, port=port)
    