# GrafiStoreMLPreAPI Documentation

## Overview

GrafiStoreMLPreAPI is a machine learning-based API designed to interact with the Premo API, preprocess data, and update a MongoDB database. It also includes functionalities for training and evaluating machine learning models.

## Project Structure
```
├── Helpers/
│   └── DataPreProcessing.py
├── LICENSE
├── main.py
├── model.py
├── modules/
│   ├── database/
│   ├── ColumnTransformer.py
│   └── PremoAPI.py
├── README.md
├── requirements.txt
├── utils/
│   ├── FlattenData.py
│   ├── SafeDataConverters.py
│   └── UpdateDatabase.py
└── web/
    ├── __init__.py
    ├── CalendarRoute.py
    └── MainRoute.py
```


## Installation

1. Clone the repository.
2. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the main script:
    ```bash
    python main.py
    ```

## Usage

### API Endpoints

#### Home Route

Handles the home route for the API.

- **GET**: Returns a welcome message.
- **POST**: Expects a JSON payload containing the number of learning days, updates the database accordingly, and returns a success message.

```python
@views.route('/', methods=['POST'])
def home():
    ...
```

***Play Data Route***

Flattens the play data and returns it as a JSON response.

```python
@views.route('/play_data')
def flat_data():
    ...
```

**Train Route**

Handles the training and evaluation of a machine learning model.

* **POST**: Starts the training process with specified parameters.
* **GET**: Evaluates the model and returns the loss on the test data.

```python
@views.route('/train', methods=['POST', 'GET'])
def train():
    ...
```

**Modules**

**PremoAPI**

Interacts with the Premo API to fetch play and learning data.

```python
class PremoAPI:
    def __init__(self) -> None:
        ...
    
    @property
    def play_data(self) -> dict:
        ...
    
    @property
    def learn_data(self) -> dict:
        ...
```

**Data Preprocessing**

Preprocesses data for model input.

```python
def preprocess_data():
    ...
```

**Flatten Input**

Flattens the input play data.

```python
def flatten_input(play_data):
    ...
```

**Update Database**

Updates the MongoDB database with new data.

```python
def update_database(learning_days: int):
    ...
```

**Update Database**

Updates the MongoDB database with new data.

```python
def update_database(learning_days: int):
    ...
```

**License**

This project is licensed under the MIT License. <br>
See the [LICENSE](LICENSE) file for details.

