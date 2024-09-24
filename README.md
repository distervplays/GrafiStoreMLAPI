# Task Planner using Neural Networks

This project is a Task Planner Module based on historical data, utilizing an LSTM model to predict future task durations and schedules.

### Project Structure

- `models.py`: Contains the LSTM model.
- `utils.py`: Contains the data preprocessing and utility functions.
- `main.py`: The main script to train the model and run inference.

### Installation

1. Clone the repository.
2. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the main script:
    ```bash
    python main.py
    ```

### Usage

- The model predicts task durations and schedules based on past data.
- Modify the `data` variable in `main.py` to include real historical data.

