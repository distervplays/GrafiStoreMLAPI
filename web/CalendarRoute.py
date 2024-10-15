from flask import Blueprint, jsonify, request
from utils.FlattenData import flatten_input
from Helpers.DataPreProcessing import preprocess_inputs, preprocess_data
from utils.SafeDataConverters import safe_to_datetime
from . import api, agent
import pandas as pd

calendar = Blueprint(
    name='calendar',
    import_name=__name__,
    url_prefix='/calendar'
)


@calendar.route('/')
def data():
    """
    Processes and predicts task durations and order creation times using preprocessed play data.
    This function performs several steps to process and predict task durations and order creation times:
        - Retrieves play data from an API.
        - Flattens the input play data.
        - Preprocesses the data for model input.
        - Sorts and ranks the play data by order ID and sort order.
        - Converts date columns to datetime format and drops rows with missing values.
        - Preprocesses the inputs for the prediction model.
        - Reindexes the preprocessed data to match the model's expected input columns.
        - Extracts necessary columns for prediction.
        - Uses a prediction agent to predict task durations and merges the predictions with the original data.
        - Cleans up and formats the prediction results.
        - Converts date columns to a specific string format.
        
    Returns:
        A JSON response containing the predicted task durations and order creation times.
    """
    
    play_data: list[dict] = api.play_data
    
    flat_play: list[dict] = flatten_input(play_data)
    df, X, _, yscaler = preprocess_data()

    flat_play: pd.DataFrame = pd.DataFrame(flat_play).sort_values(by=['order_id', 'sort_order'], ascending=True)
    flat_play['sort_order'] = flat_play.groupby(['order_id'])['sort_order'].transform(lambda x: x.rank(method='dense').astype(int) - 1)
    flat_play = flat_play.map(safe_to_datetime)
    flat_play = flat_play.dropna()
    
    preprocessed_play: list[dict] = preprocess_inputs(flat_play)
    preprocessed_play = preprocessed_play.reindex(columns=X.columns, fill_value=0)
    
    product_operation_id_list: list = flat_play['product_operation_id'].values
    order_created_at_list: list = flat_play['order_created_at'].values
    task_duration_list: list = flat_play['task_duration'].values
    
    pred = agent.predict(preprocessed_play, order_created_at_list, task_duration_list, product_operation_id_list, yscaler)
    pred = pd.DataFrame(pred).merge(flat_play, on='product_operation_id', how='left')
    
    pred['task_duration'] = pred['task_duration_x'].astype(float)
    pred['order_created_at'] = pred['order_created_at_x'].astype(str)
    pred = pred.drop(columns=['task_duration_x', 'task_duration_y', 'order_created_at_x', 'order_created_at_y'], axis=1)
    
    for col in ['order_created_at', 'delivery_date', 'start_at', 'end_at']:
        if col in pred.columns:
            pred[col] = pd.to_datetime(pred[col]).dt.strftime('%Y-%m-%dT%H:%M:%S')
    
    return jsonify(pred.to_dict(orient='records'))