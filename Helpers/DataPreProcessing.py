from utils.UpdateDatabase import update_database
from utils.SafeDataConverters import *
from utils.FlattenData import *
from web import db
import random

from modules.ColumnTransformer import ColumnTransformer
from sklearn.preprocessing import StandardScaler

def preprocess_inputs(df: pd.DataFrame):
    """ 
    Preprocesses the input DataFrame by handling missing values, computing time differences, 
    extracting date-time features, and applying transformations to categorical and numerical features.
    
    This function performs the following steps:
        - Drops rows with missing values.
        - Computes the time difference between 'delivery_date' and 'order_created_at' in hours.
        - Extracts the day of the week and time of day from 'order_created_at'.
        - One-hot encodes categorical variables and standardizes numerical variables using ColumnTransformer.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame containing order and delivery information.
            Expected columns:
            - 'delivery_date': The delivery date and time.
            - 'order_created_at': The order creation date and time.
            - 'task_title': The title of the task (categorical).
            - 'color': The color associated with the task (categorical).
            - 'material': The material associated with the task (categorical).
            - 'workspace': The workspace where the task is performed (categorical).
            - 'sort_order': The sort order of the task (numerical).
            - 'task_duration': The duration of the task (numerical).
            
    Returns:
        pd.DataFrame: A DataFrame with the transformed features ready for model input.
            The transformed features include:
            - One-hot encoded categorical features.
            - Standardized numerical features.
            - Computed 'delivery_offset' in hours.
            - Extracted 'order_day_of_week' and 'order_time_of_day'.
    """
    df = df.dropna()
    # Compute time differences in hours
    df['delivery_offset'] = (df['delivery_date'] - df['order_created_at']).dt.total_seconds() / 3600.0
    
    # Extract day of the week and time of day
    df['order_day_of_week'] = df['order_created_at'].dt.weekday  # 0 = Monday
    df['order_time_of_day'] = df['order_created_at'].dt.hour + df['order_created_at'].dt.minute / 60.0
    
    # One-hot encode categorical variables and standardize numerical variables
    ct = ColumnTransformer(
        ['task_title', 'color', 'material', 'workspace'],
        ['delivery_offset', 'sort_order', 'task_duration', 'order_day_of_week', 'order_time_of_day'],
    )
    df = ct.fit_transform(df)
    X_columns = ct.get_feature_names_out()
    X = df[X_columns]

    return X

def preprocess_outputs(df: pd.DataFrame):
    """
    Preprocesses the input DataFrame to generate normalized output features for machine learning models.
    
    This function performs the following steps:
        - Drops rows with missing values.
        - Calculates the time difference between the order creation and the start of the order in days.
        - Extracts the start time of the order in hours.
        - Normalizes the calculated features using StandardScaler.
        - Selects the normalized features to be used as output.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame containing the order data. It must have the following columns:
            - 'start_at' (datetime): The start time of the order.
            - 'order_created_at' (datetime): The creation time of the order.
            
    Returns:
        tuple: A tuple containing:
            - y (pd.DataFrame): A DataFrame with the normalized 'date_offset' and 'start_time' features.
            - scaler (StandardScaler): The scaler object used to normalize the features.
    """
    df = df.dropna()

    # Calculate the time difference between the order creation and the start of the order
    df['date_offset'] = (df['start_at'] - df['order_created_at']).dt.total_seconds()
    df['date_offset'] = df['date_offset'] / (3600 * 24) # Convert to days

    # Calculate the start time of the order
    df['start_time'] = df['start_at'].dt.hour + df['start_at'].dt.minute / 60.0

    # Normalize the values
    scaler = StandardScaler()
    df[['date_offset', 'start_time']] = scaler.fit_transform(df[['date_offset', 'start_time']])
    
    # Select the columns to be used as output
    y = df[['date_offset', 'start_time']]# .values
    return y, scaler

def preprocess_data():
    """Preprocesses the data for machine learning tasks.
    
    This function performs several preprocessing steps on the data, including:
        - Updating the database with a specific parameter.
        - Flattening the database output.
        - Creating a DataFrame from the flattened data and sorting it by 'order_id' and 'sort_order'.
        - Adjusting the 'sort_order' within each 'order_id' and 'product_id' group.
        - Converting date columns to datetime format safely.
        - Dropping rows with missing values.
        - Adding a random number of rows to the DataFrame.
        - Sorting the DataFrame by 'order_created_at' and 'sort_order'.
        - Grouping by 'order_id' and sorting within each group by 'sort_order'.
        - Preprocessing the inputs and outputs for machine learning models.
    
    Returns:
        tuple: A tuple containing the following elements:
            - df (pd.DataFrame): The preprocessed DataFrame.
            - X (pd.DataFrame): The input features for the machine learning model.
            - y (pd.Series): The target variable for the machine learning model.
            - yscaler (object): The scaler used to preprocess the target variable.
    """
    
    update_database(7)
    
    db_flat = flatten_output(db.to_json())
    
    df = pd.DataFrame(db_flat).drop(columns=['product_operation_id'], axis=1).sort_values(by=['order_id', 'sort_order'], ascending=True)
    df['sort_order'] = df.groupby(['order_id', 'product_id'])['sort_order'].transform(lambda x: x.rank(method='dense').astype(int) - 1)
    df = df.map(safe_to_datetime)
    df = df.dropna()
    
    # Add random rows to the input and output dataframes
    num_random_rows = random.randint(15, 50)
    df = df.sort_values(by=['order_created_at', 'sort_order'], ascending=True)
    df = df.groupby('order_id').apply(lambda x: x.sort_values(by='sort_order')).reset_index(drop=True)
    
    X = preprocess_inputs(df)
    y, yscaler = preprocess_outputs(df)
    
    return df, X, y, yscaler