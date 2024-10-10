from bson import ObjectId
import pandas as pd

def safe_to_datetime(x):
    try:
        return pd.to_datetime(x, format='%Y-%m-%dT%H:%M:%S.%fZ')
    except (ValueError, TypeError):
        return x
    
def safe_to_int(x):
    try:
        if isinstance(x, ObjectId):
            return int(str(x), 16)
        return int(x)
    except (ValueError, TypeError):
        return x