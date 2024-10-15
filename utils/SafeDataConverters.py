from bson import ObjectId
import pandas as pd

def safe_to_datetime(x):
    """
    Converts a given input to a datetime object if possible.
    This function attempts to convert the input `x` to a datetime object using the format 
    '%Y-%m-%dT%H:%M:%S.%fZ'. If the conversion fails due to a `ValueError` or `TypeError`, 
    the original input is returned unchanged. This is useful for safely handling data that 
    may not always be in the expected datetime format.
    
    Parameters:
        x: The input value to be converted to a datetime object. This can be any type, 
           but typically it is expected to be a string representing a date and time.
        
    Returns:
        datetime or original input: If the conversion is successful, a datetime object is 
        returned. If the conversion fails, the original input `x` is returned.
    """
    
    try:
        return pd.to_datetime(x, format='%Y-%m-%dT%H:%M:%S.%fZ')
    except (ValueError, TypeError):
        return x
    
def safe_to_int(x):
    """
    Converts a given input to an integer safely.
    This function attempts to convert the input `x` to an integer. If the input is an instance of `ObjectId`,
    it converts the `ObjectId` to its hexadecimal string representation and then to an integer. If the conversion
    fails due to a `ValueError` or `TypeError`, the original input is returned.
    
    Usage:
        - Converts `ObjectId` to an integer.
        - Converts other types to an integer if possible.
        - Returns the original input if conversion fails.
        
    Parameters:
        x: The input value to be converted to an integer. It can be of any type.
        
    Returns:
        int: The converted integer value if conversion is successful.
        original input: The original input `x` if conversion fails.
    """
    
    try:
        if isinstance(x, ObjectId):
            return int(str(x), 16)
        return int(x)
    except (ValueError, TypeError):
        return x