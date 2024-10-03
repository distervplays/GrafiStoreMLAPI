from pandas import pd

class LabelEncoder:
    def __init__(self) -> None:
        self.mapping = {}
        
    def fit(self, data: pd.Series):
        self.mapping = {v: i for i, v in enumerate(data.unique())}
    
    def transform(self, data: pd.Series):
        return data.map(self.mapping)
    
    def fit_transform(self, data: pd.Series):
        self.fit(data)
        return self.transform(data)
    
    def inverse_transform(self, data: pd.Series):
        return data.map({v: k for k, v in self.mapping.items()})
    
    def get_mapping(self):
        return self.mapping
    
class ColumnTransformer():
    def __init__(self, categorical_features: list) -> None:
        self.label_encoders = {}
        self.categorical_features = categorical_features
        
    def fit(self, data: pd.DataFrame):
        for col in self.categorical_features:
            le = LabelEncoder()
            le.fit(data[col])
            self.label_encoders[col] = le
            
    def transform(self, data: pd.DataFrame):
        for col in self.categorical_features:
            data[col] = self.label_encoders[col].transform(data[col])
        return data
    
    def fit_transform(self, data: pd.DataFrame):
        self.fit(data)
        return self.transform(data)
    
    def inverse_transform(self, data: pd.DataFrame):
        for col in self.categorical_features:
            data[col] = self.label_encoders[col].inverse_transform(data[col])
        return data