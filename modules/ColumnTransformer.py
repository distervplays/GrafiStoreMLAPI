from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
import pandas as pd


class ColumnTransformer:
    def __init__(self, categorical_features: list[str], numerical_features: list[str]):
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.one_hot_encoders = {}
        self.scalers = {}
        self._feature_names_out = []
        self._feature_names_in = []
        
    def fit(self, X: pd.DataFrame) -> None:
        for feature in self.categorical_features:
            if feature not in X.columns:
                continue
            ohe = OneHotEncoder(sparse_output=False)
            ohe.fit(X[[feature]])
            self.one_hot_encoders[feature] = ohe
            
            self._feature_names_out.extend(ohe.get_feature_names_out([feature]))
            self._feature_names_in.append(feature)
            
        for feature in self.numerical_features:
            if feature not in X.columns:
                continue
            scaler = MinMaxScaler()
            scaler.fit(X[[feature]])
            self.scalers[feature] = scaler
            
            self._feature_names_in.append(feature)
            self._feature_names_out.append(feature)
            
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = []
        for feature in self.categorical_features:
            if feature not in X.columns:
                continue
            ohe = self.one_hot_encoders[feature]
            transformed = ohe.transform(X[[feature]])
            X_out.append(pd.DataFrame(transformed, 
                                        columns=ohe.get_feature_names_out([feature]), 
                                        index=X.index))
            
        for feature in self.numerical_features:
            if feature not in X.columns:
                continue
            scaler = self.scalers[feature]
            transformed = scaler.transform(X[[feature]])
            X_out.append(pd.DataFrame(transformed, columns=[feature], index=X.index))
        
        for feature in X.columns:
            if feature not in self._feature_names_in:
                X_out.append(X[[feature]])
        
        return pd.concat(X_out, axis=1)
    
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = []
        for feature in self.categorical_features:
            ohe_feature_names = [col for col in X.columns if col.startswith(feature + '_')]
            if not ohe_feature_names:
                continue
            ohe = self.one_hot_encoders[feature]
            X_out.append(pd.DataFrame(ohe.inverse_transform(X[ohe_feature_names]), columns=[feature], index=X.index))
            
        for feature in self.numerical_features:
            if feature not in X.columns:
                continue
            scaler = self.scalers[feature]
            X_out.append(pd.DataFrame(scaler.inverse_transform(X[[feature]]), columns=[feature], index=X.index))
            
        for feature in X.columns:
            if feature not in self._feature_names_out:
                X_out.append(X[[feature]])
            
        if not X_out:
            raise ValueError("No objects to concatenate")
            
        return pd.concat(X_out, axis=1)
    
    def reset(self) -> None:
        self.one_hot_encoders = {}
        self.scalers = {}
        self._feature_names_out = []
        self._feature_names_in = []
    
    def get_feature_names_out(self) -> list[str]:
        return self._feature_names_out
    
    def get_feature_names_in(self) -> list[str]:
        return self._feature_names_in
    