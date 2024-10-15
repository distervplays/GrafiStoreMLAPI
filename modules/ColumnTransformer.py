from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
import pandas as pd


class ColumnTransformer:
    def __init__(self, categorical_features: list[str], numerical_features: list[str]) -> None:
        """
        Initializes the ColumnTransformer with specified categorical and numerical features.
            This class is designed to handle the transformation of both categorical and numerical features
            for machine learning models. It initializes the necessary attributes for encoding and scaling
            the features.
            
        Parameters:
            categorical_features (list[str]): A list of column names that are categorical features.
            numerical_features (list[str]): A list of column names that are numerical features.
            
        Attributes:
            categorical_features (list[str]): Stores the names of the categorical features.
            numerical_features (list[str]): Stores the names of the numerical features.
            one_hot_encoders (dict): A dictionary to hold one-hot encoders for categorical features.
            scalers (dict): A dictionary to hold scalers for numerical features.
            _feature_names_out (list): A list to store the names of the transformed features.
            _feature_names_in (list): A list to store the names of the input features.
            
        """
        
        self.categorical_features: list[str] = categorical_features
        self.numerical_features: list[str] = numerical_features
        self.one_hot_encoders: dict = {}
        self.scalers: dict = {}
        self._feature_names_out: list[str] = []
        self._feature_names_in: list[str] = []
        
    def fit(self, X: pd.DataFrame) -> None:
        """
        Fits the ColumnTransformer to the provided DataFrame.
        This method processes the input DataFrame to fit one-hot encoders for categorical features
        and scalers for numerical features. It updates the internal dictionaries and lists to store
        the fitted transformers and the names of the transformed features.
        
        Parameters:
            X (pd.DataFrame): The input DataFrame containing the features to be transformed.
            
        Attributes:
            categorical_features (list[str]): A list of column names that are categorical features.
            numerical_features (list[str]): A list of column names that are numerical features.
            one_hot_encoders (dict): A dictionary to hold one-hot encoders for categorical features.
            scalers (dict): A dictionary to hold scalers for numerical features.
            _feature_names_out (list): A list to store the names of the transformed features.
            _feature_names_in (list): A list to store the names of the input features.
        """
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
        """
        Transforms the input DataFrame by applying one-hot encoding to categorical features and scaling to numerical features.
        
        This method processes the input DataFrame `X` by:
            - Applying one-hot encoding to the specified categorical features using pre-fitted encoders.
            - Scaling the specified numerical features using pre-fitted scalers.
            - Retaining any additional features that were not specified in the categorical or numerical feature lists.
        
        Parameters:
            X (pd.DataFrame): The input DataFrame to be transformed. It should contain the features specified in 
                              `categorical_features` and `numerical_features`.
                              
        Returns:
            pd.DataFrame: A transformed DataFrame with one-hot encoded categorical features, scaled numerical features, 
                          and any additional features that were not transformed.
        """
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
        """
        Applies fit and transform operations on the input DataFrame.
        This method first fits the transformer to the input DataFrame `X` and then 
        transforms `X` using the fitted transformer. It is a convenient method that 
        combines the `fit` and `transform` operations into a single call.
        
        Parameters:
            X (pd.DataFrame): The input DataFrame to be transformed. It contains the 
                              features that need to be fitted and transformed.
                              
        Returns:
            pd.DataFrame: The transformed DataFrame after applying the fit and transform 
                          operations.
        """
        
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Inverse transform the given DataFrame to its original form.
        This function reverses the transformations applied to the categorical and numerical features
        of the DataFrame. It uses the stored one-hot encoders for categorical features and scalers
        for numerical features to perform the inverse transformation.
        
        - For categorical features, it identifies the one-hot encoded columns, applies the inverse
            transformation using the corresponding one-hot encoder, and appends the result to the output.
        - For numerical features, it applies the inverse transformation using the corresponding scaler
            and appends the result to the output.
        - For any other features that were not transformed, it directly appends them to the output.
        
        If no features are available for inverse transformation, it raises a ValueError.
        Parameters:
                X (pd.DataFrame): The DataFrame containing the transformed features.
                
        Returns:
                pd.DataFrame: The DataFrame with the original features restored.
        """
        
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
        """
        Resets the internal state of the ColumnTransformer.
        This method reinitializes the internal dictionaries and lists used for storing
        one-hot encoders, scalers, and feature names. It is useful for clearing any
        previously stored transformations and starting fresh.
        
        Attributes:
            one_hot_encoders (dict): A dictionary to hold one-hot encoders for categorical features.
            scalers (dict): A dictionary to hold scalers for numerical features.
            _feature_names_out (list): A list to store the names of the transformed features.
            _feature_names_in (list): A list to store the names of the input features.
            
        """
        
        self.one_hot_encoders = {}
        self.scalers = {}
        self._feature_names_out = []
        self._feature_names_in = []
    
    def get_feature_names_out(self) -> list[str]:
        """
        Retrieves the names of the transformed features.
        This method returns the names of the features after they have been transformed
        by the ColumnTransformer. It is useful for understanding the output of the 
        transformation process, especially when dealing with pipelines that include 
        multiple steps of feature engineering.
        
        Returns:
            list[str]: A list of strings representing the names of the transformed features.
        """
        
        return self._feature_names_out
    
    def get_feature_names_in(self) -> list[str]:
        """
        Retrieves the names of the input features used in the transformation process.
        This method returns a list of feature names that were used as input to the column transformer.
        It is useful for understanding which features were included in the transformation and for
        ensuring consistency in feature names across different stages of the machine learning pipeline.
        
        Returns:
            list[str]: A list of strings representing the names of the input features.
        """
        
        return self._feature_names_in
    