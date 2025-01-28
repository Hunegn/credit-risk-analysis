import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import logging
from xverse.transformer import WOE

logging.basicConfig(
    filename="../logs/feature_engineering.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class FeatureEngineering:
    def __init__(self, data_path, output_path):
        self.data_path = data_path
        self.output_path = output_path
        self.data = None
        logging.info("FeatureEngineering initialized.")

    def load_data(self):
        logging.info("Loading data...")
        self.data = pd.read_csv(self.data_path)
        logging.info(f"Data loaded with shape: {self.data.shape}")

    def create_aggregate_features(self):
        """
        Create aggregate features for each customer.
        """
        required_column = 'CustomerId'  
        if required_column not in self.data.columns:
            raise KeyError(f"'{required_column}' column not found in the dataset. Cannot compute aggregate features.")

        logging.info("Creating aggregate features...")
        self.data['TotalTransactionAmount'] = self.data.groupby('CustomerId')['Amount'].transform('sum')
        self.data['AverageTransactionAmount'] = self.data.groupby('CustomerId')['Amount'].transform('mean')
        self.data['TransactionCount'] = self.data.groupby('CustomerId')['Amount'].transform('count')
        self.data['StdTransactionAmount'] = self.data.groupby('CustomerId')['Amount'].transform('std')

        logging.info("Aggregate features created successfully.")

    def extract_features(self):
        """
        Extract features from existing columns.
        """
        logging.info("Extracting features from TransactionStartTime...")
        self.data['TransactionStartTime'] = pd.to_datetime(self.data['TransactionStartTime'])
        self.data['TransactionHour'] = self.data['TransactionStartTime'].dt.hour
        self.data['TransactionDay'] = self.data['TransactionStartTime'].dt.day
        self.data['TransactionMonth'] = self.data['TransactionStartTime'].dt.month
        self.data['TransactionYear'] = self.data['TransactionStartTime'].dt.year
        logging.info("Feature extraction complete.")

    def encode_categorical_features(self):
        """
            Perform Weight of Evidence (WOE) encoding for categorical variables.
        """
        logging.info("Encoding categorical variables using WOE...")
        woe_encoder = WOE()
        categorical_cols = ['CurrencyCode', 'ProviderId', 'ProductCategory', 'ChannelId']

        unique_values = self.data['FraudResult'].nunique()
        if unique_values != 2:
            logging.warning(f"Target 'FraudResult' is not binary (unique values: {unique_values}). Skipping WOE encoding.")
            return

        for col in categorical_cols:
            if col in self.data.columns:
                logging.info(f"Applying WOE encoding on {col}...")
                
                feature_df = self.data[[col]]
                woe_encoder.fit(feature_df, self.data['FraudResult'])
                self.data[col] = woe_encoder.transform(feature_df)
            else:
                logging.warning(f"Column '{col}' not found. Skipping WOE encoding.")

        logging.info("Categorical variables encoded successfully.")



    def normalize_numerical_features(self):
        """
        Normalize numerical features using StandardScaler.
        """
        logging.info("Normalizing numerical features...")
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        scaler = StandardScaler()
        self.data[numeric_cols] = scaler.fit_transform(self.data[numeric_cols])
        logging.info("Normalization complete.")

    def save_data(self):
        logging.info(f"Saving processed data to {self.output_path}...")
        self.data.to_csv(self.output_path, index=False)
        logging.info(f"Data saved successfully to {self.output_path}.")

    def run_feature_engineering(self):
        self.load_data()
        self.create_aggregate_features()
        self.extract_features()
        self.encode_categorical_features()
        self.normalize_numerical_features()
        self.save_data()


if __name__ == "__main__":
    fe = FeatureEngineering(
        data_path="../data/processed/cleaned_data.csv",
        output_path="../data/processed/feature_engineered_data.csv"
    )
    fe.run_feature_engineering()
