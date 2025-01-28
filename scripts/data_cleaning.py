import pandas as pd
import numpy as np
import logging


logging.basicConfig(
    filename="../logs/data_cleaning.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class DataCleaner:
    def __init__(self, data_path, definitions_path, output_path):
        self.data_path = data_path
        self.definitions_path = definitions_path
        self.output_path = output_path
        self.data = None
        self.cleaned_data = None
        logging.info("DataCleaner initialized.")

    def load_data(self):
        logging.info("Loading data...")
        self.data = pd.read_csv(self.data_path)
        logging.info(f"Data loaded with shape: {self.data.shape}")

    def inspect_data(self):
        logging.info("Inspecting data...")
        logging.info(f"Columns: {self.data.columns}")
        logging.info(f"Missing values:\n{self.data.isnull().sum()}")

    def handle_missing_values(self):
        logging.info("Handling missing values...")
        
        for col in self.data.select_dtypes(include=[np.number]).columns:
            self.data[col] = self.data[col].fillna(self.data[col].median())
        
        
        for col in self.data.select_dtypes(include=[object]).columns:
            self.data[col] = self.data[col].fillna(self.data[col].mode()[0])
        logging.info("Missing values handled.")


    def handle_outliers(self):
        logging.info("Handling outliers...")
        
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        z_scores = (self.data[numeric_cols] - self.data[numeric_cols].mean()) / self.data[numeric_cols].std()
        self.data = self.data[(z_scores.abs() < 3).all(axis=1)]
        logging.info(f"Data shape after removing outliers: {self.data.shape}")

    def convert_data_types(self):
        logging.info("Converting data types...")
        
        if 'Date' in self.data.columns:
            self.data['Date'] = pd.to_datetime(self.data['Date'])
            logging.info("Converted 'Date' column to datetime.")

    def save_cleaned_data(self):
        logging.info("Saving cleaned data...")
        self.cleaned_data = self.data
        self.cleaned_data.to_csv(self.output_path, index=False)
        logging.info(f"Cleaned data saved to {self.output_path}")

    def run_cleaning_pipeline(self):
        self.load_data()
        self.inspect_data()
        self.handle_missing_values()
        self.handle_outliers()
        self.convert_data_types()
        self.save_cleaned_data()

if __name__ == "__main__":
    cleaner = DataCleaner(
        data_path="../data/raw/data.csv",
        definitions_path="../data/raw/Xente_Variable_Definitions.csv",
        output_path="../data/processed/cleaned_data.csv"
    )
    cleaner.run_cleaning_pipeline()
