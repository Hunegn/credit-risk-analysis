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
        
        # Check initial shape
        logging.info(f"Initial shape of data: {self.data.shape}")

        # Impute numerical columns with median
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if self.data[col].isnull().sum() > 0:
                self.data[col] = self.data[col].fillna(self.data[col].median())

        # Impute categorical columns with mode
        categorical_cols = self.data.select_dtypes(include=[object]).columns
        for col in categorical_cols:
            if self.data[col].isnull().sum() > 0:
                self.data[col] = self.data[col].fillna(self.data[col].mode()[0])

        # Log remaining missing values
        missing_summary = self.data.isnull().sum()
        logging.info(f"Remaining missing values:\n{missing_summary}")

        # Drop columns with too many missing values (threshold: 80%)
        missing_threshold = 0.8
        cols_to_drop = missing_summary[missing_summary > len(self.data) * missing_threshold].index
        self.data.drop(columns=cols_to_drop, inplace=True)
        logging.info(f"Dropped columns with > {missing_threshold*100}% missing values: {cols_to_drop.tolist()}")

        # Check final shape
        logging.info(f"Final shape of data after handling missing values: {self.data.shape}")



    def handle_outliers(self):
        logging.info("Handling outliers...")
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns

        initial_shape = self.data.shape
        for col in numeric_cols:
            Q1 = self.data[col].quantile(0.25)
            Q3 = self.data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Log the number of outliers for each column
            outliers = ((self.data[col] < lower_bound) | (self.data[col] > upper_bound)).sum()
            logging.info(f"Column {col}: {outliers} outliers detected")

            # Remove outliers (consider skipping if removing too many rows)
            self.data = self.data[(self.data[col] >= lower_bound) & (self.data[col] <= upper_bound)]

        logging.info(f"Data shape after removing outliers: {self.data.shape}")
        if self.data.empty:
            logging.warning("Warning: All rows removed after outlier handling!")
            self.data = self.original_data.copy()  # Restore data if completely removed


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
