import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from xverse.transformer import WOE


class DefaultEstimator:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = pd.read_csv(data_path)
        print(self.data['TransactionStartTime'].head())


    def calculate_rfms(self):
        print("Calculating RFMS metrics...")
        self.data['TransactionStartTime'] = pd.to_datetime(self.data['TransactionStartTime'], utc=True)
        self.data['Recency'] = self.data.groupby('CustomerId')['TransactionStartTime'].transform(lambda x: (pd.Timestamp.now(tz="UTC") - x.max()).days)
        self.data['Seniority'] = self.data.groupby('CustomerId')['TransactionStartTime'].transform(lambda x: (pd.Timestamp.now(tz="UTC") - x.min()).days)
        self.data['Frequency'] = self.data.groupby('CustomerId')['TransactionId'].transform('count')
        self.data['Monetary'] = self.data.groupby('CustomerId')['Amount'].transform('sum')
        scaler = MinMaxScaler()
        self.data[['Recency', 'Frequency', 'Monetary', 'Seniority']] = scaler.fit_transform(self.data[['Recency', 'Frequency', 'Monetary', 'Seniority']])
        self.data['RFMS_Score'] = 0.25 * self.data['Recency'] + 0.25 * self.data['Frequency'] + 0.25 * self.data['Monetary'] + 0.25 * self.data['Seniority']

    def classify_credit_risk(self):
        print("Classifying credit risk...")
        threshold = self.data['RFMS_Score'].median()
        self.data['CreditRisk'] = self.data['RFMS_Score'].apply(lambda x: 'Good' if x >= threshold else 'Bad')

        plt.figure(figsize=(10, 6))
        sns.histplot(self.data['RFMS_Score'], kde=True, bins=30, color='blue')
        plt.axvline(threshold, color='red', linestyle='--', label='Threshold')
        plt.title('Distribution of RFMS Scores')
        plt.legend()
        plt.savefig("../plots/rfms_distribution.png")

    def apply_woe_binning(self):
    
        
        unique_values = self.data['FraudResult'].unique()
        if len(unique_values) != 2:
            raise ValueError(
                f"The target column 'FraudResult' must be binary. Found unique values: {unique_values}"
            )
        
       
        woe_encoder = WOE()
        categorical_features = self.data.select_dtypes(include=['object']).columns
        feature_df = self.data[categorical_features]
        woe_encoder.fit(feature_df, self.data['FraudResult'])
        self.data[categorical_features] = woe_encoder.transform(feature_df)


    def save_data(self):
        print("Saving processed data...")
        self.data.to_csv("../data/processed/credit_risk_with_rfms.csv", index=False)

    def run_all(self):
        self.calculate_rfms()
        self.classify_credit_risk()
        self.apply_woe_binning()
        self.save_data()

if __name__ == "__main__":
    estimator = DefaultEstimator(data_path="../data/processed/cleaned_data.csv")
    estimator.run_all()