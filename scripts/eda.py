import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os


logging.basicConfig(
    filename="../logs/eda_detailed.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class EDA:
    def __init__(self, cleaned_data_path):
        self.cleaned_data_path = cleaned_data_path
        self.data = None
        logging.info("EDA initialized.")

    def load_data(self):
        """
        Load cleaned data from the specified path.
        """
        logging.info("Loading cleaned data...")
        self.data = pd.read_csv(self.cleaned_data_path)
        logging.info(f"Data loaded with shape: {self.data.shape}")

    def overview_data(self):
        """
        Provide an overview of the data structure.
        """
        logging.info("Generating data overview...")
        print("Dataset Overview:")
        print(f"Number of rows: {self.data.shape[0]}")
        print(f"Number of columns: {self.data.shape[1]}")
        print("\nColumn Data Types:")
        print(self.data.dtypes)
        print("\nFirst 5 Rows:")
        print(self.data.head())

    def summary_statistics(self):
        """
        Display summary statistics of the dataset.
        """
        logging.info("Generating summary statistics...")
        print("\nSummary Statistics:")
        print(self.data.describe(include='all'))

    def plot_numerical_distributions(self):
        """
        Plot the distribution of numerical features.
        """
        logging.info("Plotting numerical feature distributions...")
        numeric_cols = self.data.select_dtypes(include=[float, int]).columns
        for col in numeric_cols:
            plt.figure(figsize=(8, 4))
            sns.histplot(self.data[col], kde=True, bins=30, color='blue')
            plt.title(f"Distribution of {col}")
            plt.xlabel(col)
            plt.ylabel("Frequency")
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.savefig(f"../plots/distribution_{col}.png")
            plt.show()

    def plot_categorical_distributions(self, top_n=10):
        """
        Plot the distribution of categorical features.
        Args:
            top_n (int): Number of top categories to display in the plots.
        """
        logging.info("Plotting categorical feature distributions...")
        categorical_cols = self.data.select_dtypes(include=['object']).columns

        
        plots_dir = "../plots"
        os.makedirs(plots_dir, exist_ok=True)

        for col in categorical_cols:
            plt.figure(figsize=(8, 4))
            
            
            value_counts = self.data[col].value_counts().nlargest(top_n)
            sns.barplot(y=value_counts.index, x=value_counts.values, palette='viridis')
            
            plt.title(f"Distribution of Top {top_n} {col} Categories")
            plt.xlabel("Count")
            plt.ylabel(col)
            plt.grid(axis='x', linestyle='--', alpha=0.7)
            plt.savefig(f"{plots_dir}/distribution_{col}.png")
            plt.show()


    def correlation_analysis(self):
        """
        Analyze and visualize the correlation matrix for numerical features.
        """
        logging.info("Analyzing correlations...")
        numeric_cols = self.data.select_dtypes(include=[float, int])
        corr_matrix = numeric_cols.corr()
        plt.figure(figsize=(12, 8))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", square=True)
        plt.title("Correlation Matrix")
        plt.savefig("../plots/correlation_matrix.png")
        plt.show()



   
