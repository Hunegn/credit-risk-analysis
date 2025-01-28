import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
import logging

# Setup logging
logging.basicConfig(
    filename="../logs/modeling.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
class ModelTraining:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = None
        self.models = {}
        logging.info("ModelTraining class initialized.")

    def load_data(self):
        """Load and preprocess data."""
        logging.info("Loading data...")
        self.data = pd.read_csv(self.data_path)
        logging.info(f"Data loaded with shape: {self.data.shape}")

        # Split data into features and target
        X = self.data.drop(columns=['FraudResult'])  # Replace 'FraudResult' with your target column name
        y = self.data['FraudResult']

        # Split into training and test sets
        logging.info("Splitting data into training and test sets...")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    def train_model(self, model, model_name, param_grid=None):
        """Train the model with optional hyperparameter tuning."""
        logging.info(f"Training {model_name}...")
        if param_grid:
            logging.info(f"Performing GridSearchCV for {model_name}...")
            grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='roc_auc')
            grid_search.fit(self.X_train, self.y_train)
            best_model = grid_search.best_estimator_
            logging.info(f"Best parameters for {model_name}: {grid_search.best_params_}")
        else:
            best_model = model.fit(self.X_train, self.y_train)

        self.models[model_name] = best_model

    def evaluate_model(self, model_name):
        """Evaluate the trained model."""
        logging.info(f"Evaluating {model_name}...")
        model = self.models[model_name]
        y_pred = model.predict(self.X_test)
        y_prob = model.predict_proba(self.X_test)[:, 1] if hasattr(model, 'predict_proba') else None

        # Metrics
        report = classification_report(self.y_test, y_pred)
        logging.info(f"Classification Report for {model_name}:\n{report}")
        print(f"\nClassification Report for {model_name}:\n{report}")

        # ROC-AUC
        if y_prob is not None:
            roc_auc = roc_auc_score(self.y_test, y_prob)
            fpr, tpr, _ = roc_curve(self.y_test, y_prob)
            plt.figure()
            plt.plot(fpr, tpr, label=f'{model_name} (ROC-AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
            plt.title(f'ROC Curve for {model_name}')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.legend(loc='best')
            plt.savefig(f"../plots/roc_curve_{model_name}.png")
            plt.close()
        else:
            logging.warning(f"{model_name} does not support probability predictions.")

    def run_all(self):
        """Run the complete training and evaluation pipeline."""
        self.load_data()

        # Logistic Regression
        self.train_model(LogisticRegression(max_iter=1000), "Logistic Regression", param_grid={
            'C': [0.1, 1, 10]
        })
        self.evaluate_model("Logistic Regression")

        # Decision Tree
        self.train_model(DecisionTreeClassifier(random_state=42), "Decision Tree", param_grid={
            'max_depth': [5, 10, 15],
            'min_samples_split': [2, 5, 10]
        })
        self.evaluate_model("Decision Tree")

        # Random Forest
        self.train_model(RandomForestClassifier(random_state=42), "Random Forest", param_grid={
            'n_estimators': [50, 100, 150],
            'max_depth': [5, 10, 15]
        })
        self.evaluate_model("Random Forest")

        # Gradient Boosting
        self.train_model(GradientBoostingClassifier(random_state=42), "Gradient Boosting", param_grid={
            'n_estimators': [50, 100, 150],
            'learning_rate': [0.01, 0.1, 0.2]
        })
        self.evaluate_model("Gradient Boosting")
if __name__ == "__main__":
    trainer = ModelTraining(data_path="../data/processed/cleaned_data.csv")
    trainer.run_all()
