import os
import numpy as np
import pandas as pd
import sys
import time

from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split as split_data, RandomizedSearchCV as RandSearch
from sklearn.metrics import accuracy_score as acc_score, classification_report as class_report
from sklearn.preprocessing import StandardScaler as Scaler
from sklearn.pipeline import Pipeline as ML_Pipeline
from data_handling import load_and_prepare, load_saved_data, save_data

if __name__ == '__main__':
    base_dir = './'
    processed_data_dir = './processed_data'

    if os.path.exists(processed_data_dir):
        print("Loading preprocessed data...")
        train_val_X, train_val_y, test_X = load_saved_data(processed_data_dir)
    else:
        print("Processing dataset...")
        train_val_X, train_val_y = load_and_prepare(base_dir, dataset='LS')
        test_X, _ = load_and_prepare(base_dir, dataset='TS')
        save_data(train_val_X, train_val_y, test_X, processed_data_dir)

    print("Splitting data into training and validation sets...")
    train_X, val_X, train_y, val_y = split_data(
        train_val_X, train_val_y, test_size=0.2, random_state=17, stratify=train_val_y
    )

    print("Setting up the model pipeline and hyperparameters...")
    model_pipeline = ML_Pipeline([
    ('scaler_step', Scaler()),
    ('log_reg_step', LogisticRegression(multi_class='multinomial', class_weight='balanced', max_iter=150, random_state=17))
    ])

    # Define hyperparameters for RandomizedSearchCV
    parameters = {
        'log_reg_step__C': [0.01, 0.1, 1, 10],
        'log_reg_step__solver': ['lbfgs', 'saga', 'sag', 'newton-cg'],
        'log_reg_step__max_iter': [100, 200, 300, 400, 500]
    }

    search = RandSearch(
        estimator=model_pipeline,
        param_distributions=parameters,
        n_iter=100,
        scoring='accuracy',
        cv=5,
        n_jobs=-1,
        verbose=2,
        random_state=42
    )
    search.fit(train_X, train_y)

    print(f"Optimal parameters: {search.best_params_}")
    print(f"Best cross-validation accuracy: {search.best_score_:.4f}")

    optimal_model = search.best_estimator_
    print("Assessing model performance on validation data...")
    val_predictions = optimal_model.predict(val_X)
    print(f"Validation Accuracy: {acc_score(val_y, val_predictions):.4f}")
    print("Detailed Classification Report:")
    print(class_report(val_y, val_predictions))


    print("Creating predictions for the test dataset...")

    start_time = time.time()
    test_predictions = optimal_model.predict(test_X)
    end_time = time.time()
    prediction_time = end_time - start_time

    test_predictions = [p + 1 for p in test_predictions]

    submission = pd.DataFrame({
        'Id': np.arange(1, len(test_predictions) + 1),
        'Prediction': test_predictions
    })
    submission.to_csv('log_reg_submission_test.csv', index=False)
    print("Test predictions saved to 'log_reg_submission.csv'")

    print(f"Predictions completed in {prediction_time:.2f} seconds.")

    sys.exit(0)