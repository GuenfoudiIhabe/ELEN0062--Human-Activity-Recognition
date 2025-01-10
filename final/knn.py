import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import seaborn as sns

from sklearn.model_selection import train_test_split as split_data, RandomizedSearchCV as RandSearch
from sklearn.metrics import accuracy_score as acc_score, classification_report as class_report
from sklearn.preprocessing import StandardScaler as Scaler
from sklearn.pipeline import Pipeline as ML_Pipeline
from sklearn.neighbors import KNeighborsClassifier

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
        train_val_X, train_val_y, test_size=0.2, random_state=42, stratify=train_val_y
    )

    print("Setting up the model pipeline and hyperparameters...")
    model_pipeline = ML_Pipeline([
    ('scaler_step', Scaler()),
    ('knn_step', KNeighborsClassifier())
    ])

    parameters = {
        'knn_step__n_neighbors': np.arange(1, 401, 1),
        'knn_step__weights': ['uniform', 'distance'],        
        'knn_step__p': [1, 2],
        'knn_step__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'knn_step__leaf_size': np.arange(10, 51, 10),
        'knn_step__metric': ['minkowski', 'euclidean', 'manhattan'],
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
    print(f"Predictions completed in {prediction_time:.2f} seconds.")
    test_predictions = [p + 1 for p in test_predictions]

    submission = pd.DataFrame({
        'Id': np.arange(1, len(test_predictions) + 1),
        'Prediction': test_predictions
    })
    submission.to_csv('knn_submission.csv', index=False)
    print("Test predictions saved to 'knn_submission.csv'")

    sensor_cols = [col for col in train_val_X.columns if col.startswith("sensor_")]
    sensor_data = train_val_X[sensor_cols]
    
    results_df = pd.DataFrame(search.cv_results_)

    results_df = results_df.rename(columns={
        "param_knn_step__n_neighbors": "n_neighbors",
        "param_knn_step__weights": "weights",
        "param_knn_step__p": "p",
        "param_knn_step__algorithm": "algorithm",
        "param_knn_step__leaf_size": "leaf_size",
        "param_knn_step__metric": "metric",
        "mean_test_score": "mean_accuracy"
    })
        
    categorical_params = ["weights", "algorithm", "metric"]

    for param in categorical_params:
        plt.figure(figsize=(8, 6))
        sns.barplot(x=param, y="mean_accuracy", data=results_df, ci=None)
        plt.title(f"Impact of {param} on Accuracy")
        plt.xlabel(param.capitalize())
        plt.ylabel("Mean Accuracy")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
    numerical_params = ["n_neighbors", "leaf_size"]

    for param in numerical_params:
        plt.figure(figsize=(8, 6))
        sns.lineplot(x=param, y="mean_accuracy", data=results_df)
        plt.title(f"Impact of {param} on Accuracy")
        plt.xlabel(param.capitalize())
        plt.ylabel("Mean Accuracy")
        plt.tight_layout()
        plt.show()

    plt.figure(figsize=(8, 6))
    sns.barplot(x="p", y="mean_accuracy", data=results_df, ci=None)
    plt.title("Impact of Distance Metric (p) on Accuracy")
    plt.xlabel("p (Distance Metric)")
    plt.ylabel("Mean Accuracy")
    plt.tight_layout()
    plt.show()
    
    heatmap_data = results_df.pivot_table(index="n_neighbors", columns="weights", values="mean_accuracy")
    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, annot=True, fmt=".4f", cmap="coolwarm")
    plt.title("Impact of n_neighbors and Weights on Accuracy")
    plt.xlabel("Weights")
    plt.ylabel("n_neighbors")
    plt.tight_layout()
    plt.show()
