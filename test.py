import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

test_csv = np.genfromtxt('test_labels.csv', delimiter=',', skip_header=1)
random_forest = np.genfromtxt('rf_submission.csv', delimiter=',', skip_header=1)
perceptron = np.genfromtxt('perceptron_submission.csv', delimiter=',', skip_header=1)
rnn = np.genfromtxt('rnn_submission.csv', delimiter=',', skip_header=1)
log_reg = np.genfromtxt('log_reg_submission_test2.csv', delimiter=',', skip_header=1)
mlp = np.genfromtxt('mlp_submission.csv', delimiter=',', skip_header=1)
gb = np.genfromtxt('gb_submission.csv', delimiter=',', skip_header=1)

test_csv = test_csv[:, :2] 

def compute_accuracy_ratio(test, prediction):
    correct_count = 0
    total_count = 0

    test_dict = {int(row[0]): row[1] for row in test}
    
    for row in prediction:
        index = int(row[0])
        predicted_value = row[1]
        
        if index in test_dict:
            total_count += 1
            if test_dict[index] == predicted_value:
                correct_count += 1
    
    return correct_count / total_count if total_count > 0 else 0

def compute_confusion_matrix(test_csv, predictions):
    test_dict = {int(row[0]): int(row[1]) for row in test_csv}

    true_labels = []
    predicted_labels = []

    for row in predictions:
        index = int(row[0])
        predicted_value = int(row[1])
        
        if index in test_dict:
            true_labels.append(test_dict[index])
            predicted_labels.append(predicted_value)

    cm = confusion_matrix(true_labels, predicted_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title('Confusion Matrix')
    plt.show()

rf_accuracy = compute_accuracy_ratio(test_csv, random_forest)
perceptron_accuracy = compute_accuracy_ratio(test_csv, perceptron)
rnn_accuracy = compute_accuracy_ratio(test_csv, rnn)
log_reg_accuracy = compute_accuracy_ratio(test_csv, log_reg)
mlp_accuracy = compute_accuracy_ratio(test_csv, mlp)
gb_accuracy = compute_accuracy_ratio(test_csv, gb)

print(f"Random Forest Accuracy Ratio: {rf_accuracy}")
print(f"Perceptron Accuracy Ratio: {perceptron_accuracy}")
print(f"RNN Accuracy Ratio: {rnn_accuracy}")
print(f"Logistic Regression Accuracy Ratio: {log_reg_accuracy}")
print(f"MLP Accuracy Ratio: {mlp_accuracy}")
print(f"Gradient Boosting Accuracy Ratio: {gb_accuracy}")

compute_confusion_matrix(test_csv, random_forest)
compute_confusion_matrix(test_csv, perceptron)
compute_confusion_matrix(test_csv, rnn)
compute_confusion_matrix(test_csv, log_reg)
compute_confusion_matrix(test_csv, mlp)
compute_confusion_matrix(test_csv, gb)

