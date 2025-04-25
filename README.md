# Human Activity Recognition 🏃‍♂️🚶‍♀️ ![Grade](https://img.shields.io/badge/Grade-17%2F20-brightgreen)

## Overview

This repository contains our solution for the ELEN0062 - Introduction to Machine Learning final project on Human Activity Recognition. The project focuses on accurately classifying 14 different human activities using time series data from 31 sensors.

**Project Goal**: Develop a machine learning solution to predict which activity a person is performing based on sensor data.

### Dataset

The dataset consists of measurements from 31 sensors capturing 5-second time series data (512 data points each) from subjects performing various activities:

- **Activities**: Lying, Sitting, Standing, Walking very slow, Normal walking, Nordic walking, Running, Ascending stairs, Descending stairs, Cycling, Ironing, Vacuum cleaning, Rope jumping, Playing soccer
- **Sensors**: Include heart rate, temperature, and 3D measurements (acceleration, gyroscope, magnetometer) from hand, chest, and foot
- **Subjects**: Training data from 5 subjects, test data from 3 different subjects

## Methodology

### Data Preprocessing

1. **Missing Value Handling**: 
   - Learning set: Conditional mean imputation based on activity
   - Test set: Unconditional mean imputation

2. **Feature Engineering**:
   - Extracted 15 metrics from each time series including: average, deviation, minimum, maximum, median, skewness, kurtosis, signal energy, spectral centroid, and more

### Models Implemented

We explored several machine learning approaches:

| Model | Accuracy (%) |
|-------|--------------|
| Random Forest | 89.4 |
| XGBoost | 83.5 |
| Multi-Layer Perceptron | 75.6 |
| Logistic Regression | 73.2 |
| Perceptron | 68.0 |
| RNN (LSTM) | 60.1 |

**Best Model**: Random Forest achieved the highest accuracy while maintaining computational efficiency. It excelled at classifying most activities with minimal confusion between similar movements.

## Results & Insights

- Random Forest emerged as the best model with 89.4% accuracy, offering an excellent balance between performance and efficiency
- Some activities (e.g., "Nordic walking" vs "Normal walking") were commonly confused across multiple models
- Feature importance analysis revealed that foot and hand acceleration metrics were most significant for classification

## Repository Structure

- `final/`: Contains the final versions of all tested methods and the data handling file
- Data preprocessing scripts
- Model implementation and evaluation code

## Authors

This project was carried out by three students:
- Ihabe GUENFOUDI
- Renaux NTAKIRUTIMANA
- Robert FLOREA

## Resources

- [Final Report (This Project)](https://fr.overleaf.com/read/zjpdghhpbyrr#551ba5)
- [Project 1 Report](https://www.overleaf.com/read/jtwzvdcfxrqx#4219fa)
- [Project 2 Report](https://fr.overleaf.com/read/bstrdprwmmqp#966247)
