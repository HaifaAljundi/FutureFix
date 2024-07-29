<div align="center">
  <img src="https://github.com/user-attachments/assets/d690e851-3df1-4965-8af6-1b78473c7cef" alt="Elevator Predictive Maintenance" width="50%"/>
</div>

[![MIT License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)




# Elevator Predictive Maintenance
**Haifa Aljundi | Zyad Abdelhafiz**  
*Apsiyon Hackathon*

## Abstract
This project aims to predict future maintenance times for elevators by analyzing the Huawei German Research Center dataset. We use features such as revolutions, humidity, and vibration to develop two models: one binary model to determine if immediate maintenance is needed and another regression model to predict the number of days remaining until maintenance is required. The project involves data preprocessing, model training, and API integration for real-time testing. The models achieved 95% accuracy in predicting maintenance times for elevators, focusing on efficiency and reducing downtime to ensure safety.

Our project supports the following Sustainable Development Goals (SDGs):
- Goal 12: Responsible Consumption and Production
- Goal 11: Sustainable Cities and Communities
- Goal 9: Industry, Innovation, and Infrastructure

<p align="center">
    <img src="https://github.com/user-attachments/assets/30208ce8-e72c-42d6-84d1-610c8a250a7a" alt="9" width="20%">
    <img src="https://github.com/user-attachments/assets/8e52df22-cef5-4db5-a6ab-eaeb240a961a" alt="11" width="20%">
    <img src="https://github.com/user-attachments/assets/18dd624e-b6c7-43e6-a2fd-a1afea7d01ef" alt="12" width="20%">
</p>


## Table of Contents
1. [Introduction](#introduction)
2. [Prove of Concept](#prove-of-concept)
   - Economic Impact
   - Safety Improvements
   - Case Studies
   - Scalability
   - Technological Integration
   - Environmental Impact
3. [Data Collection](#data-collection)
4. [Data Preprocessing](#data-preprocessing)
5. [Feature Selection](#feature-selection)
6. [Model Selection and Training](#model-selection-and-training)
7. [Evaluation Metrics](#evaluation-metrics)
8. [Implementation](#implementation)
   - Model Implementation
   - API Integration
9. [Results](#results)
10. [Conclusion](#conclusion)
11. [References](#references)

## Introduction
Elevator maintenance is important for ensuring safety and minimizing downtime. Predictive maintenance helps determine the best time for servicing elevators by analyzing operational data. By using features such as humidity and vibration, we can predict when an elevator requires maintenance through trained models.

## Prove of Concept
1. **Economic Impact**
   - Saves money by reducing unexpected elevator breakdowns.
   - Extends the life of elevator parts, lowering replacement costs.
2. **Safety Improvements**
   - Prevents accidents by detecting issues early.
   - Ensures elevators are always in good working condition, enhancing user safety.
3. **Case Studies**
   - Provides examples from other industries where predictive maintenance has been successful.
   - Shows practical benefits and effectiveness through real-world examples.
4. **Scalability**
   - Can handle data from many elevators in different locations.
   - Suitable for large buildings or multiple properties managed by Apsiyon.
5. **Technological Integration**
   - Can be integrated with Apsiyon’s current building management systems.
   - Allows for real-time monitoring and alerts, improving response times.
6. **Environmental Impact**
   - Reduces energy use by maintaining elevators more efficiently.
   - Lowers the carbon footprint, supporting sustainability efforts.

## Data Collection
- **Source**: The dataset, [Kaggle Elevator Predictive Maintenance Dataset](https://www.kaggle.com/datasets/shivamb/elevator-predictive-maintenance-dataset). It includes operational data from elevators with features such as revolutions, humidity, and vibration.

## Data Preprocessing
- **Cleaning**: The data was cleaned to handle missing values by removing the missing entries from the vibration feature and removing sensors from 1 to 5.

## Feature Selection
- **Selected Features**: The features used in the model are revolutions, humidity, and vibration. These were chosen based on their relevance to elevator maintenance. Additionally, a maintenance date feature was created.

## Model Selection and Training
- **Chosen Model**: Binary Classification Model was selected to predict whether immediate maintenance is needed. And Regression Model was selected to predict the number of days remaining until the next maintenance, using the last maintenance date as a reference.
- **Training Process**: The model was trained on the Huawei German Research Center dataset with a focus on optimizing accuracy and minimizing overfitting.

## Evaluation Metrics
- **Metrics Used**: Accuracy, precision, recall, and F1 score were used to evaluate the model’s performance.
- **Results**: The model achieved an accuracy of 95% and demonstrated good performance in predicting maintenance dates.

- 
### Binary Classification

![image](https://github.com/user-attachments/assets/fcbbc668-7d28-4c69-80bb-d1ca327910f8)



![image](https://github.com/user-attachments/assets/bb34841a-9f97-4107-b373-6885abae0459)




### Regression 

![image](https://github.com/user-attachments/assets/3160be2e-d80a-4114-a260-5e2ca0573248)



![image](https://github.com/user-attachments/assets/dce4cacf-7b93-4cb8-9fb3-f6b1a3b39465)




## Implementation
### Model Implementation
- **Binary Classification Model**: A binary classification model was selected to predict whether immediate maintenance is needed. The decision tree algorithm was implemented using Scikit-Learn. Key steps included training the model with historical data and tuning hyperparameters for optimal performance.

![image](https://github.com/user-attachments/assets/907ae32d-db37-4b55-9901-cd7fc2ac135f)




- **Regression Model**: A regression model was selected to predict the number of days remaining until the next maintenance, using the last maintenance date as a reference. The decision tree algorithm was also used for this model, implemented using Scikit-Learn, with similar steps of training and hyperparameter tuning for optimal performance.



![image](https://github.com/user-attachments/assets/a873cf75-7f3b-4a60-8c76-84c6a0dd88d6)



### API Integration
- **Integration Process**: The model was linked to an API for real-time testing. This integration allows continuous monitoring and prediction of maintenance needs based on live data inputs.
- **Testing**: The API was tested to ensure accurate and timely predictions.

![Screen Shot 2024-07-27 at 8 00 21 PM](https://github.com/user-attachments/assets/3e788e95-8fcb-47b3-94ee-09e0e0590ddc)


## Results
- **Findings**: The model successfully predicted maintenance time needs with an accuracy of 95%. The results showed that the most significant predictors were revolutions and vibration.

## Conclusion
The project demonstrates the potential of machine learning for predictive maintenance in elevators. The model’s accuracy in predicting maintenance needs can help reduce downtime and improve safety. Future work could explore additional features and more advanced models to enhance prediction capabilities.

## References
1. Brown, M. T., & Smith, J. A. (2019). Predictive maintenance for industrial machinery. Springer.
2. Huawei German Research Center. (n.d.). Dataset on elevator maintenance. [Kaggle](https://www.kaggle.com/datasets/shivamb/elevator-predictive-maintenance-dataset).
3. Liaw, A., & Wiener, M. (2002). Classification and regression by randomForest. R News, 2(3).
4. Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., … & Vanderplas, J. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, 2825-2830.
5. Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. Journal of the Royal Statistical Society: Series B (Methodological).







