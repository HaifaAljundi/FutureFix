# Elevator-Predictive-Maintenance

This project aims to predict future maintenance times for elevators by analyzing the Huawei German Research Center dataset. We use features such as revolutions, humidity, and vibration to develop two models: one binary model to determine if immediate maintenance is needed and another regression model to predict the number of days remaining until maintenance is required. The project involves data preprocessing, model training, and API integration for real-time testing. The models accuracy achieved 95% in predicting maintenance times for elevators, focusing on efficiency and reducing downtime to ensure safety.
Our project supports the following Sustainable Development Goals (SDGs):

•	Goal 12: Responsible Consumption and Production

•	Goal 11: Sustainable Cities and Communities

•	Goal 9: Industry, Innovation, and Infrastructure

![image](https://github.com/user-attachments/assets/77a0ca12-1f7d-41a3-bfeb-162700e42c88)






 ## Evaluation Metrics
•	Metrics Used: Accuracy, precision, recall, and F1 score were used to evaluate the model’s performance.

• Results: The model achieved an accuracy of 95% and demonstrated good performance in predicting maintenance date.

### Binary Classification

![image](https://github.com/user-attachments/assets/f47cba3c-40b4-40e8-9d20-67a8ebb55bb9)


![image](https://github.com/user-attachments/assets/3915f84c-229b-4103-9de3-8e9a381f86b1)



### Regression 

![image](https://github.com/user-attachments/assets/36c777cf-7f8f-42b5-a0a9-6e2df162d9d4)


![image](https://github.com/user-attachments/assets/2158506f-07bd-4444-b9c6-6fd27ecced66)






## Implementation

1.	Model Implementation

•	Binary Classification Model:
A binary classification model was selected to predict whether immediate maintenance is needed. The decision tree algorithm was implemented using Scikit-Learn. Key steps included training the model with historical data and tuning hyperparameters for optimal performance.

![image](https://github.com/user-attachments/assets/2489433a-b67c-4e5c-8e2a-9fd9757422ae)

•	Regression Model:
A regression model was selected to predict the number of days remaining until the next maintenance, using the last maintenance date as a reference. The decision tree algorithm was also used for this model, implemented using Scikit-Learn, with similar steps of training and hyperparameter tuning for optimal performance.


![image](https://github.com/user-attachments/assets/53c0881c-4349-459b-a4e1-0e8e6daab7cd)






