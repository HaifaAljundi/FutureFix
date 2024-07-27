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

![model_accuracy](https://github.com/user-attachments/assets/9b6f4d35-07bf-4be5-88d0-3cbdf53224f0)

![model_loss](https://github.com/user-attachments/assets/9d45c5e4-5d59-4390-a73b-e823e7fde5c3)

![model_mae](https://github.com/user-attachments/assets/21af487a-df50-4653-a383-e17f9fda5b56)

![model_regression_loss](https://github.com/user-attachments/assets/b01cacab-0be7-4308-a8e9-a6215e05aece)





## Implementation

1.	Model Implementation

•	Binary Classification Model:
A binary classification model was selected to predict whether immediate maintenance is needed. The decision tree algorithm was implemented using Scikit-Learn. Key steps included training the model with historical data and tuning hyperparameters for optimal performance.

![image](https://github.com/user-attachments/assets/2489433a-b67c-4e5c-8e2a-9fd9757422ae)

•	Regression Model:
A regression model was selected to predict the number of days remaining until the next maintenance, using the last maintenance date as a reference. The decision tree algorithm was also used for this model, implemented using Scikit-Learn, with similar steps of training and hyperparameter tuning for optimal performance.


![image](https://github.com/user-attachments/assets/53c0881c-4349-459b-a4e1-0e8e6daab7cd)






