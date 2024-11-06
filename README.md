# Elevator Pitch

This project applies machine learning techniques to predict flight delays using PySpark, focusing on efficient data processing and model evaluation. The dataset, sourced from Kaggle, contains flight details such as carrier, origin, distance, and departure time. We use a Logistic Regression model within a pipeline that includes data preprocessing, feature engineering, and cross-validation for tuning hyperparameters. The model's performance is assessed using precision, recall, F1 score, and AUC metrics, offering insights into improving predictions of whether a flight will be delayed. This project highlights the power of PySpark in handling large-scale datasets and building scalable predictive models.

## Predicting Flight Delays using PySpark and Machine Leaning

1.	Introduction
   
In this report, we explore the application of machine learning techniques to predict flight delays using PySpark, a powerful framework for distributed data processing. Flight delays pose significant challenges for airlines, airports, and passengers, making predictive models valuable tools for improving operational efficiency and passenger experience. By leveraging historical flight data, we aim to build and evaluate a classification model to determine whether a flight will arrive late (Ball et al., 2024).

The dataset used in this analysis, sourced from Kaggle, includes various features such as the month, day of the month, day of the week, carrier information, flight number, origin airport, flight distance (in miles), scheduled departure time, flight duration, and the actual delay time. The primary objective is to predict the delay status of flights, classifying them as either delayed or on-time (Kaggle, n.d.). Furthermore, we employ a Logistic Regression model within a pipeline that includes several stages for data preprocessing and feature engineering. The pipeline includes transforming categorical features, assembling feature vectors, scaling numeric features, and finally, training the logistic regression model. We also utilize cross-validation to tune hyperparameters and optimize the model. 

Likewise, the performance of the model is evaluated using key metrics such as precision, recall, F1 score, and the Area Under the ROC Curve (AUC). These metrics help us understand the model's ability to accurately predict flight delays and its overall robustness (Sokolova and Lapalme, 2009). Hence, through this analysis, we aim to demonstrate the effectiveness of using PySpark for large-scale machine learning tasks and provide insights into improving flight delay predictions using advanced data processing and modeling techniques.

2.	Dataset and Data Analysis
   
2.1.	Overview of Dataset

The dataset used in this analysis, sourced from Kaggle, provides comprehensive details about various flights, including the factors that could potentially influence delays (Kaggle, n.d.). There are all together 10 columns and 275001 rows which qualifies this data for big data analysis. Below is a summary of the dataset's key features:

•	mon: Month of the flight (e.g., 1 for January, 2 for February)

•	dom: Day of the month when the flight occurred

•	dow: Day of the week (e.g., 1 for Monday, 2 for Tuesday)

•	carrier: The airline carrier code (e.g., OO for SkyWest Airlines, B6 for JetBlue Airways)

•	flight: Flight number assigned by the airline

•	org: Origin airport code (e.g., ORD for Chicago O'Hare, JFK for John F. Kennedy)

•	mile: Distance of the flight in miles

•	depart: Scheduled departure time (in 24-hour format)

•	duration: Scheduled flight duration in minutes

•	delay: Actual delay in minutes (negative values indicate early arrivals, positive values indicate delays)

3. Logistic Regression
   
After evaluating the Decision Tree model, we explore Logistic Regression as an alternative classification algorithm. Logistic Regression is a popular and straightforward method for binary classification problems. It models the probability that a given input point belongs to a certain class by using a logistic function (Hosmer Jr, Sturdivant, & Lemeshow, 2013). This makes it well-suited for our task of predicting flight delays.

•	Model Training

We train the Logistic Regression model on the flights_train dataset. The training code is shown in the following figure.

![1](https://github.com/user-attachments/assets/04cdd431-26b1-43fb-ac30-193bed3fb6ad)

•	Model Prediction

Once the model is trained, we use it to make predictions on the test dataset (flights_test). The predictions and their evaluation are shown by the following figure.

![2](https://github.com/user-attachments/assets/4ec7cb06-fe19-439a-9bdc-c6d02101f275)

•	Model Evaluation

The evaluation metrics for the Logistic Regression model are calculated using the precision and recall as shown in the following figure.

![3](https://github.com/user-attachments/assets/58b5cd14-ca26-4848-b213-57eb32e4fd82)

•	Calculating Confusion Matrix Elements

Following figure shows the calculation of confusion matrix.

![4](https://github.com/user-attachments/assets/78f9d55d-6397-4dad-a778-852740f6ecd9)

•	Calculating Model Accuracy

The precision and recall values are show on the following figure.

![5](https://github.com/user-attachments/assets/c0913e6c-e785-40eb-8828-5dbb381fd2a5)

4. Explorative Analysis and Visualizations
   
Throughout the above points, we targeted to under the data thoroughly by understating the nature of the dataset, setting up PySpark to handle the data processing, any corrections required, and utilization of machine learning models to predict flight delays using PySpark. While such points were discussed various findings was also listed now. Now, we aim to present such facts about the pre-processed data with the help of tableau through visualization.

The following figure, shows how the dataset is distributed amongst three categories which is delay, early, and on time. Further, it shows that a very unprecedented number of flights are delayed which is followed by flights that are early, and very few number of flights are on time. 

![6](https://github.com/user-attachments/assets/c4aba4b5-eaaa-4287-b3f1-958abbc27d60)

Likewise, the following figure show quite the similar concept graph where the delay of flights are shown per airport. Further, the bar graph is arranged in descending order. The visualities shows that O'Hare International Airport in Chicago, Illinois, USA (ORD) has most of delayed flights in comparison to others.

![7](https://github.com/user-attachments/assets/22b9dbeb-23a7-4e2b-b385-2b4d22965261)

Likewise, the following figure shows the average delay of flights by carrier. Carrier are the airline carrier code (e.g., OO for SkyWest Airlines, B6 for JetBlue Airways). The visualization of this particular information has shown that American Airlines (AA) has the highest average of delay flights. 

![8](https://github.com/user-attachments/assets/23314a8c-3406-4d42-af51-260ac42c5ad4)

The following figure shows the total delay of flights per months. This visualization has been done by a pie chart. Each of the slice represents a month. The details of each of slice is also provide in the legend section on the right-hand side. As shown in the figure, May and November are the months where the flights are mostly delayed. 

![9](https://github.com/user-attachments/assets/ffcea2eb-618c-4b96-a3cb-a3af10c94e2b)

The following figure, shows the average delay of various airplanes per months, where each of the colored lines represents various airplanes. Likewise, each of the points of the lines also show the value of each month.   

![10](https://github.com/user-attachments/assets/774243c9-7df0-462d-ae4a-cb83250de140)

The following figure shows an area chart for the comparison of flight duration vs average delay of flights. Looking at the figure there is no conclusive evidence to say that with the increase in flight duration there is any chance in the delay of flights, never the less there are times when the flight is extremely delays or is early.  

![11](https://github.com/user-attachments/assets/c7299359-2607-4274-bd12-428db5b5bfda)
