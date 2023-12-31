Customer Inactivity Forecasting
This project focuses on developing a machine learning model capable of forecasting the probability of each customer becoming inactive, refraining from making any transactions for a 90-day period. By identifying potentially inactive customers in advance, businesses can implement strategies to retain them.

Problem Statement
The objective is to predict which customers are likely to become inactive within the next 90 days. The F1 score will be used as the error metric for this competition. The F1 score combines both precision and recall, providing a performance metric ranging from 0 (total failure) to 1 (perfect score).

Key Metrics
F1 Score: 
2
×
Precision
×
Recall
/
(
Precision
+
Recall
)
2×Precision×Recall/(Precision+Recall)

Precision: 
)
TP/(TP+FP)

Recall / Sensitivity / True Positive Rate (TPR): 

TP/(TP+FN)

Where:

TP=True Positive
FP=False Positive
TN=True Negative
FN=False Negative
Project Structure
lua
Copy code
|-- data
|   |-- train.csv                # Training data
|   |-- test.csv                 # Test data
|   |-- VariableDescription.csv  # Description of variables
|   |-- SampleSubmission.csv     # Sample submission format
|
|-- src
|   |-- main.py                   # Main script for data processing and modeling
|   |-- preprocessing.py          # Data preprocessing functions
|   |-- modeling.py               # Machine learning model functions
|   |-- evaluation.py             # Evaluation functions
|
|-- requirements.txt             # Python dependencies
|-- README.md                    # Project overview and instructions
Usage
Clone the Repository
bash
Copy code
git clone https://github.com/MngomaZAR/sasol_submission.git
Install Dependencies
bash
Copy code
pip install -r requirements.txt
Run the Model
bash
Copy code
python src/main.py
Contributing
If you'd like to contribute to this project, please follow these steps:

Fork the repository
Create a new branch (git checkout -b feature/improvement)
Make your changes and commit them (git commit -m 'Add new feature')
Push to the branch (git push origin feature/improvement)
Create a pull request
License
This project is licensed under the MIT License.﻿# sasol_submission
