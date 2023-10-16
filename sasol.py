import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer

# Step 1: Load Data
train_data = pd.read_csv(r"C:\Users\27790\Downloads\Train.csv")
test_data = pd.read_csv(r"C:\Users\27790\Downloads\Test.csv")

# Print column names for verification
print("Training Data Columns:", train_data.columns)
print("Test Data Columns:", test_data.columns)

# Step 2: Data Preprocessing and Feature Engineering
# Handling Text Data
for col in train_data.columns:
    if train_data[col].dtype == 'object':
        train_data[col] = train_data[col].astype('category').cat.codes
        test_data[col] = test_data[col].astype('category').cat.codes

# Handle Missing Values
imputer = SimpleImputer(strategy='mean')
X = train_data.drop(columns=['Target'])  
y = train_data['Target']
X = imputer.fit_transform(X)

# Step 3: Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Model Training (Using Random Forest as an example)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 5: Model Evaluation
y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred)

# Print the classification report
print(report)

# Step 6: Make Predictions on Test Data
# Handle missing values in test data
test_data_processed = imputer.transform(test_data)

# Generate predictions
predictions = model.predict(test_data_processed)

# Step 7: Prepare Submission File
submission_df = pd.DataFrame({'ID': test_data['ID'], 'Target': predictions})

# Step 8: Save Submission File
submission_df.to_csv('sasol_submission.csv', index=False)
