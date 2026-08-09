import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import mlflow
import os

# Enable autologging for MLflow
mlflow.sklearn.autolog(log_input_examples=False)

# Load the training and testing data
X_train = pd.read_csv('Xtrain.csv')
X_test = pd.read_csv('Xtest.csv')
y_train = pd.read_csv('ytrain.csv').squeeze() # Use squeeze to convert DataFrame to Series
y_test = pd.read_csv('ytest.csv').squeeze()   # Use squeeze to convert DataFrame to Series

# Drop the 'Unnamed: 0' column if it exists, as it's usually an artifact of saving/loading CSVs
if 'Unnamed: 0' in X_train.columns:
    X_train = X_train.drop(columns=['Unnamed: 0'])
if 'Unnamed: 0' in X_test.columns:
    X_test = X_test.drop(columns=['Unnamed: 0'])

# Define numerical and categorical features based on the data description
numerical_features = [
    'Age', 'CityTier', 'DurationOfPitch', 'NumberOfPersonVisiting',
    'PreferredPropertyStar', 'NumberOfTrips', 'Passport', 'OwnCar',
    'NumberOfChildrenVisiting', 'MonthlyIncome', 'PitchSatisfactionScore',
    'NumberOfFollowups'
]
categorical_features = [
    'TypeofContact', 'Occupation', 'Gender', 'MaritalStatus', 'Designation',
    'ProductPitched'
]

# Create preprocessing pipelines for numerical and categorical features
numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough' # Keep other columns if any, though none are expected here
)

# Define the model pipeline with preprocessing and XGBoost Classifier
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False))
])

# Define the hyperparameter grid for GridSearchCV
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__learning_rate': [0.05, 0.1],
    'classifier__max_depth': [3, 5]
}

# Perform GridSearchCV for hyperparameter tuning
print("Starting GridSearchCV for hyperparameter tuning...")
with mlflow.start_run():
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    # Get the best model
    best_model = grid_search.best_estimator_

    print("GridSearchCV complete.")
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")

    # Evaluate the best model on the test set
    y_pred = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"Test accuracy of the best model: {test_accuracy:.4f}")
    print("Classification Report:")
    print(report)

    # Log additional metrics to MLflow (autologging handles most, but can add more explicitly)
    mlflow.log_metric("test_accuracy", test_accuracy)

    # Save the best model
    deployment_path = "tourism_project/deployment"
    os.makedirs(deployment_path, exist_ok=True)
    model_save_path = os.path.join(deployment_path, "model.joblib")
    joblib.dump(best_model, model_save_path)
    print(f"Best model saved to {model_save_path}")

    # Log the model as an MLflow artifact (autologging also does this, but explicit can be useful)
    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="xgboost_model",
        registered_model_name="XGBoostTourismPredictor"
    )

print("Model training and registration complete.")
