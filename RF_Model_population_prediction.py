import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# File locations
input_path = r"D:\\Alpha map tech\\Current Project\\epidermology\\Population predition\\population data"
output_path = r"D:\\Alpha map tech\\Current Project\\epidermology\\Population predition\\Output"

input_files = {
    "1991": os.path.join(input_path, "Maharashtra Population 1991.csv"),
    "2001": os.path.join(input_path, "Maharashtra Population 2001.csv"),
}
output_file = os.path.join(output_path, "Maharashtra Population 2011.csv")

# Load data
data_1991 = pd.read_csv(input_files["1991"])
data_2001 = pd.read_csv(input_files["2001"])

# Required columns for prediction
columns_required = [
    "State", "District", "Total population", "Total male population",
    "Total female population", "Total 0 to 6 year children",
    "Male 0 to 6 year children", "Female 0 to 6 year children"
]

def preprocess_data(data):
    """Preprocess the input data to retain required columns, aggregate at district level, and handle missing data."""
    data = data[columns_required]
    data = data.groupby(["State", "District"], as_index=False).sum()
    data.dropna(inplace=True)  # Drop rows with missing values
    return data

# Preprocess datasets
data_1991 = preprocess_data(data_1991)
data_2001 = preprocess_data(data_2001)

# Merge data on State and District for training
training_data = data_1991.merge(
    data_2001, on=["State", "District"], suffixes=("_1991", "_2001")
)

# Prepare features and target variables
X = training_data[[
    "Total population_1991", "Total male population_1991", "Total female population_1991",
    "Total 0 to 6 year children_1991", "Male 0 to 6 year children_1991", "Female 0 to 6 year children_1991",
    "Total population_2001", "Total male population_2001", "Total female population_2001",
    "Total 0 to 6 year children_2001", "Male 0 to 6 year children_2001", "Female 0 to 6 year children_2001"
]]

y = training_data[[
    "Total population_2001", "Total male population_2001", "Total female population_2001",
    "Total 0 to 6 year children_2001", "Male 0 to 6 year children_2001", "Female 0 to 6 year children_2001"
]].add_suffix("_target")

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Regressor
model = RandomForestRegressor(random_state=42, n_estimators=100)
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error on Test Set: {mae}")

# Predict for 2011 data
data_2011 = data_2001.copy()  # Using 2001 data as base for prediction
X_2011 = data_2011[[
    "Total population", "Total male population", "Total female population",
    "Total 0 to 6 year children", "Male 0 to 6 year children", "Female 0 to 6 year children"
]].add_suffix("_2001")

# Add dummy columns for 1991 data
for col in [
    "Total population", "Total male population", "Total female population",
    "Total 0 to 6 year children", "Male 0 to 6 year children", "Female 0 to 6 year children"
]:
    X_2011[f"{col}_1991"] = 0  # You can adjust default values as necessary

# Ensure column order matches training data
X_2011 = X_2011[[
    "Total population_1991", "Total male population_1991", "Total female population_1991",
    "Total 0 to 6 year children_1991", "Male 0 to 6 year children_1991", "Female 0 to 6 year children_1991",
    "Total population_2001", "Total male population_2001", "Total female population_2001",
    "Total 0 to 6 year children_2001", "Male 0 to 6 year children_2001", "Female 0 to 6 year children_2001"
]]

# Make predictions for 2011
data_2011_predictions = model.predict(X_2011)

data_2011["Total population"] = data_2011_predictions[:, 0]
data_2011["Total male population"] = data_2011_predictions[:, 1]
data_2011["Total female population"] = data_2011_predictions[:, 2]
data_2011["Total 0 to 6 year children"] = data_2011_predictions[:, 3]
data_2011["Male 0 to 6 year children"] = data_2011_predictions[:, 4]
data_2011["Female 0 to 6 year children"] = data_2011_predictions[:, 5]

# Retain required columns for output
output_data = data_2011[columns_required]

# Save predictions to output CSV
os.makedirs(output_path, exist_ok=True)
output_data.to_csv(output_file, index=False)
print(f"Predicted 2011 population data saved to {output_file}")

print("Starting the population prediction script...")

# After loading each dataset
print("1991 data loaded successfully.")
print("2001 data loaded successfully.")

# Before saving the output
print("Predictions completed. Saving to CSV...")
