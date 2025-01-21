import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import chardet

# Helper function to detect file encoding
def detect_encoding(file_path):
    with open(file_path, 'rb') as file:
        result = chardet.detect(file.read(10000))
        return result['encoding']

# File paths
predicted_file = r"D:\Alpha map tech\Current Project\epidermology\Population predition\Random Forest\Output\Maharashtra Population 2011.csv"
original_file = r"D:\Alpha map tech\Current Project\epidermology\Population predition\Random Forest\Testing\Maharashtra Population 2011_original.csv"

# Debug: print file paths and check if files exist
print(f"Predicted File Path: {predicted_file}")
print(f"Original File Path: {original_file}")

# Detect encodings for both files
predicted_encoding = detect_encoding(predicted_file)
original_encoding = detect_encoding(original_file)

print(f"Predicted File Encoding: {predicted_encoding}")
print(f"Original File Encoding: {original_encoding}")

# Load data with detected encodings
try:
    predicted_data = pd.read_csv(predicted_file, encoding=predicted_encoding)
    print("Predicted Data Loaded Successfully")
except UnicodeDecodeError:
    print("Error reading predicted data with detected encoding. Trying 'latin1'.")
    predicted_data = pd.read_csv(predicted_file, encoding='latin1')

try:
    original_data = pd.read_csv(original_file, encoding=original_encoding)
    print("Original Data Loaded Successfully")
except UnicodeDecodeError:
    print("Error reading original data with detected encoding. Trying 'latin1'.")
    original_data = pd.read_csv(original_file, encoding='latin1')

# Debug: Check the first few rows of both datasets
print("Predicted Data Head:")
print(predicted_data.head())

print("Original Data Head:")
print(original_data.head())

# Ensure matching columns for comparison
columns_to_compare = ["District", "Total population", "Total male population", 
                      "Total female population", "Total 0 to 6 year children", 
                      "Male 0 to 6 year children", "Female 0 to 6 year children"]

# Check if the required columns are in both datasets
missing_columns_predicted = [col for col in columns_to_compare if col not in predicted_data.columns]
missing_columns_original = [col for col in columns_to_compare if col not in original_data.columns]

if missing_columns_predicted:
    print(f"Missing columns in predicted data: {missing_columns_predicted}")
if missing_columns_original:
    print(f"Missing columns in original data: {missing_columns_original}")

# Filter the columns to ensure consistency
predicted_data = predicted_data[columns_to_compare]
original_data = original_data[columns_to_compare]

# Debug: Check the data after filtering columns
print("Filtered Predicted Data Head:")
print(predicted_data.head())

print("Filtered Original Data Head:")
print(original_data.head())

# Merge dataframes on 'District'
merged_data = pd.merge(predicted_data, original_data, on="District", suffixes=('_predicted', '_original'))

# Debug: Check the first few rows of the merged data
print("Merged Data Head:")
print(merged_data.head())

# Calculate metrics
metrics = {}
for column in columns_to_compare[1:]:  # Skip 'District'
    # Calculate R², MAE, MSE, RMSE, Adjusted R², and MAPE
    r2 = r2_score(merged_data[f"{column}_original"], merged_data[f"{column}_predicted"])
    mae = mean_absolute_error(merged_data[f"{column}_original"], merged_data[f"{column}_predicted"])
    mse = mean_squared_error(merged_data[f"{column}_original"], merged_data[f"{column}_predicted"])
    rmse = np.sqrt(mse)
    
    # Adjusted R²: 1 - (1-R²) * (n-1)/(n-p-1) where n is the number of data points and p is the number of predictors
    n = len(merged_data)
    p = 1  # One predictor in this case (the population values)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    
    # MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((merged_data[f"{column}_original"] - merged_data[f"{column}_predicted"]) / merged_data[f"{column}_original"])) * 100
    
    metrics[column] = {
        "R2 Score": r2, 
        "MAE": mae, 
        "MSE": mse,
        "RMSE": rmse, 
        "Adjusted R²": adj_r2, 
        "MAPE": mape
    }

# Debug: Print the calculated metrics
print("Metrics:")
print(metrics)

# Save final indepth accuracy metrics as a table image and PDF
fig, ax = plt.subplots(figsize=(12, 6))
ax.axis('off')
ax.axis('tight')
data_for_table = [
    [key] + list(value.values()) for key, value in metrics.items()
]
columns = ["Attribute", "R² Score", "Mean Absolute Error (MAE)", "Mean Squared Error (MSE)", "RMSE", "Adjusted R²", "MAPE"]
table = ax.table(cellText=data_for_table, colLabels=columns, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)

# Save the table as an image
output_image_path = r"D:\Alpha map tech\Current Project\epidermology\Population predition\Random Forest\Output\final_indepth_accuracy.png"
plt.savefig(output_image_path)

# Save the table as a PDF
output_pdf_path = r"D:\Alpha map tech\Current Project\epidermology\Population predition\Random Forest\Output\final_indepth_accuracy.pdf"
with PdfPages(output_pdf_path) as pdf:
    pdf.savefig(fig)

plt.show()

# Debug: Print output file paths
print(f"Image saved at: {output_image_path}")
print(f"PDF saved at: {output_pdf_path}")

# Final confirmation message
print("Validation complete. Final indepth accuracy metrics saved as image and PDF.")
