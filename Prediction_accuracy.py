import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import chardet

# Helper function to detect file encoding
def detect_encoding(file_path):
    with open(file_path, 'rb') as file:
        result = chardet.detect(file.read(10000))
        return result['encoding']

# File paths
predicted_file = r"D:\\Alpha map tech\\Current Project\\epidermology\\Population predition\\tree depth random forest\\Output\\TDRF_Population_2011.csv"
original_file = r"D:\\Alpha map tech\\Current Project\\epidermology\\Population predition\\tree depth random forest\\Testing\\Maharashtra Population 2011_original.csv"

# Detect encodings for both files
predicted_encoding = detect_encoding(predicted_file)
original_encoding = detect_encoding(original_file)

# Load data with detected encodings
try:
    predicted_data = pd.read_csv(predicted_file, encoding=predicted_encoding)
except UnicodeDecodeError:
    predicted_data = pd.read_csv(predicted_file, encoding='latin1')

try:
    original_data = pd.read_csv(original_file, encoding=original_encoding)
except UnicodeDecodeError:
    original_data = pd.read_csv(original_file, encoding='latin1')

# Ensure matching columns for comparison
columns_to_compare = ["District", "Total population", "Total male population", 
                      "Total female population", "Total 0 to 6 year children", 
                      "Male 0 to 6 year children", "Female 0 to 6 year children"]

predicted_data = predicted_data[columns_to_compare]
original_data = original_data[columns_to_compare]

# Merge dataframes on 'District'
merged_data = pd.merge(predicted_data, original_data, on="District", suffixes=('_predicted', '_original'))

# Calculate metrics
metrics = {}
for column in columns_to_compare[1:]:  # Skip 'District'
    r2 = r2_score(merged_data[f"{column}_original"], merged_data[f"{column}_predicted"])
    mae = mean_absolute_error(merged_data[f"{column}_original"], merged_data[f"{column}_predicted"])
    mse = mean_squared_error(merged_data[f"{column}_original"], merged_data[f"{column}_predicted"])
    metrics[column] = {"R2 Score": r2, "MAE": mae, "MSE": mse}

# Save accuracy metrics as a table image and PDF
fig, ax = plt.subplots(figsize=(12, 7))
ax.axis('off')
ax.axis('tight')
data_for_table = [
    [key] + list(value.values()) for key, value in metrics.items()
]
columns = ["Attribute", "R2 Score", "Mean Absolute Error (MAE)", "Mean Squared Error (MSE)"]
table = ax.table(cellText=data_for_table, colLabels=columns, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.5, 1.5)

# Add title
plt.title("Tree Depth Random Forest Accuracy Assessment", fontsize=16)

# Save the table as an image
output_image_path = r"D:\\Alpha map tech\\Current Project\\epidermology\\Population predition\\tree depth random forest\\Output\\Tree_Depth_Random_Forest_Accuracy.png"
plt.savefig(output_image_path, bbox_inches='tight')

# Save the table as a PDF
output_pdf_path = r"D:\\Alpha map tech\\Current Project\\epidermology\\Population predition\\tree depth random forest\\Output\\Tree_Depth_Random_Forest_Accuracy.pdf"
with PdfPages(output_pdf_path) as pdf:
    pdf.savefig(fig, bbox_inches='tight')

plt.show()
print("Validation complete. Accuracy metrics saved as image and PDF.")