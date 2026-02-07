import pandas as pd
import os

# Dataset: (size, bedrooms, price)
data = [
    (800, 1, 3000000),
    (1000, 2, 4200000),
    (1200, 2, 5000000),
    (1500, 3, 6500000),
    (1800, 3, 8000000)
]

# Initialize weights and bias
w1 = 0.0
w2 = 0.0
b = 0.0
lr = 0.000000001

# Train using gradient descent
for epoch in range(1000):
    for size, bed, price in data:
        y_pred = w1 * size + w2 * bed + b
        error = y_pred - price

        w1 -= lr * error * size
        w2 -= lr * error * bed
        b  -= lr * error

# Test prediction
test_size = 1400
test_bed = 3
prediction = w1 * test_size + w2 * test_bed + b

print("Predicted price:", int(prediction))

# ---------- SAVE DATA ----------
data_dir = "data"
os.makedirs(data_dir, exist_ok=True)

# DEFINE df (this was missing)
df = pd.DataFrame(
    data,
    columns=["size_sqft", "bedrooms", "price"]
)

file_path = os.path.join(data_dir, "sample_data.csv")
df.to_csv(file_path, index=False)

print(f"CSV file is saved to {file_path}")
