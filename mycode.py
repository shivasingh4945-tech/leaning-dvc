# model.py

# Dataset: (size, bedrooms, price)
data = [
    (800, 1, 3000000),
    (1000, 2, 4200000),
    (1200, 2, 5000000),
    (1500, 3, 6500000),
    (1800, 3, 8000000)
]

# Initialize weights and bias
w1 = 0.0  # weight for size
w2 = 0.0  # weight for bedrooms
b = 0.0   # bias
lr = 0.000000001  # learning rate

# Train using gradient descent
for epoch in range(1000):
    for size, bed, price in data:
        y_pred = w1 * size + w2 * bed + b
        error = y_pred - price

        # Update parameters
        w1 -= lr * error * size
        w2 -= lr * error * bed
        b  -= lr * error

# Test prediction
test_size = 1400
test_bed = 3
prediction = w1 * test_size + w2 * test_bed + b

print("Predicted price:", int(prediction))
