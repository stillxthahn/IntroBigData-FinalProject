# Problem 1: Classification with Logistic Regression - Low-Level Operations Implementation
# This file implements logistic regression for fraud detection using only low-level Spark RDD operations.
# Each cell is separated by a line of hyphens, and a markdown comment explains the key concepts.

#--------------------------------------------------------------------#

"""
## Setup and Initialization
In this section, we import only Spark-related libraries and create a SparkSession.
Key points:
- Avoiding external libraries like NumPy or Pandas as per requirements
- Configuring adequate memory for Spark to handle the dataset
"""

# Import only Spark-related libraries
import pyspark
from pyspark.sql import SparkSession
import math
import time

# Create a SparkSession
spark = SparkSession.builder \
    .appName("CreditCardFraud_LowLevel") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

sc = spark.sparkContext
print(f"Spark version: {spark.version}")
print(f"PySpark version: {pyspark.__version__}")

#--------------------------------------------------------------------#

"""
## Data Loading
This section loads the Credit Card Fraud dataset using low-level RDD operations.
Key points:
- Using textFile to read the CSV directly into an RDD
- Filtering out the header row to work with only data rows
"""

# Path to the credit card dataset in Kaggle
credit_card_path = "/kaggle/input/creditcardfraud/creditcard.csv"

# Load the CSV file as a text file and filter out the header
lines_rdd = sc.textFile(credit_card_path)
header = lines_rdd.first()
data_rdd_raw = lines_rdd.filter(lambda line: line != header)

# Quick count of the raw data
print(f"Total number of records: {data_rdd_raw.count()}")

#--------------------------------------------------------------------#

"""
## Data Parsing and Initial Analysis
This section parses the raw text data into features and labels and performs initial analysis.
Key points:
- Converting text lines to tuples of (features, label)
- Analyzing class distribution to understand the imbalance
- The dataset is known to be highly imbalanced (very few frauds)
"""

# Parse each line into features and label
def parse_line(line):
    try:
        parts = [float(x.strip().replace('"', '')) for x in line.split(",")]
        features = parts[:-1]  # All columns except the last one
        label = parts[-1]      # Last column is the label (Class)
        return (features, label)
    except Exception as e:
        return None

parsed_data_rdd = data_rdd_raw.map(parse_line).filter(lambda x: x is not None).cache()
num_features = len(parsed_data_rdd.first()[0])
total_count = parsed_data_rdd.count()

print(f"Total valid records: {total_count}")
print(f"Number of features: {num_features}")

# Check class distribution (important for imbalanced dataset)
class_counts = parsed_data_rdd.map(lambda x: (x[1], 1)).reduceByKey(lambda a, b: a + b).collectAsMap()
print(f"Class distribution: {class_counts}")
print(f"Percentage of fraudulent transactions: {class_counts.get(1.0, 0)/total_count*100:.4f}%")

# Show a sample record
print("\nSample record (features, label):")
print(parsed_data_rdd.first())

#--------------------------------------------------------------------#

"""
## Feature Scaling
This section scales the features using min-max normalization to ensure faster convergence.
Key points:
- Using a custom RDD aggregation to find min and max for each feature
- Normalizing all features to [0,1] range
- Broadcasting min/max values to all worker nodes for efficiency
"""

# Collect min and max for each feature using mapPartitions for better performance
def compute_min_max(iterator):
    local_min_max = ([float('inf')] * num_features, [float('-inf')] * num_features)
    for record in iterator:
        features = record[0]
        for i in range(num_features):
            local_min_max[0][i] = min(local_min_max[0][i], features[i])
            local_min_max[1][i] = max(local_min_max[1][i], features[i])
    yield local_min_max

# Aggregate min/max across all partitions
min_max_stats = parsed_data_rdd.mapPartitions(compute_min_max).reduce(
    lambda x, y: (
        [min(x[0][i], y[0][i]) for i in range(num_features)],
        [max(x[1][i], y[1][i]) for i in range(num_features)]
    )
)

min_features = min_max_stats[0]
max_features = min_max_stats[1]

# Calculate feature ranges, ensuring no division by zero
epsilon = 1e-8  # Small value to prevent division by zero
feature_ranges = []
for i in range(num_features):
    range_val = max_features[i] - min_features[i]
    feature_ranges.append(max(range_val, epsilon))

# Broadcast min, max, and ranges to all worker nodes
bc_min_features = sc.broadcast(min_features)
bc_max_features = sc.broadcast(max_features)
bc_feature_ranges = sc.broadcast(feature_ranges)

# Scale the features using min-max normalization
def scale_record(record):
    features, label = record
    scaled_features = []
    for i in range(len(features)):
        scaled_val = (features[i] - bc_min_features.value[i]) / bc_feature_ranges.value[i]
        scaled_features.append(scaled_val)
    return (scaled_features, label)

scaled_data_rdd = parsed_data_rdd.map(scale_record).cache()

print("Sample Scaled Record (Features, Label):")
print(scaled_data_rdd.first())

# Unpersist the unscaled data
parsed_data_rdd.unpersist()

#--------------------------------------------------------------------#

"""
## Train-Test Split
This section randomly splits the data into training and testing sets.
Key points:
- Using an 80/20 split, which is standard in machine learning
- Setting a fixed random seed (42) for reproducibility of results
- Caching the training and test RDDs for multiple uses
"""

# Randomly split data into training (80%) and testing (20%) sets
train_rdd, test_rdd = scaled_data_rdd.randomSplit([0.8, 0.2], seed=42)
train_rdd.cache()
test_rdd.cache()

train_count = train_rdd.count()
test_count = test_rdd.count()
print(f"Training set size: {train_count}")
print(f"Test set size: {test_count}")

# Unpersist the full scaled data RDD
scaled_data_rdd.unpersist()

#--------------------------------------------------------------------#

"""
## Class Weighting for Imbalance Handling
This section calculates weights for each class to address the extreme imbalance.
Key points:
- The credit card fraud dataset has very few positive examples (~0.17%)
- We compute weights inversely proportional to class frequencies
- This ensures the minority class has more impact during training
"""

# Calculate class weights on the training data
train_class_counts = train_rdd.map(lambda x: (x[1], 1)).reduceByKey(lambda a, b: a + b).collectAsMap()
count_class_0 = train_class_counts.get(0.0, 0)
count_class_1 = train_class_counts.get(1.0, 0)

# Compute weights: larger weight for the minority class
total = count_class_0 + count_class_1
if count_class_0 == 0 or count_class_1 == 0:
    print("Warning: One class is missing in the training set. Using equal weights.")
    class_weights = {0.0: 1.0, 1.0: 1.0}
else:
    # Simple inverse frequency weighting
    class_weights = {
        0.0: total / (2.0 * count_class_0),
        1.0: total / (2.0 * count_class_1)
    }

print(f"Class distribution in training set - 0: {count_class_0}, 1: {count_class_1}")
print(f"Class weights - 0: {class_weights[0.0]:.4f}, 1: {class_weights[1.0]:.4f}")

# Broadcast class weights to all worker nodes
bc_class_weights = sc.broadcast(class_weights)

#--------------------------------------------------------------------#

"""
## Logistic Regression Helper Functions
This section defines the core mathematical functions needed for logistic regression.
Key points:
- Sigmoid function with overflow protection for numerical stability
- Dot product implementation for vector multiplication
- Prediction function to compute probabilities
- Gradient and loss computation with L2 regularization
"""

def sigmoid(z):
    """Sigmoid function with simple overflow protection"""
    if z < -500:
        return 0.0
    elif z > 500:
        return 1.0
    else:
        return 1.0 / (1.0 + math.exp(-z))

def dot_product(vec1, vec2):
    """Compute dot product of two vectors (lists)"""
    return sum(v1 * v2 for v1, v2 in zip(vec1, vec2))

def predict(features, weights):
    """Predict probability using dot product and sigmoid"""
    # Add a 1.0 for the bias term
    features_with_bias = [1.0] + features
    z = dot_product(features_with_bias, weights)
    return sigmoid(z)

def compute_gradient_and_loss(record, weights, reg_param):
    """Compute gradient and loss for a single record with regularization"""
    features, label = record
    features_with_bias = [1.0] + features  # Add bias term
    
    # Compute prediction
    prediction = predict(features, weights)
    
    # Get the weight for this class
    class_weight = bc_class_weights.value.get(label, 1.0)
    
    # Compute error
    error = prediction - label
    
    # Compute gradient for each feature
    gradient = [class_weight * error * x for x in features_with_bias]
    
    # Add L2 regularization to all weights except bias
    for i in range(1, len(weights)):
        gradient[i] += reg_param * weights[i]
    
    # Compute log loss with epsilon to avoid log(0)
    epsilon = 1e-9
    if label == 1.0:
        loss = -class_weight * math.log(prediction + epsilon)
    else:
        loss = -class_weight * math.log(1 - prediction + epsilon)
        
    # Add L2 regularization term to loss (exclude bias term)
    reg_loss = 0.0
    for i in range(1, len(weights)):
        reg_loss += 0.5 * reg_param * (weights[i] ** 2)
        
    return (gradient, loss + reg_loss)

#--------------------------------------------------------------------#

"""
## Gradient Descent Implementation
This section implements the gradient descent algorithm from scratch.
Key points:
- Initializing weights (including bias term) to zeros
- Using L2 regularization to prevent overfitting
- Implementing early stopping via convergence check
- Training over multiple iterations until convergence or max iterations
"""

# Initialize weights (including bias term)
num_features_with_bias = num_features + 1  # Add 1 for bias term
initial_weights = [0.0] * num_features_with_bias

# Gradient Descent Parameters
learning_rate = 0.1
max_iterations = 30
reg_param = 0.01  # L2 regularization strength
tolerance = 1e-4  # Convergence threshold

# Training loop
current_weights = initial_weights.copy()
previous_loss = float('inf')
iteration_stats = []

print(f"\nStarting Low-Level Logistic Regression Training...")
print(f"Parameters: Learning Rate={learning_rate}, L2 Reg={reg_param}, Max Iterations={max_iterations}")

for iteration in range(max_iterations):
    start_time = time.time()
    
    # Process each partition to compute local gradients and losses
    def process_partition(iterator):
        local_gradients = [0.0] * num_features_with_bias
        local_loss = 0.0
        local_count = 0
        
        for record in iterator:
            grad, loss = compute_gradient_and_loss(record, current_weights, reg_param)
            # Add to local gradient sum
            local_gradients = [local_gradients[i] + grad[i] for i in range(num_features_with_bias)]
            local_loss += loss
            local_count += 1
            
        yield (local_gradients, local_loss, local_count)
    
    # Map partitions and reduce results
    result = train_rdd.mapPartitions(process_partition).reduce(
        lambda x, y: (
            [x[0][i] + y[0][i] for i in range(num_features_with_bias)],
            x[1] + y[1],
            x[2] + y[2]
        )
    )
    
    total_gradients, total_loss, count = result
    
    # Average the gradients and loss
    avg_gradients = [g / count for g in total_gradients]
    avg_loss = total_loss / count
    
    # Update weights using gradient descent
    for i in range(num_features_with_bias):
        current_weights[i] -= learning_rate * avg_gradients[i]
    
    # Compute time taken
    elapsed_time = time.time() - start_time
    
    # Check for convergence
    loss_change = abs(avg_loss - previous_loss)
    iteration_stats.append((iteration+1, avg_loss, loss_change, elapsed_time))
    
    # Print progress periodically
    if (iteration+1) % 5 == 0 or iteration == 0 or iteration == max_iterations-1:
        print(f"Iteration {iteration+1}/{max_iterations} | Loss: {avg_loss:.6f} | Change: {loss_change:.6f} | Time: {elapsed_time:.2f}s")
    
    if loss_change < tolerance:
        print(f"\nConverged at iteration {iteration+1}. Loss change below tolerance ({tolerance}).")
        break
        
    previous_loss = avg_loss

print("\n--- Training Complete ---")
print(f"Final Weights (first 5): {current_weights[:5]}")

#--------------------------------------------------------------------#

"""
## Model Evaluation
This section evaluates the trained model on the test set.
Key points:
- Computing the confusion matrix (TP, FP, TN, FN)
- Calculating key metrics: accuracy, precision, recall, F1 score
- Evaluating at a specific probability threshold (default 0.5)
- For fraud detection, recall is often more important than precision
"""

def evaluate_model(test_data, weights, threshold=0.5):
    """Evaluate the model on test data"""
    # Function to predict and compare with actual label
    def predict_and_evaluate(record):
        features, actual_label = record
        prob = predict(features, weights)
        predicted_label = 1.0 if prob >= threshold else 0.0
        return (predicted_label, actual_label, prob)
    
    # Get predictions for all test records
    predictions = test_data.map(predict_and_evaluate)
    predictions.cache()
    
    # Calculate confusion matrix counts
    tp = predictions.filter(lambda x: x[0] == 1.0 and x[1] == 1.0).count()
    fp = predictions.filter(lambda x: x[0] == 1.0 and x[1] == 0.0).count()
    tn = predictions.filter(lambda x: x[0] == 0.0 and x[1] == 0.0).count()
    fn = predictions.filter(lambda x: x[0] == 0.0 and x[1] == 1.0).count()
    
    # Calculate metrics
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Unpersist predictions RDD
    predictions.unpersist()
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn}
    }

# Evaluate model with default threshold of 0.5
print("\n--- Evaluating Model on Test Set ---")
evaluation = evaluate_model(test_rdd, current_weights, threshold=0.5)

print("Results with threshold=0.5:")
print(f"Accuracy:  {evaluation['accuracy']:.4f}")
print(f"Precision: {evaluation['precision']:.4f}")
print(f"Recall:    {evaluation['recall']:.4f} (Sensitivity)")
print(f"F1-Score:  {evaluation['f1_score']:.4f}")
print("Confusion Matrix:")
print(f"  True Positives:  {evaluation['confusion_matrix']['tp']}")
print(f"  False Positives: {evaluation['confusion_matrix']['fp']}")
print(f"  True Negatives:  {evaluation['confusion_matrix']['tn']}")
print(f"  False Negatives: {evaluation['confusion_matrix']['fn']}")

#--------------------------------------------------------------------#

"""
## Threshold Tuning
This section explores different decision thresholds to optimize performance.
Key points:
- The default 0.5 threshold may not be optimal for imbalanced data
- Adjusting the threshold controls the precision-recall trade-off
- Higher thresholds typically increase precision but decrease recall
- The optimal threshold depends on the specific business cost of FP vs. FN
"""

# Try different thresholds to find better precision-recall balance
print("\n--- Evaluating Different Thresholds ---")
thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
for threshold in thresholds:
    eval_result = evaluate_model(test_rdd, current_weights, threshold)
    print(f"Threshold={threshold:.1f} | Precision: {eval_result['precision']:.4f} | Recall: {eval_result['recall']:.4f} | F1: {eval_result['f1_score']:.4f}")

#--------------------------------------------------------------------#

"""
## AUC Calculation
This section calculates the Area Under the ROC Curve (AUC) metric.
Key points:
- AUC measures the model's ability to distinguish between classes
- An AUC of 0.5 means random guessing, 1.0 is perfect classification
- For imbalanced datasets, AUC is more informative than accuracy
- We approximate AUC using the trapezoidal rule for efficiency
"""

def calculate_auc_approximation(test_data, weights):
    """Calculate an approximation of the AUC using discrete thresholds"""
    # Function to get prediction probabilities and actual labels
    def get_prediction_score(record):
        features, actual_label = record
        prob = predict(features, weights)
        return (prob, actual_label)
    
    # Get prediction scores and sort them
    pred_scores = test_data.map(get_prediction_score).collect()
    pred_scores.sort(key=lambda x: x[0], reverse=True)
    
    # Initialize counters
    num_positive = sum(1 for _, label in pred_scores if label == 1.0)
    num_negative = len(pred_scores) - num_positive
    
    if num_positive == 0 or num_negative == 0:
        print("Warning: Only one class present in test set. AUC calculation not possible.")
        return 0.0
    
    # Initialize the area
    auc = 0.0
    tp = 0
    fp = 0
    prev_fp = 0
    prev_tp = 0
    
    # Process each prediction
    for prob, label in pred_scores:
        if label == 1.0:
            tp += 1
        else:
            fp += 1
            
        # Add trapezoid area under the curve
        if fp > prev_fp:
            auc += (tp + prev_tp) * (fp - prev_fp) / (2.0 * num_positive * num_negative)
            prev_fp = fp
            prev_tp = tp
    
    return auc

print("\n--- AUC Approximation ---")
auc = calculate_auc_approximation(test_rdd, current_weights)
print(f"Area Under ROC Curve (AUC): {auc:.4f}")

#--------------------------------------------------------------------#

"""
## Cleanup and Shutdown
This section cleans up resources and stops the Spark session.
Key points:
- Unpersisting cached RDDs to free memory
- Unpersisting broadcast variables
- Properly stopping the Spark session
"""

# Clean up
print("\n--- Cleaning Up ---")
train_rdd.unpersist()
test_rdd.unpersist()
bc_min_features.unpersist()
bc_max_features.unpersist()
bc_feature_ranges.unpersist()
bc_class_weights.unpersist()

# Stop Spark session
print("--- Stopping Spark Session ---")
spark.stop()