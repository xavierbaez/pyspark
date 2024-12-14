# @author Xavier Baez <xavierbaez@gmail.com>
# PySpark script to ensure data consistency across two environments using AI tools

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
import tensorflow as tf

# Initialize PySpark session
spark = SparkSession.builder \
    .appName("Data Synchronization with AI") \
    .getOrCreate()

# Define file paths for both environments
environment_1_path = "environment1/data.csv"
environment_2_path = "environment2/data.csv"

# Load data from both environments
data_env1 = spark.read.csv(environment_1_path, header=True, inferSchema=True)
data_env2 = spark.read.csv(environment_2_path, header=True, inferSchema=True)

# Check schema consistency
if data_env1.schema != data_env2.schema:
    raise ValueError("Schemas do not match across environments!")

# Combine data for analysis and training
data_combined = data_env1.union(data_env2)

# Data preprocessing (Example: Selecting specific columns and assembling features)
assembler = VectorAssembler(inputCols=["feature1", "feature2", "feature3"], outputCol="features")
data_prepared = assembler.transform(data_combined).select("features", "label")

# Train a simple regression model using PySpark MLlib
lr = LinearRegression(featuresCol="features", labelCol="label")
model = lr.fit(data_prepared)

# Save the trained model for further use
model.save("path/to/save/spark_model")

# Example: Use TensorFlow for advanced AI processing
# Convert PySpark DataFrame to Pandas for TensorFlow processing
data_pandas = data_combined.toPandas()

# Define a simple TensorFlow model
model_tf = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(len(data_pandas.columns) - 1,)),
    tf.keras.layers.Dense(1)
])

model_tf.compile(optimizer='adam', loss='mse')

# Extract features and labels for TensorFlow
features_tf = data_pandas.drop("label", axis=1).values
labels_tf = data_pandas["label"].values

# Train the TensorFlow model
model_tf.fit(features_tf, labels_tf, epochs=10, batch_size=32)

# Save the TensorFlow model
model_tf.save("path/to/save/tf_model")

# Verify data consistency across environments
count_env1 = data_env1.count()
count_env2 = data_env2.count()

if count_env1 != count_env2:
    print("Warning: Row counts differ between environments.")
else:
    print("Row counts are consistent between environments.")

# Stop the Spark session
spark.stop()
