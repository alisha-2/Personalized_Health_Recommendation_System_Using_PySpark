import findspark
findspark.init()
import pyspark
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
import pandas as pd
import os
from datetime import datetime

# Simulate fetching updated data (placeholder for real API call)
def fetch_updated_data():
    """
    Simulate fetching updated symptom-diagnosis data.
    In a real scenario, this would involve an API call to a health database.
    Returns a DataFrame with new symptom-diagnosis mappings.
    """
    # Mock new data (in a real scenario, this would come from an API)
    new_data = [
        (1001, 2001, 0.9),  # Example: new symptom-diagnosis mapping
        (1002, 2002, 0.8),
    ]
    new_df = pd.DataFrame(new_data, columns=["syd", "did", "wei"])
    
    # Simulate saving the updated data to datasets/
    new_df.to_csv("datasets/sym_dia_diff_updated.csv", index=False)
    return new_df

# Update the main dataset with new data
def update_dataset():
    """
    Update the main dataset with new symptom-diagnosis mappings.
    """
    # Load existing dataset
    existing_df = pd.read_csv("datasets/sym_dia_diff.csv")
    
    # Fetch new data
    new_df = fetch_updated_data()
    
    # Append new data to existing dataset (avoid duplicates)
    updated_df = pd.concat([existing_df, new_df]).drop_duplicates(subset=["syd", "did"])
    updated_df.to_csv("datasets/sym_dia_diff.csv", index=False)
    
    # Record the update timestamp
    with open("datasets/last_updated.txt", "w") as f:
        f.write(str(datetime.now()))
    return updated_df

# Main training script
spark = SparkSession.builder.appName("HealthRecommendation").getOrCreate()

# Update dataset before training
update_dataset()

df = spark.read.option("header", "true").csv("datasets/sym_dia_diff.csv", inferSchema=True)
df = df.na.drop(subset=["syd", "did", "wei"])

(training, test) = df.randomSplit([0.8, 0.2], seed=24)

als = ALS(maxIter=10, regParam=0.01, userCol="syd", itemCol="did", ratingCol="wei", coldStartStrategy="drop", nonnegative=True)
model = als.fit(training)

predictions = model.transform(test)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="wei", predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print(f"Root Mean Square Error (RMSE) on test data: {rmse}")

model.write().overwrite().save("models/als_model")

spark.stop()