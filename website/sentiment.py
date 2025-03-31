from flask import Flask, request, render_template
import requests
from bs4 import BeautifulSoup
import os
import json
import findspark
findspark.init()

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_replace, lower, udf
from pyspark.sql.types import IntegerType, FloatType, ArrayType, StringType
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, CountVectorizer
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.ml.feature import VectorAssembler

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.utils.data import DataLoader, TensorDataset

# Initialize Flask app
app = Flask(__name__)

# Initialize Spark session
spark = SparkSession.builder \
    .appName("DarkPatternsDetection") \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.memory", "8g") \
    .getOrCreate()

# Define keyword dictionaries for dark pattern types
urgency_words = ["limited", "hurry", "soon", "now", "quick", "fast", "expire", "deadline", 
                "flash", "urgent", "today", "last", "ending", "final", "exclusive", "now or never"]

scarcity_words = ["only", "left", "few", "limited", "exclusive", "rare", "stock", "running out", 
                 "almost gone", "remaining", "last chance", "shortage", "sell out", "popular"]

social_proof_words = ["popular", "best seller", "trending", "others", "customers", "reviews", 
                       "rating", "people", "join", "everyone", "trending", "favorite", "recommended"]

# Create functions for pattern detection
def count_matches(text, word_list):
    if not text:
        return 0
    text_lower = text.lower()
    count = 0
    for word in word_list:
        if word.lower() in text_lower:
            count += 1
    return count

# Register UDFs
urgency_udf = udf(lambda text: count_matches(text, urgency_words), IntegerType())
scarcity_udf = udf(lambda text: count_matches(text, scarcity_words), IntegerType())
social_proof_udf = udf(lambda text: count_matches(text, social_proof_words), IntegerType())
text_length_udf = udf(lambda text: len(text.split()) if text else 0, IntegerType())
caps_ratio_udf = udf(lambda t: sum(1 for c in t if c.isupper()) / max(len(t), 1) if t else 0.0, FloatType())
words_udf = udf(lambda text: text.lower().split() if text else [], ArrayType(StringType()))

# Load and preprocess data
df = spark.read.csv("Data/websitedata.tsv", header=True, inferSchema=True, sep="\t")

# Clean and preprocess
df = df.drop("page_id")
df = df.na.drop(subset=["text", "label"])
df = df.withColumnRenamed("label", "label_index")
df = df.withColumn("label_index", col("label_index").cast("integer"))
df = df.na.drop(subset=["label_index"])

# Add custom features
df = df.withColumn("text_cleaned", regexp_replace(lower(col("text")), "[^a-zA-Z0-9\\s]", " "))
df = df.withColumn("urgency_score", urgency_udf(col("text")))
df = df.withColumn("scarcity_score", scarcity_udf(col("text")))
df = df.withColumn("social_proof_score", social_proof_udf(col("text")))
df = df.withColumn("text_length", text_length_udf(col("text")))
df = df.withColumn("caps_ratio", caps_ratio_udf(col("text")))

# Instead of using the ML Pipeline's tokenizer which causes conflicts,
# precompute the tokens directly as a feature
df = df.withColumn("words_array", words_udf(col("text_cleaned")))

# Cache the dataframe
df = df.repartition(10)
df.cache()

# Create TF-IDF features from precomputed words
cv = CountVectorizer(inputCol="words_array", outputCol="word_counts", vocabSize=1000, minDF=2.0)
cv_model = cv.fit(df)
df_counts = cv_model.transform(df)

idf = IDF(inputCol="word_counts", outputCol="tfidf_features")
idf_model = idf.fit(df_counts)
df_tfidf = idf_model.transform(df_counts)

# Create feature vector
assembler = VectorAssembler(
    inputCols=["tfidf_features", "urgency_score", "scarcity_score", 
               "social_proof_score", "text_length", "caps_ratio"],
    outputCol="features"
)
df_assembled = assembler.transform(df_tfidf)

# Split data
train, test = df_assembled.randomSplit([0.8, 0.2], seed=42)

# Train LogisticRegression model
lr = LogisticRegression(featuresCol="features", labelCol="label_index", maxIter=20)
lr_model = lr.fit(train)

# Evaluate model
lr_predictions = lr_model.transform(test)
evaluator = MulticlassClassificationEvaluator(labelCol="label_index", metricName="accuracy")
accuracy = evaluator.evaluate(lr_predictions)
binary_evaluator = BinaryClassificationEvaluator(labelCol="label_index")
auroc = binary_evaluator.evaluate(lr_predictions)

# Try RandomForest as well
rf = RandomForestClassifier(featuresCol="features", labelCol="label_index", numTrees=20)
rf_model = rf.fit(train)

rf_predictions = rf_model.transform(test)
rf_accuracy = evaluator.evaluate(rf_predictions)
rf_auroc = binary_evaluator.evaluate(rf_predictions)

# Choose the best model
if rf_accuracy > accuracy:
    best_model = rf_model
else:
    best_model = lr_model

# Load pre-trained BERT model for sentiment analysis
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
bert_model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Function to predict if a webpage contains dark patterns
def predict_dark_patterns(webpage_text):
    input_data = spark.createDataFrame([(webpage_text,)], ["text"])
    input_data = input_data.withColumn("text_cleaned", regexp_replace(lower(col("text")), "[^a-zA-Z0-9\\s]", " "))
    input_data = input_data.withColumn("urgency_score", urgency_udf(col("text")))
    input_data = input_data.withColumn("scarcity_score", scarcity_udf(col("text")))
    input_data = input_data.withColumn("social_proof_score", social_proof_udf(col("text")))
    input_data = input_data.withColumn("text_length", text_length_udf(col("text")))
    input_data = input_data.withColumn("caps_ratio", caps_ratio_udf(col("text")))
    input_data = input_data.withColumn("words_array", words_udf(col("text_cleaned")))

    input_data_counts = cv_model.transform(input_data)
    input_data_tfidf = idf_model.transform(input_data_counts)
    input_data_features = assembler.transform(input_data_tfidf)

    prediction = best_model.transform(input_data_features)
    result = prediction.select("prediction").collect()[0][0]
    return bool(result)

# Function to predict sentiment using BERT model
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding=True)
    outputs = bert_model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)
    sentiment = "Positive" if predictions.item() == 1 else "Negative"
    return sentiment

# Function to scrape data from a single web page
def scrape_page(url):
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to retrieve data from {url}")
        return None
    
    soup = BeautifulSoup(response.content, 'html.parser')
    webpage_text = soup.get_text(separator=' ', strip=True)
    
    return webpage_text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    url = request.form['url']
    webpage_text = scrape_page(url)
    if webpage_text:
        contains_dark_patterns = predict_dark_patterns(webpage_text)
        sentiment = predict_sentiment(webpage_text)
        return render_template('index.html', url=url, prediction=contains_dark_patterns, sentiment=sentiment)
    else:
        return render_template('index.html', url=url, prediction="Failed to retrieve data", sentiment="")

if __name__ == "__main__":
    app.run(debug=True)