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

# Set environment variable for Hadoop (but we won't use it for saving)
os.environ['HADOOP_HOME'] = 'C:\\Program Files\\hadoop'


# Configure Spark with proper settings for Windows
spark = SparkSession.builder \
    .appName("DarkPatternsDetection") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .config("spark.sql.warehouse.dir", "spark-warehouse") \
    .config("spark.hadoop.hadoop.native.lib", "false") \
    .getOrCreate()
# Configure Spark with proper settings for Windows

hadoop_conf = spark._jsc.hadoopConfiguration()
hadoop_conf.set("fs.file.impl", "org.apache.hadoop.fs.LocalFileSystem")
hadoop_conf.set("mapreduce.fileoutputcommitter.marksuccessfuljobs", "false")
# Define keyword dictionaries for dark pattern types
urgency_words = ["limited", "hurry", "soon", "now", "quick", "fast", "expire", "deadline", 
                "flash", "urgent", "today", "last", "ending", "final", "exclusive", "now or never"]

scarcity_words = ["only", "left", "few", "limited", "exclusive", "rare", "stock", "running out", 
                 "almost gone", "remaining", "last chance", "shortage", "sell out", "popular"]

social_proof_words = ["popular", "best seller", "trending", "others", "customers", "reviews", 
                       "rating", "people", "join", "everyone", "trending", "favorite", "recommended","fake"]

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
print("Preprocessing done")
# Cache the dataframe
df = df.repartition(10)
df.cache()