from flask import Flask, request, render_template
import requests
from bs4 import BeautifulSoup
import findspark
findspark.init()

# from pyspark.sql import SparkSession
# from pyspark.sql.functions import col, regexp_replace, lower, udf
# from pyspark.sql.types import IntegerType, FloatType, ArrayType, StringType

# Initialize Flask app
app = Flask(__name__)

# # Initialize Spark session
# spark = SparkSession.builder \
#     .appName("DarkPatternsDetection") \
#     .config("spark.driver.memory", "4g") \
#     .config("spark.executor.memory", "4g") \
#     .config("spark.sql.warehouse.dir", "spark-warehouse") \
#     .config("spark.hadoop.hadoop.native.lib", "false") \
#     .getOrCreate()

# Define keyword dictionaries for dark pattern types
# urgency_words = ["limited", "hurry", "soon", "now", "quick", "fast", "expire", "deadline", 
#                 "flash", "urgent", "today", "last", "ending", "final", "exclusive", "now or never"]

# scarcity_words = ["only", "left", "few", "limited", "exclusive", "rare", "stock", "running out", 
#                  "almost gone", "remaining", "last chance", "shortage", "sell out", "popular"]

# social_proof_words = ["popular", "best seller", "trending", "others", "customers", "reviews", 
#                        "rating", "people", "join", "everyone", "trending", "favorite", "recommended"]

# # Create functions for pattern detection
# def count_matches(text, word_list):
#     if not text:
#         return 0
#     text_lower = text.lower()
#     count = 0
#     for word in word_list:
#         if word.lower() in text_lower:
#             count += 1
#     return count

# # Register UDFs
# urgency_udf = udf(lambda text: count_matches(text, urgency_words), IntegerType())
# scarcity_udf = udf(lambda text: count_matches(text, scarcity_words), IntegerType())
# social_proof_udf = udf(lambda text: count_matches(text, social_proof_words), IntegerType())
# text_length_udf = udf(lambda text: len(text.split()) if text else 0, IntegerType())
# caps_ratio_udf = udf(lambda t: sum(1 for c in t if c.isupper()) / max(len(t), 1) if t else 0.0, FloatType())
# words_udf = udf(lambda text: text.lower().split() if text else [], ArrayType(StringType()))

# # Load previously trained models and transformers
# cv_model = ... # Load your CountVectorizer model
# idf_model = ... # Load your IDF model
# assembler = ... # Load your VectorAssembler
# best_model = ... # Load your best model (Logistic Regression or Random Forest)

# # Function to predict if a webpage contains dark patterns
# def predict_dark_patterns(webpage_text):
#     input_data = spark.createDataFrame([(webpage_text,)], ["text"])
#     input_data = input_data.withColumn("text_cleaned", regexp_replace(lower(col("text")), "[^a-zA-Z0-9\\s]", " "))
#     input_data = input_data.withColumn("urgency_score", urgency_udf(col("text")))
#     input_data = input_data.withColumn("scarcity_score", scarcity_udf(col("text")))
#     input_data = input_data.withColumn("social_proof_score", social_proof_udf(col("text")))
#     input_data = input_data.withColumn("text_length", text_length_udf(col("text")))
#     input_data = input_data.withColumn("caps_ratio", caps_ratio_udf(col("text")))
#     input_data = input_data.withColumn("words_array", words_udf(col("text_cleaned")))

#     input_data_counts = cv_model.transform(input_data)
#     input_data_tfidf = idf_model.transform(input_data_counts)
#     input_data_features = assembler.transform(input_data_tfidf)

#     prediction = best_model.transform(input_data_features)
#     result = prediction.select("prediction").collect()[0][0]
#     return bool(result)

# # Function to scrape data from a single web page
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
    print(webpage_text)
    return render_template('index.html', url=url, prediction=webpage_text)
    # if webpage_text:
    #     contains_dark_patterns = predict_dark_patterns(webpage_text)
    #     return render_template('index.html', url=url, prediction=contains_dark_patterns)
    # else:
    #     return render_template('index.html', url=url, prediction="Failed to retrieve data")

if __name__ == "__main__":
    app.run(debug=True)