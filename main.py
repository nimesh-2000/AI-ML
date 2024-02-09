from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import mysql.connector
import pandas as pd

app = Flask(__name__)
CORS(app)

# MySQL database configuration
db_config = {
    'host': '127.0.0.1',
    'user': 'root',
    'password': '1234',
    'database': 'feedBackAnalysis'
}

def save_to_database(feedback_text, predicted_class, input_clusters):
    connection = mysql.connector.connect(**db_config)
    cursor = connection.cursor()

    # Set the type based on predicted_class value
    feedback_type = 'positive' if predicted_class >= 3 else 'negative'

    # Insert the data into the database
    query = "INSERT INTO feedbacksentimate (area, points, type) VALUES (%s, %s, %s)"
    values = (', '.join(input_clusters), predicted_class, feedback_type)

    cursor.execute(query, values)
    connection.commit()

    cursor.close()
    connection.close()

class Model:
    def __init__(self, model_name, csv_file_path_1, csv_file_path_2):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name)
        self.data_1 = pd.DataFrame()  # Initialize an empty DataFrame for data_1
        self.data_2 = pd.DataFrame()  # Initialize an empty DataFrame for data_2
        self.clusters = self.populate_clusters()

    def populate_clusters(self):
        clusters = {}
        for column in self.data_2.columns:
            cluster_name = column
            keywords = self.data_2[column].dropna().tolist()
            clusters[cluster_name] = keywords
        return clusters

    def identify_cluster(self, input_text):
        identified_clusters = []
        for cluster, keywords in self.clusters.items():
            if any(keyword in input_text.lower() for keyword in keywords):
                identified_clusters.append(cluster)
        return identified_clusters if identified_clusters else ['Uncategorized']

    def process_feedbacks(self, feedbacks):
        final_outputs = []
        for feedback in feedbacks:
            # Tokenize the input text and obtain the model's predictions
            tokens = self.tokenizer(feedback, return_tensors='pt')
            outputs = self.model(**tokens)

            # Get the predicted sentiment score
            sentiment_score = outputs.logits.softmax(dim=1)
            predicted_class = torch.argmax(sentiment_score, dim=1).item()

            # Identify the clusters for the input text
            input_clusters = self.identify_cluster(feedback)

            final_outputs.append({
                "feedback_text": feedback,
                "predicted_class": predicted_class,
                "input_clusters": input_clusters
            })

        return final_outputs

@app.route('/get_all_data', methods=['GET'])
def get_all_data():
    connection = mysql.connector.connect(**db_config)
    cursor = connection.cursor(dictionary=True)

    # Retrieve all data from the database
    query = "SELECT(SELECT COUNT(*) FROM feedBackSentimate WHERE type = 'positive') as positive_count," \
            "(SELECT COUNT(*) FROM feedBackSentimate WHERE type = 'negative') as negative_count;"
    cursor.execute(query)
    data = cursor.fetchall()

    cursor.close()
    connection.close()
    return jsonify(data)

@app.route('/get_chart_data', methods=['GET'])
def get_chart_data():
    try:
        connection = mysql.connector.connect(**db_config)
        cursor = connection.cursor(dictionary=True)

        # Retrieve data for the bar chart
        query = "SELECT area, SUM(CASE WHEN type='positive' THEN 1 ELSE 0 END) AS positive_count, " \
            "SUM(CASE WHEN type='negative' THEN 1 ELSE 0 END) AS negative_count " \
            "FROM feedbacksentimate GROUP BY area"
        cursor.execute(query)
        data = cursor.fetchall()

        cursor.close()
        connection.close()

        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/login', methods=['POST'])
def login():
    try:
        # Get login data from request
        login_data = request.json
        name = login_data.get('name')
        password = login_data.get('password')

        # Validate login data
        if not name or not password:
            return jsonify({'error': 'Username and password are required'}), 400

        # Check if user exists
        connection = mysql.connector.connect(**db_config)
        cursor = connection.cursor(dictionary=True)
        
        # Select user by username
        cursor.execute("SELECT * FROM user WHERE name = %s AND password = %s", (name, password))

        user = cursor.fetchone()
        cursor.close()

        if not user:
            return jsonify({'error': 'Invalid username or password'}), 401  # HTTP 401 Unauthorized status code

        # Handle successful login
        return jsonify({'message': 'Login successful'}), 200

    except Exception as e:
        return jsonify({'error': 'An error occurred'}), 500

@app.route('/get_pie_chart_data', methods=['GET'])
def get_pie_chart_data():
    try:
        connection = mysql.connector.connect(**db_config)
        cursor = connection.cursor(dictionary=True)

        # Retrieve data for the pie chart (positive counts only)
        query = "SELECT area, COUNT(*) AS positive_count FROM feedbacksentimate WHERE type='positive' GROUP BY area"
        cursor.execute(query)
        data = cursor.fetchall()

        cursor.close()
        connection.close()

        return jsonify(data)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    model_name = 'nlptown/bert-base-multilingual-uncased-sentiment'
    csv_file_path_1 = 'https://feedbacksentimate.s3.ap-south-1.amazonaws.com/uploads/British_Air_Customer_Reviews.csv'
    csv_file_path_2 = 'https://feedbacksentimate.s3.ap-south-1.amazonaws.com/uploads/clusters.csv'

    my_model = Model(model_name, csv_file_path_1, csv_file_path_2)

    # Load CSV files
    my_model.data_1 = pd.read_csv(csv_file_path_1)
    my_model.data_2 = pd.read_csv(csv_file_path_2)

    feedbacks = my_model.data_1['feedback'].tolist()[:20]

    final_outputs = my_model.process_feedbacks(feedbacks)

    for output in final_outputs:
        print("Feedback Text:", output["feedback_text"])
        print("Predicted Sentiment Class:", output["predicted_class"])
        print("Identified Clusters:", output["input_clusters"])
        print("\n" + "=" * 50 + "\n")

    # Start the Flask app
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=80)
