from transformers import BertTokenizer, BertForSequenceClassification
import torch
import pandas as pd

class Model:
    def __init__(self, model_name, csv_file_path_1, csv_file_path_2):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name)
        self.data_1 = pd.read_csv(csv_file_path_1)
        self.data_2 = pd.read_csv(csv_file_path_2)
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
