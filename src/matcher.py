from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
import os

class Matcher:
    def __init__(self, database_path='data/templates.json'):
        self.database_path = database_path
        self.database = self.load_database()

    def load_database(self):
        if os.path.exists(self.database_path):
            with open(self.database_path, 'r') as f:
                return json.load(f)
        return {}

    def save_database(self):
        with open(self.database_path, 'w') as f:
            json.dump(self.database, f)

    def add_template(self, name, embedding):
        self.database[name] = embedding.tolist()
        self.save_database()

    def identify(self, query_embedding, threshold=0.6):
        """
        Identifies the person by comparing query embedding with stored templates.
        Returns (name, score) or (None, 0)
        """
        if not self.database:
            return None, 0

        names = list(self.database.keys())
        templates = np.array([self.database[n] for n in names])
        
        # Reshape query for cosine_similarity
        query = query_embedding.reshape(1, -1)
        sims = cosine_similarity(query, templates)[0]
        
        best_idx = np.argmax(sims)
        best_score = sims[best_idx]
        
        if best_score >= threshold:
            return names[best_idx], best_score
        
        return "Unknown", best_score
