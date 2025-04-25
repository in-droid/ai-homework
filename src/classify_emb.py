from typing import Union, Callable
import numpy as np
from .model import Model
"""
Inference class for embedding-based classification.
Only as a proof of concept.
Compatible with the GeminiLLM class.
"""

class EmbeddingClassifier(Model):
    def __init__(self, encoder, classifier, class_map: list, agg="cat"):
        super().__init__()
        self.encoder = encoder
    
        self.classifier = classifier
        self.class_map = class_map
        self.agg = agg
    

    
    def encode(self, text: str) -> np.ndarray:
        return self.encoder.encode(text)
    
    
    def categorize(self, company_name: str, description:str = '') -> str:
        embedding_name = self.encode(company_name)
        embedding_description = self.encode(description)
        if self.agg == "cat":
            embedding = np.concatenate((embedding_name, embedding_description))
        elif self.agg == "mean":
            embedding = (embedding_name + embedding_description) / 2
        else:
            raise ValueError("Invalid aggregation method. Use 'cat' or 'mean'.")
        class_id = self.classifier.predict(embedding.reshape(1, -1))[0]
        return self.class_map[class_id]
    

