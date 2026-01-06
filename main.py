import json
import numpy as np
import torch
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer, AutoModel


class VectorDatabase:
    def __init__(self):
        self.vectors = []
        self.texts = []

    def insert(self, vectors, texts):
        self.vectors.extend(vectors)
        self.texts.extend(texts)

    def cosine_similarity(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def search(self, query_vector, k=3):
        scores = []
        for i, vec in enumerate(self.vectors):
            score = self.cosine_similarity(query_vector, vec)
            scores.append((score, self.texts[i]))
        scores.sort(reverse=True, key=lambda x: x[0])
        return scores[:k]

class Embedder:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("TaylorAI/bge-micro")
        self.model = AutoModel.from_pretrained("TaylorAI/bge-micro")

    def encode(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1)
        return embedding[0].numpy()

class Generator:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
        self.model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")

    def generate(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id
)

        return self.tokenizer.decode(output[0], skip_special_tokens=True)

if __name__ == "__main__":
    db = VectorDatabase()
    embedder = Embedder()

    # Load blog.json
    with open("blog.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    # Extract text from JSON
    documents = [item["metadata"]["text"] for item in data]

    # Create embeddings
    doc_vectors = [embedder.encode(text) for text in documents]

    # Insert into vector database
    db.insert(doc_vectors, documents)

    # User query
    query = "What is retrieval augmented generation?"
    query_vector = embedder.encode(query)

    # Retrieve top 3 documents
    results = db.search(query_vector, k=3)

    generator = Generator()

    context = "\n".join([text for _, text in results])
    prompt = f"Context:\n{context}\n\nQuestion:\n{query}\n\nAnswer:"

    answer = generator.generate(prompt)

    print("\nFinal Answer:\n")
    print(answer)


