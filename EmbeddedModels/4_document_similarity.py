from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embedding = OpenAIEmbeddings(model = "text-embedding-3-large", dimensions = 300)

documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting style and leadership skills.",
    "MS Dhoni is a former Indian cricketer and captain of the Indian national team, renowned for his calm demeanor and finishing abilities.",
    "Sachin Tendulkar is a legendary Indian cricketer, often referred to as the 'God of Cricket', with numerous records to his name.",
    "Rohit Sharma is an Indian cricketer known for his elegant batting and ability to score big centuries.",
    "Jasprit Bumrah is an Indian cricketer known for his unique bowling action and ability to bowl yorkers consistently."
]

query = "Tell me about Virat Kohli"

doc_embeddings = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
print("Similarity Scores:")
for i, score in enumerate(similarities):
    print(f"Document: {documents[i]}\nSimilarity Score: {score:.4f}\n")

index, score = sorted(list(enumerate(similarities)), key=lambda x: x[1], reverse=True)[0]

print(f"Most similar document: {documents[index]}\nSimilarity Score: {score:.4f}")

