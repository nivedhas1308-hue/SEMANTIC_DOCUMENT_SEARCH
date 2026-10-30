
import os
import pandas as pd
import chromadb

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# STEP 1: Load embedding model
embedding_function = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# STEP 2: Connect to ChromaDB
client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(
    name="semantic_documents"
)

# STEP 3: Load documents
documents = []
ids = []

folder = "./dataset/documents"

for file in os.listdir(folder):
    if file.endswith(".txt"):
        with open(os.path.join(folder, file), "r", encoding="utf-8") as f:
            documents.append(f.read())
            ids.append(file)

# STEP 4: Add documents to ChromaDB
for doc_id, content in zip(ids, documents):
    collection.upsert(
        ids=[doc_id],
        documents=[content]
    )

print("Documents added successfully!")
print("Total documents:", collection.count())

# STEP 5: Connect LangChain with ChromaDB
vectorstore = Chroma(
    collection_name="semantic_documents",
    persist_directory="./chroma_db",
    embedding_function=embedding_function
)

print("LangChain connected successfully!")

# STEP 6: Load user queries
queries = pd.read_csv("./dataset/queries.csv")

search_results = []
similarity_scores = []

# STEP 7: Perform semantic search
for _, row in queries.iterrows():

    query = row["query"]

    results = vectorstore.similarity_search_with_score(
        query,
        k=5
    )

    for doc, distance in results:

        search_results.append({
            "query": query,
            "document": doc.metadata.get("source", "Unknown"),
            "content": doc.page_content
        })

        similarity_scores.append({
            "query": query,
            "document": doc.metadata.get("source", "Unknown"),
            "distance": distance
        })

# STEP 8: Save semantic results
pd.DataFrame(search_results).to_csv(
    "./dataset/search_results.csv",
    index=False
)

pd.DataFrame(similarity_scores).to_csv(
    "./dataset/similarity_scores.csv",
    index=False
)

print("Semantic search completed!")
print("Total queries:", len(queries))
print("Total retrieved results:", len(search_results))

# STEP 9: Keyword-based search
keyword_results = []

for _, row in queries.iterrows():

    query = row["query"].lower()
    keywords = query.split()

    scores = []

    for doc_id, content in zip(ids, documents):

        score = sum(
            1 for word in keywords
            if word in content.lower()
        )

        scores.append({
            "query": query,
            "document": doc_id,
            "keyword_score": score
        })

    scores.sort(
        key=lambda x: x["keyword_score"],
        reverse=True
    )

    keyword_results.extend(scores[:5])

# STEP 10: Save keyword results
keyword_df = pd.DataFrame(keyword_results)

keyword_df.to_csv(
    "./dataset/keyword_results.csv",
    index=False
)

# STEP 11: Compare both search methods
comparison = pd.merge(
    pd.DataFrame(search_results),
    keyword_df,
    left_on=["query", "document"],
    right_on=["query", "document"],
    how="outer"
)

comparison.to_csv(
    "./dataset/response_comparison.csv",
    index=False
)

print("Keyword search completed!")
print("Comparison completed!")
print("All CSV files saved successfully!")
