from sentence_transformers import SentenceTransformer
import torch
from transformers import pipeline
import faiss
import numpy as np
import os
import warnings

class RAGModel:
    def __init__(self, documents, model_name='all-MiniLM-L6-v2', generator_model='google/flan-t5-base'):
        """
        Initialize the RAG Model with documents, embedding model, and generator.

        :param documents: List of documents to embed and use for retrieval.
        :param model_name: Name of the SentenceTransformer model for embedding.
        :param generator_model: Name of the generator model (e.g., T5 or GPT).
        """
        # Suppress warnings
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        warnings.filterwarnings("ignore", category=FutureWarning)

        # Initialize the embedder and generator
        self.embedder = SentenceTransformer(
            model_name,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.generator = pipeline(
            "text2text-generation",
            model=generator_model,
            device=0 if torch.cuda.is_available() else -1
        )

        # Precompute document embeddings
        self.documents = documents
        self.document_embeddings = self.embed_documents(documents)

        # Create and populate the FAISS index
        self.index = self.create_faiss_index(self.document_embeddings)
        
    def embed_documents(self, documents):
        """
        Batch encode documents and ensure correct dtype for FAISS.
        """
        embeddings = self.embedder.encode(documents, batch_size=32, show_progress_bar=True)
        return np.array(embeddings, dtype=np.float32)

    def create_faiss_index(self, embeddings):
        """
        Create a FAISS index to store and search document embeddings.

        :param embeddings: The document embeddings to store in the FAISS index.
        :return: A FAISS index populated with document embeddings.
        """
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)

        # Use GPU if available
        if faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)

        index.add(embeddings)
        return index

    def rag_query(self, query, top_k=2, max_gen_length=100):
        """
        Perform Retrieval-Augmented Generation for a given query.

        :param query: The query string to ask.
        :param top_k: The number of top documents to retrieve for context.
        :return: The generated answer based on the retrieved documents.
        """
        # Embed the query
        query_embedding = self.embedder.encode([query], show_progress_bar=False)
        query_embedding = np.array(query_embedding, dtype=np.float32)

        # Retrieve top-k documents
        distances, indices = self.index.search(query_embedding, top_k)
        retrieved_docs = [self.documents[i] for i in indices[0]]

        # Combine context
        context = " ".join(retrieved_docs)
        prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"

        # Generate answer
        response = self.generator(prompt, max_length=max_gen_length, do_sample=False)
        result = response[0]['generated_text']
        result = result[:result.rfind('.') + 1] if '.' in result else result
        return result
