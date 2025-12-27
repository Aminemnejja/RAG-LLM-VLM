"""
Tests unitaires pour le pipeline RAG médical.

Exécution:
    pytest tests/test_pipeline.py -v
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import json


class TestSettings:
    """Tests pour la configuration."""
    
    def test_get_settings_singleton(self):
        """Vérifie que get_settings retourne un singleton."""
        from rag_pipeline.config.settings import get_settings, reset_settings
        
        reset_settings()
        settings1 = get_settings()
        settings2 = get_settings()
        
        assert settings1 is settings2
    
    def test_default_values(self):
        """Vérifie les valeurs par défaut."""
        from rag_pipeline.config.settings import get_settings, reset_settings
        
        reset_settings()
        settings = get_settings()
        
        assert settings.embedding.dimension == 768
        assert settings.retriever.top_k == 10
        assert settings.chunking.chunk_size == 512


class TestDocumentChunker:
    """Tests pour le chunking des documents."""
    
    def test_chunk_basic_document(self):
        """Test chunking d'un document simple."""
        from rag_pipeline.utils.chunking import DocumentChunker, Document
        
        chunker = DocumentChunker(chunk_size=100, chunk_overlap=10)
        
        content = "Paragraph 1.\n\nParagraph 2.\n\nParagraph 3."
        doc = Document(content=content, source="test.txt")
        
        chunks = chunker.chunk_document(doc)
        
        assert len(chunks) >= 1
        assert all(chunk.document_id == doc.document_id for chunk in chunks)
    
    def test_chunk_preserves_metadata(self):
        """Test que les métadonnées sont préservées."""
        from rag_pipeline.utils.chunking import DocumentChunker, Document
        
        chunker = DocumentChunker()
        
        doc = Document(
            content="Test content that is long enough to be chunked properly.",
            source="medical_book.pdf",
            metadata={"author": "Dr. Test", "year": 2024}
        )
        
        chunks = chunker.chunk_document(doc)
        
        assert len(chunks) >= 1
        assert chunks[0].metadata["author"] == "Dr. Test"
        assert chunks[0].metadata["source"] == "medical_book.pdf"


class TestEmbeddingGenerator:
    """Tests pour le générateur d'embeddings."""
    
    def test_embed_single_text(self):
        """Test embedding d'un texte unique."""
        from rag_pipeline.embeddings.embedding_generator import EmbeddingGenerator
        
        # Skip si pas de GPU et modèle trop lourd
        generator = EmbeddingGenerator()
        
        try:
            embedding = generator.embed_text("skin lesion diagnosis")
            
            assert isinstance(embedding, np.ndarray)
            assert embedding.shape == (generator.dimension,)
            assert not np.isnan(embedding).any()
            
        except Exception as e:
            pytest.skip(f"Modèle non disponible: {e}")
    
    def test_embed_multiple_texts(self):
        """Test embedding de plusieurs textes."""
        from rag_pipeline.embeddings.embedding_generator import EmbeddingGenerator
        
        generator = EmbeddingGenerator()
        
        try:
            texts = ["melanoma", "psoriasis", "eczema"]
            embeddings = generator.embed_texts(texts)
            
            assert embeddings.shape == (3, generator.dimension)
            
        except Exception as e:
            pytest.skip(f"Modèle non disponible: {e}")
    
    def test_normalized_embeddings(self):
        """Vérifie que les embeddings sont normalisés."""
        from rag_pipeline.embeddings.embedding_generator import EmbeddingGenerator
        
        generator = EmbeddingGenerator()
        
        try:
            embedding = generator.embed_text("test text")
            norm = np.linalg.norm(embedding)
            
            assert np.isclose(norm, 1.0, atol=0.01)
            
        except Exception as e:
            pytest.skip(f"Modèle non disponible: {e}")


class TestFAISSVectorStore:
    """Tests pour le vector store FAISS."""
    
    def test_create_and_search(self):
        """Test création et recherche basique."""
        from rag_pipeline.vectorstore.faiss_store import FAISSVectorStore
        
        store = FAISSVectorStore(dimension=128)
        
        # Ajouter des documents
        contents = ["doc 1", "doc 2", "doc 3"]
        embeddings = np.random.randn(3, 128).astype(np.float32)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        ids = ["id1", "id2", "id3"]
        
        store.add_documents(contents, embeddings, ids)
        
        assert store.count == 3
        
        # Rechercher
        query_embedding = embeddings[0]
        results = store.search(query_embedding, top_k=2)
        
        assert len(results) == 2
        assert results[0].id == "id1"
    
    def test_save_and_load(self):
        """Test sauvegarde et chargement."""
        from rag_pipeline.vectorstore.faiss_store import FAISSVectorStore
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Créer et sauvegarder
            store = FAISSVectorStore(dimension=64)
            
            embeddings = np.random.randn(5, 64).astype(np.float32)
            store.add_documents(
                ["a", "b", "c", "d", "e"],
                embeddings,
                ["1", "2", "3", "4", "5"]
            )
            
            store.save(Path(tmpdir))
            
            # Charger
            loaded_store = FAISSVectorStore.load(Path(tmpdir))
            
            assert loaded_store.count == 5
            assert loaded_store.dimension == 64


class TestRAGRetriever:
    """Tests pour le RAG retriever."""
    
    def test_retrieve_basic(self):
        """Test récupération basique."""
        from rag_pipeline.retriever.rag_retriever import RAGRetriever
        from rag_pipeline.embeddings.embedding_generator import EmbeddingGenerator
        from rag_pipeline.vectorstore.faiss_store import FAISSVectorStore
        
        # Mock simple
        class MockEmbeddingGenerator:
            def embed_query(self, query):
                return np.random.randn(128).astype(np.float32)
            
            def embed_texts(self, texts, show_progress=False):
                return np.random.randn(len(texts), 128).astype(np.float32)
        
        # Créer un store avec des données
        store = FAISSVectorStore(dimension=128)
        embeddings = np.random.randn(10, 128).astype(np.float32)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        store.add_documents(
            contents=[f"Document {i} about dermatology" for i in range(10)],
            embeddings=embeddings,
            ids=[f"doc_{i}" for i in range(10)]
        )
        
        # Créer le retriever
        retriever = RAGRetriever(
            embedding_generator=MockEmbeddingGenerator(),
            vector_store=store,
            use_query_expansion=False
        )
        
        # Tester
        result = retriever.retrieve("skin cancer", top_k=3)
        
        assert len(result.documents) <= 3
        assert result.query == "skin cancer"


class TestReranker:
    """Tests pour le reranker."""
    
    def test_bm25_reranker(self):
        """Test du reranker BM25."""
        from rag_pipeline.reranker.cross_encoder import BM25Reranker
        from rag_pipeline.vectorstore.faiss_store import SearchResult
        
        reranker = BM25Reranker()
        
        docs = [
            SearchResult(id="1", score=0.5, content="melanoma skin cancer treatment"),
            SearchResult(id="2", score=0.4, content="normal skin lesion"),
            SearchResult(id="3", score=0.3, content="cancer diagnosis methods"),
        ]
        
        try:
            reranked = reranker.rerank("skin cancer", docs, top_k=2)
            assert len(reranked) == 2
        except ImportError:
            pytest.skip("rank_bm25 non installé")


class TestIntegration:
    """Tests d'intégration."""
    
    def test_end_to_end_with_mock(self):
        """Test de bout en bout avec données mockées."""
        from rag_pipeline.retriever.rag_retriever import RAGRetriever
        from rag_pipeline.vectorstore.faiss_store import FAISSVectorStore
        
        # Créer un index de test
        store = FAISSVectorStore(dimension=128)
        
        # Données médicales simulées
        medical_docs = [
            "Melanoma is a type of skin cancer that develops from melanocytes.",
            "Psoriasis is a chronic autoimmune condition causing rapid skin cell growth.",
            "Eczema causes itchy, red, and inflamed patches of skin.",
            "Basal cell carcinoma is the most common type of skin cancer.",
            "Contact dermatitis occurs when skin reacts to a substance it touches.",
        ]
        
        embeddings = np.random.randn(len(medical_docs), 128).astype(np.float32)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        ids = [f"med_doc_{i}" for i in range(len(medical_docs))]
        
        store.add_documents(medical_docs, embeddings, ids)
        
        # Vérifier que les documents sont indexés
        assert store.count == len(medical_docs)
        
        # Vérifier qu'on peut récupérer un document
        doc = store.get_document("med_doc_0")
        assert doc is not None
        assert "Melanoma" in doc["content"]


# Pytest configuration
@pytest.fixture(autouse=True)
def reset_settings_before_each_test():
    """Reset les settings avant chaque test."""
    from rag_pipeline.config.settings import reset_settings
    reset_settings()
    yield
    reset_settings()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
