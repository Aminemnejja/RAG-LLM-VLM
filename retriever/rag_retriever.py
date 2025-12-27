"""
RAG Retriever pour la récupération de documents médicaux.

Combine la recherche vectorielle avec query expansion et reranking
pour maximiser la pertinence des résultats.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np

from ..config.settings import get_settings
from ..embeddings.embedding_generator import EmbeddingGenerator
from ..vectorstore.faiss_store import FAISSVectorStore, SearchResult
from ..utils.logging_config import get_logger

logger = get_logger("rag_retriever")


@dataclass
class RetrievalResult:
    """
    Résultat de récupération RAG enrichi.
    
    Attributes:
        query: Requête originale
        expanded_queries: Requêtes étendues générées
        documents: Documents récupérés avec scores
        reranked: Indique si le reranking a été appliqué
        metadata: Métadonnées de la recherche
    """
    query: str
    expanded_queries: List[str] = field(default_factory=list)
    documents: List[SearchResult] = field(default_factory=list)
    reranked: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def context(self) -> str:
        """Retourne le contexte combiné des documents pour le LLM."""
        contexts = []
        for i, doc in enumerate(self.documents, 1):
            source = doc.metadata.get("source", "Unknown")
            contexts.append(f"[Document {i} - {source}]\n{doc.content}")
        return "\n\n---\n\n".join(contexts)
    
    @property
    def top_document(self) -> Optional[SearchResult]:
        """Retourne le document le plus pertinent."""
        return self.documents[0] if self.documents else None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "expanded_queries": self.expanded_queries,
            "documents": [doc.to_dict() for doc in self.documents],
            "reranked": self.reranked,
            "metadata": self.metadata
        }


class RAGRetriever:
    """
    Système de récupération RAG complet pour le domaine médical.
    
    Combine:
    - Recherche vectorielle avec FAISS
    - Query expansion pour améliorer le recall
    - Reranking optionnel pour améliorer la précision
    
    Example:
        >>> retriever = RAGRetriever.from_index("./data/index")
        >>> result = retriever.retrieve("What causes melanoma?", top_k=5)
        >>> print(result.context)
    """
    
    def __init__(
        self,
        embedding_generator: EmbeddingGenerator,
        vector_store: FAISSVectorStore,
        reranker: Optional[Any] = None,
        use_query_expansion: Optional[bool] = None,
        top_k: Optional[int] = None,
        top_k_reranked: Optional[int] = None
    ):
        """
        Initialise le RAG Retriever.
        
        Args:
            embedding_generator: Générateur d'embeddings
            vector_store: Vector store FAISS
            reranker: Reranker optionnel
            use_query_expansion: Activer l'expansion de requêtes
            top_k: Nombre de documents à récupérer
            top_k_reranked: Nombre final après reranking
        """
        settings = get_settings()
        
        self.embedding_generator = embedding_generator
        self.vector_store = vector_store
        self.reranker = reranker
        
        self.use_query_expansion = use_query_expansion if use_query_expansion is not None else settings.retriever.use_query_expansion
        self.top_k = top_k or settings.retriever.top_k
        self.top_k_reranked = top_k_reranked or settings.retriever.top_k_reranked
        self.similarity_threshold = settings.retriever.similarity_threshold
        
        logger.info(
            f"RAGRetriever initialisé: top_k={self.top_k}, "
            f"expansion={self.use_query_expansion}"
        )
    
    @classmethod
    def from_index(
        cls,
        index_path: str,
        reranker: Optional[Any] = None
    ) -> "RAGRetriever":
        """
        Crée un retriever à partir d'un index existant.
        
        Args:
            index_path: Chemin vers l'index sauvegardé
            reranker: Reranker optionnel
        
        Returns:
            Instance de RAGRetriever
        """
        from pathlib import Path
        
        vector_store = FAISSVectorStore.load(Path(index_path))
        embedding_generator = EmbeddingGenerator()
        
        return cls(
            embedding_generator=embedding_generator,
            vector_store=vector_store,
            reranker=reranker
        )
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        use_reranking: bool = True,
        filters: Optional[Dict[str, Any]] = None
    ) -> RetrievalResult:
        """
        Récupère les documents pertinents pour une requête.
        
        Args:
            query: Requête utilisateur
            top_k: Override du nombre de résultats
            use_reranking: Appliquer le reranking
            filters: Filtres sur les métadonnées
        
        Returns:
            RetrievalResult avec documents et contexte
        """
        logger.info(f"Récupération pour: {query[:100]}...")
        
        top_k = top_k or self.top_k
        expanded_queries = []
        
        # Query expansion (optionnel)
        if self.use_query_expansion:
            expanded_queries = self._expand_query(query)
            all_queries = [query] + expanded_queries
        else:
            all_queries = [query]
        
        # Recherche pour toutes les requêtes
        all_results = []
        seen_ids = set()
        
        for q in all_queries:
            query_embedding = self.embedding_generator.embed_query(q)
            results = self.vector_store.search(
                query_embedding,
                top_k=top_k,
                threshold=self.similarity_threshold
            )
            
            for result in results:
                if result.id not in seen_ids:
                    # Appliquer les filtres
                    if filters and not self._match_filters(result, filters):
                        continue
                    all_results.append(result)
                    seen_ids.add(result.id)
        
        # Trier par score
        all_results.sort(key=lambda x: x.score, reverse=True)
        
        # Limiter aux top_k
        all_results = all_results[:top_k]
        
        # Reranking (optionnel)
        reranked = False
        if use_reranking and self.reranker and all_results:
            all_results = self.reranker.rerank(query, all_results)
            all_results = all_results[:self.top_k_reranked]
            reranked = True
        
        result = RetrievalResult(
            query=query,
            expanded_queries=expanded_queries,
            documents=all_results,
            reranked=reranked,
            metadata={
                "total_retrieved": len(all_results),
                "queries_used": len(all_queries)
            }
        )
        
        logger.info(f"Récupéré {len(all_results)} documents")
        return result
    
    def retrieve_with_scores(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Récupère les documents avec scores détaillés.
        
        Utile pour l'évaluation et le debugging.
        
        Args:
            query: Requête utilisateur
            top_k: Nombre de résultats
        
        Returns:
            Liste de dictionnaires avec documents et scores
        """
        result = self.retrieve(query, top_k=top_k, use_reranking=False)
        
        return [
            {
                "id": doc.id,
                "content": doc.content,
                "score": doc.score,
                "source": doc.metadata.get("source"),
                "metadata": doc.metadata
            }
            for doc in result.documents
        ]
    
    def _expand_query(self, query: str) -> List[str]:
        """
        Génère des variantes de la requête pour améliorer le recall.
        
        Utilise des techniques simples ou un LLM pour la reformulation.
        """
        expanded = []
        
        # Expansion médicale simple (synonymes communs)
        medical_synonyms = {
            "skin cancer": ["melanoma", "carcinoma", "skin tumor"],
            "rash": ["dermatitis", "skin eruption", "skin inflammation"],
            "mole": ["nevus", "birthmark", "pigmented lesion"],
            "itching": ["pruritus", "scratching", "skin irritation"],
            "spot": ["macule", "patch", "lesion"],
            "bump": ["papule", "nodule", "swelling"],
        }
        
        query_lower = query.lower()
        for term, synonyms in medical_synonyms.items():
            if term in query_lower:
                for syn in synonyms[:2]:  # Limiter à 2 synonymes
                    expanded.append(query_lower.replace(term, syn))
        
        # Ajouter une version reformulée simple
        if "what is" not in query_lower:
            expanded.append(f"What is {query}")
        if "symptoms" not in query_lower:
            expanded.append(f"symptoms of {query}")
        
        # Limiter le nombre d'expansions
        return expanded[:3]
    
    def _match_filters(
        self,
        result: SearchResult,
        filters: Dict[str, Any]
    ) -> bool:
        """Vérifie si un résultat correspond aux filtres."""
        for key, value in filters.items():
            if key not in result.metadata:
                return False
            if isinstance(value, list):
                if result.metadata[key] not in value:
                    return False
            elif result.metadata[key] != value:
                return False
        return True
    
    def add_documents(
        self,
        contents: List[str],
        ids: List[str],
        metadata_list: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """
        Ajoute des documents à l'index.
        
        Args:
            contents: Contenus textuels
            ids: Identifiants uniques
            metadata_list: Métadonnées optionnelles
        """
        logger.info(f"Indexation de {len(contents)} documents")
        
        embeddings = self.embedding_generator.embed_texts(contents, show_progress=True)
        
        self.vector_store.add_documents(
            contents=contents,
            embeddings=embeddings,
            ids=ids,
            metadata_list=metadata_list
        )
        
        logger.info("Documents indexés avec succès")
    
    def save_index(self, path: str) -> None:
        """Sauvegarde l'index sur disque."""
        from pathlib import Path
        self.vector_store.save(Path(path))
    
    @property
    def document_count(self) -> int:
        """Nombre de documents dans l'index."""
        return self.vector_store.count
