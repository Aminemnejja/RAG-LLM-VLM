"""
Reranker Cross-Encoder pour améliorer la pertinence des résultats.

Utilise un modèle cross-encoder pour scorer la pertinence
query-document de manière plus précise que la similarité vectorielle.
"""

from typing import List, Optional
import numpy as np

from ..config.settings import get_settings
from ..vectorstore.faiss_store import SearchResult
from ..utils.logging_config import get_logger

logger = get_logger("cross_encoder")


class CrossEncoderReranker:
    """
    Reranker basé sur un modèle Cross-Encoder.
    
    Les cross-encoders sont plus précis que les bi-encoders pour
    scorer la pertinence car ils analysent query et document ensemble.
    
    Example:
        >>> reranker = CrossEncoderReranker()
        >>> reranked_docs = reranker.rerank("What is melanoma?", docs)
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: Optional[int] = None
    ):
        """
        Initialise le reranker.
        
        Args:
            model_name: Nom du modèle cross-encoder
            device: Device cible ('cuda', 'cpu', ou None pour auto)
            batch_size: Taille du batch pour l'inférence
        """
        settings = get_settings()
        
        self.model_name = model_name or settings.reranker.model_name
        self.device = device or settings.reranker.device
        self.batch_size = batch_size or settings.reranker.batch_size
        self.max_length = settings.reranker.max_length
        
        # Lazy loading
        self._model = None
        
        logger.info(f"CrossEncoderReranker configuré avec {self.model_name}")
    
    @property
    def model(self):
        """Lazy loading du modèle."""
        if self._model is None:
            self._load_model()
        return self._model
    
    def _load_model(self) -> None:
        """Charge le modèle cross-encoder."""
        try:
            from sentence_transformers import CrossEncoder
            import torch
            
            logger.info(f"Chargement du modèle: {self.model_name}")
            
            # Auto-detect device
            if self.device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            self._model = CrossEncoder(
                self.model_name,
                max_length=self.max_length,
                device=self.device
            )
            
            logger.info(f"Modèle chargé sur {self.device}")
            
        except ImportError as e:
            raise ImportError(
                "Installez sentence-transformers: pip install sentence-transformers"
            ) from e
    
    def rerank(
        self,
        query: str,
        documents: List[SearchResult],
        top_k: Optional[int] = None
    ) -> List[SearchResult]:
        """
        Reranque les documents par pertinence avec la requête.
        
        Args:
            query: Requête utilisateur
            documents: Documents à reranker
            top_k: Nombre de documents à retourner (None = tous)
        
        Returns:
            Documents reranqués par score décroissant
        """
        if not documents:
            return []
        
        logger.debug(f"Reranking {len(documents)} documents")
        
        # Préparer les paires query-document
        pairs = [[query, doc.content] for doc in documents]
        
        # Scorer avec le cross-encoder
        scores = self.model.predict(pairs, batch_size=self.batch_size)
        
        # Normaliser les scores (sigmoïde)
        normalized_scores = self._normalize_scores(scores)
        
        # Combiner scores originaux et reranking
        reranked_docs = []
        for doc, new_score in zip(documents, normalized_scores):
            # Créer une copie avec le nouveau score
            reranked_doc = SearchResult(
                id=doc.id,
                score=float(new_score),  # Score du reranker
                content=doc.content,
                metadata={
                    **doc.metadata,
                    "original_score": doc.score,
                    "rerank_score": float(new_score)
                }
            )
            reranked_docs.append(reranked_doc)
        
        # Trier par nouveau score
        reranked_docs.sort(key=lambda x: x.score, reverse=True)
        
        # Limiter si demandé
        if top_k:
            reranked_docs = reranked_docs[:top_k]
        
        logger.debug(f"Reranking terminé, top score: {reranked_docs[0].score:.4f}")
        return reranked_docs
    
    def score_pairs(
        self,
        query: str,
        documents: List[str]
    ) -> np.ndarray:
        """
        Calcule les scores de pertinence pour des paires query-document.
        
        Args:
            query: Requête
            documents: Liste de textes de documents
        
        Returns:
            Array de scores normalisés
        """
        pairs = [[query, doc] for doc in documents]
        scores = self.model.predict(pairs, batch_size=self.batch_size)
        return self._normalize_scores(scores)
    
    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Normalise les scores avec sigmoïde pour les ramener entre 0 et 1."""
        return 1 / (1 + np.exp(-scores))
    
    def compute_relevance(
        self,
        query: str,
        document: str
    ) -> float:
        """
        Calcule le score de pertinence pour une paire unique.
        
        Args:
            query: Requête
            document: Texte du document
        
        Returns:
            Score de pertinence normalisé
        """
        scores = self.score_pairs(query, [document])
        return float(scores[0])


class BM25Reranker:
    """
    Reranker simple basé sur BM25 (fallback sans modèle).
    
    Utile comme baseline ou quand les ressources sont limitées.
    """
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Initialise le reranker BM25.
        
        Args:
            k1: Paramètre de saturation des termes
            b: Paramètre de normalisation par longueur
        """
        self.k1 = k1
        self.b = b
        logger.info("BM25Reranker initialisé")
    
    def rerank(
        self,
        query: str,
        documents: List[SearchResult],
        top_k: Optional[int] = None
    ) -> List[SearchResult]:
        """
        Reranque les documents avec BM25.
        
        Args:
            query: Requête utilisateur
            documents: Documents à reranker
            top_k: Nombre de documents à retourner
        
        Returns:
            Documents reranqués
        """
        if not documents:
            return []
        
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            logger.warning("rank_bm25 non installé, retourne documents sans reranking")
            return documents[:top_k] if top_k else documents
        
        # Tokenization simple
        tokenized_docs = [doc.content.lower().split() for doc in documents]
        query_tokens = query.lower().split()
        
        # Calculer BM25
        bm25 = BM25Okapi(tokenized_docs, k1=self.k1, b=self.b)
        scores = bm25.get_scores(query_tokens)
        
        # Normaliser les scores
        max_score = max(scores) if max(scores) > 0 else 1
        normalized_scores = scores / max_score
        
        # Créer les documents reranqués
        reranked_docs = []
        for doc, score in zip(documents, normalized_scores):
            reranked_doc = SearchResult(
                id=doc.id,
                score=float(score),
                content=doc.content,
                metadata={**doc.metadata, "bm25_score": float(score)}
            )
            reranked_docs.append(reranked_doc)
        
        reranked_docs.sort(key=lambda x: x.score, reverse=True)
        
        return reranked_docs[:top_k] if top_k else reranked_docs
