"""
RAG Pipeline pour Dermatologie Médicale.

Pipeline complet de Retrieval-Augmented Generation pour le diagnostic
dermatologique, supportant à la fois LLM (avec features OpenCV) et VLM
(vision directe).

Example:
    >>> from rag_pipeline import query_llm, query_vlm
    >>> 
    >>> # Diagnostic avec LLM (features + RAG)
    >>> result = query_llm("./lesion.jpg", "Quel est ce type de lésion?")
    >>> print(result.diagnosis)
    >>> 
    >>> # Diagnostic avec VLM (image directe + RAG)
    >>> result = query_vlm("./lesion.jpg", "Analyse cette lésion cutanée")
    >>> print(result.diagnosis)
"""

from pathlib import Path
from typing import Optional, Union

from .config.settings import Settings, get_settings
from .utils.logging_config import setup_logging, get_logger
from .embeddings.embedding_generator import EmbeddingGenerator
from .vectorstore.faiss_store import FAISSVectorStore
from .retriever.rag_retriever import RAGRetriever, RetrievalResult
from .reranker.cross_encoder import CrossEncoderReranker
from .pipelines.llm_pipeline import LLMPipeline, DiagnosticResult, FeatureExtractor
from .pipelines.vlm_pipeline import VLMPipeline

__version__ = "1.0.0"
__author__ = "Medical RAG Project"

# Logger du module principal
logger = get_logger("main")

# Singletons pour les pipelines
_llm_pipeline: Optional[LLMPipeline] = None
_vlm_pipeline: Optional[VLMPipeline] = None


def init_pipelines(
    index_path: Optional[str] = None,
    log_level: str = "INFO"
) -> None:
    """
    Initialise les pipelines LLM et VLM.
    
    Args:
        index_path: Chemin vers l'index FAISS (défaut: ./data/index)
        log_level: Niveau de log
    """
    global _llm_pipeline, _vlm_pipeline
    
    settings = get_settings()
    
    # Setup logging
    setup_logging(
        level=log_level,
        log_dir=settings.logging.log_dir
    )
    
    # Chemin de l'index
    if index_path is None:
        index_path = str(settings.index_dir)
    
    logger.info(f"Initialisation des pipelines avec index: {index_path}")
    
    # Créer le reranker
    reranker = CrossEncoderReranker()
    
    # Créer le retriever
    retriever = RAGRetriever.from_index(index_path, reranker=reranker)
    
    # Créer les pipelines
    _llm_pipeline = LLMPipeline(retriever=retriever)
    _vlm_pipeline = VLMPipeline(retriever=retriever)
    
    logger.info("Pipelines initialisés avec succès")


def get_llm_pipeline() -> LLMPipeline:
    """Retourne le pipeline LLM (initialise si nécessaire)."""
    if _llm_pipeline is None:
        init_pipelines()
    return _llm_pipeline


def get_vlm_pipeline() -> VLMPipeline:
    """Retourne le pipeline VLM (initialise si nécessaire)."""
    if _vlm_pipeline is None:
        init_pipelines()
    return _vlm_pipeline


def query_llm(
    image_path: Union[str, Path],
    query: str,
    use_rag: bool = True,
    use_features: bool = True
) -> DiagnosticResult:
    """
    Génère un diagnostic avec le pipeline LLM.
    
    Extrait les features visuelles de l'image avec OpenCV,
    récupère le contexte médical via RAG, et génère un diagnostic
    avec un LLM.
    
    Args:
        image_path: Chemin vers l'image de la lésion
        query: Question de l'utilisateur
        use_rag: Utiliser le contexte RAG
        use_features: Extraire les features de l'image
    
    Returns:
        DiagnosticResult avec diagnostic, différentiels et recommandations
    
    Example:
        >>> result = query_llm("./melanoma.jpg", "Est-ce un mélanome?")
        >>> print(result.diagnosis)
        >>> print(result.differential_diagnoses)
    """
    pipeline = get_llm_pipeline()
    return pipeline.diagnose(
        image_path=image_path,
        query=query,
        use_rag=use_rag,
        use_features=use_features
    )


def query_vlm(
    image_path: Union[str, Path],
    query: str,
    use_rag: bool = True
) -> DiagnosticResult:
    """
    Génère un diagnostic avec le pipeline VLM.
    
    Envoie directement l'image au Vision Language Model
    avec le contexte médical RAG pour une analyse visuelle
    complète.
    
    Args:
        image_path: Chemin vers l'image de la lésion
        query: Question de l'utilisateur
        use_rag: Utiliser le contexte RAG
    
    Returns:
        DiagnosticResult avec diagnostic, différentiels et recommandations
    
    Example:
        >>> result = query_vlm("./lesion.jpg", "Analyse cette lésion")
        >>> print(result.diagnosis)
    """
    pipeline = get_vlm_pipeline()
    return pipeline.diagnose(
        image_path=image_path,
        query=query,
        use_rag=use_rag
    )


def retrieve_context(
    query: str,
    top_k: int = 5
) -> RetrievalResult:
    """
    Récupère uniquement le contexte RAG sans générer de diagnostic.
    
    Utile pour l'évaluation du système de retrieval ou pour
    des usages personnalisés.
    
    Args:
        query: Requête de recherche
        top_k: Nombre de documents à récupérer
    
    Returns:
        RetrievalResult avec documents et contexte
    
    Example:
        >>> result = retrieve_context("traitement du psoriasis")
        >>> for doc in result.documents:
        ...     print(f"{doc.id}: {doc.score:.2f}")
    """
    pipeline = get_llm_pipeline()
    return pipeline.retriever.retrieve(query, top_k=top_k)


# Exports publics
__all__ = [
    # Fonctions principales
    "query_llm",
    "query_vlm",
    "retrieve_context",
    "init_pipelines",
    
    # Pipelines
    "LLMPipeline",
    "VLMPipeline",
    "get_llm_pipeline",
    "get_vlm_pipeline",
    
    # Composants
    "RAGRetriever",
    "EmbeddingGenerator",
    "FAISSVectorStore",
    "CrossEncoderReranker",
    "FeatureExtractor",
    
    # Types de données
    "DiagnosticResult",
    "RetrievalResult",
    "Settings",
    
    # Utilitaires
    "get_settings",
    "setup_logging",
]
