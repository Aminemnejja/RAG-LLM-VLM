"""
Configuration centralisée pour le pipeline RAG médical.

Ce module fournit une configuration unifiée pour tous les composants
du pipeline avec validation, valeurs par défaut, et support des
variables d'environnement.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Literal
from functools import lru_cache


@dataclass
class EmbeddingConfig:
    """Configuration pour le générateur d'embeddings."""
    
    # Modèle PubMedBERT optimisé pour le domaine médical
    model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    dimension: int = 768
    max_seq_length: int = 512
    batch_size: int = 32
    normalize_embeddings: bool = True
    device: Optional[str] = None  # Auto-detect GPU/CPU
    cache_dir: Optional[Path] = None


@dataclass
class FAISSConfig:
    """Configuration pour le vector store FAISS."""
    
    index_type: Literal["flat", "ivf", "hnsw"] = "flat"
    metric: Literal["cosine", "l2", "ip"] = "cosine"
    nlist: int = 100  # Nombre de clusters pour IVF
    nprobe: int = 10  # Nombre de clusters à rechercher
    ef_construction: int = 200  # Pour HNSW
    ef_search: int = 50  # Pour HNSW
    index_path: Optional[Path] = None


@dataclass
class ChunkingConfig:
    """Configuration pour le chunking des documents."""
    
    chunk_size: int = 512
    chunk_overlap: int = 50
    min_chunk_size: int = 100
    separator: str = "\n\n"
    keep_separator: bool = True


@dataclass
class RetrieverConfig:
    """Configuration pour le RAG retriever."""
    
    top_k: int = 10  # Nombre de documents à récupérer
    top_k_reranked: int = 5  # Après reranking
    use_query_expansion: bool = True
    expansion_model: str = "gpt-3.5-turbo"
    similarity_threshold: float = 0.5


@dataclass
class RerankerConfig:
    """Configuration pour le reranker cross-encoder."""
    
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    batch_size: int = 16
    max_length: int = 512
    device: Optional[str] = None


@dataclass
class LLMConfig:
    """Configuration pour le pipeline LLM."""
    
    model_name: str = "gpt-4"
    temperature: float = 0.1
    max_tokens: int = 1024
    api_base: Optional[str] = None
    api_key: Optional[str] = None
    
    # Prompt templates
    system_prompt: str = """Tu es un assistant médical spécialisé en dermatologie.
Tu analyses les caractéristiques des lésions cutanées et fournis des diagnostics
basés sur les informations du contexte médical fourni.
Sois précis, cite tes sources, et recommande toujours une consultation médicale."""


@dataclass
class VLMConfig:
    """Configuration pour le pipeline VLM."""
    
    model_name: str = "gpt-4-vision-preview"
    temperature: float = 0.1
    max_tokens: int = 1024
    api_base: Optional[str] = None
    api_key: Optional[str] = None
    max_image_size: int = 2048  # pixels
    detail: Literal["low", "high", "auto"] = "high"
    
    system_prompt: str = """Tu es un dermatologue expert analysant des images de lésions cutanées.
Examine attentivement l'image et utilise le contexte médical fourni pour:
1. Décrire les caractéristiques visuelles observées
2. Proposer des diagnostics différentiels
3. Recommander des examens complémentaires si nécessaire
Sois précis et professionnel."""


@dataclass
class CacheConfig:
    """Configuration pour le système de cache."""
    
    enabled: bool = True
    cache_dir: Path = field(default_factory=lambda: Path("./cache"))
    embeddings_cache: bool = True
    features_cache: bool = True
    max_cache_size_gb: float = 5.0
    ttl_hours: int = 168  # 1 semaine


@dataclass
class LoggingConfig:
    """Configuration pour le logging."""
    
    level: str = "INFO"
    format: str = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    log_dir: Path = field(default_factory=lambda: Path("./logs"))
    max_file_size_mb: int = 10
    backup_count: int = 5
    json_format: bool = False


@dataclass
class Settings:
    """
    Configuration globale du pipeline RAG.
    
    Utilisation:
        settings = get_settings()
        print(settings.embedding.model_name)
    """
    
    # Chemins du projet
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent)
    data_dir: Path = field(default_factory=lambda: Path("./data"))
    documents_dir: Path = field(default_factory=lambda: Path("./data/documents"))
    index_dir: Path = field(default_factory=lambda: Path("./data/index"))
    
    # Configurations des composants
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    faiss: FAISSConfig = field(default_factory=FAISSConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    vlm: VLMConfig = field(default_factory=VLMConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    def __post_init__(self):
        """Initialise les chemins et charge les variables d'environnement."""
        self._load_env_vars()
        self._ensure_directories()
    
    def _load_env_vars(self) -> None:
        """Charge les variables d'environnement pour les secrets."""
        if os.getenv("OPENAI_API_KEY"):
            self.llm.api_key = os.getenv("OPENAI_API_KEY")
            self.vlm.api_key = os.getenv("OPENAI_API_KEY")
        
        if os.getenv("OPENAI_API_BASE"):
            self.llm.api_base = os.getenv("OPENAI_API_BASE")
            self.vlm.api_base = os.getenv("OPENAI_API_BASE")
        
        if os.getenv("RAG_LOG_LEVEL"):
            self.logging.level = os.getenv("RAG_LOG_LEVEL")
    
    def _ensure_directories(self) -> None:
        """Crée les répertoires nécessaires s'ils n'existent pas."""
        for dir_path in [self.data_dir, self.documents_dir, self.index_dir,
                         self.cache.cache_dir, self.logging.log_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Retourne une instance singleton de Settings.
    
    Returns:
        Settings: Configuration globale du pipeline.
    
    Example:
        >>> settings = get_settings()
        >>> settings.embedding.model_name
        'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'
    """
    return Settings()


def reset_settings() -> None:
    """Réinitialise le cache des settings (utile pour les tests)."""
    get_settings.cache_clear()
