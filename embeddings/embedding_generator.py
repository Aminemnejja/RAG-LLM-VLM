"""
Générateur d'embeddings pour documents médicaux.

Utilise PubMedBERT optimisé pour le domaine médical avec support
GPU/CPU automatique, batch processing, et cache intelligent.
"""

import hashlib
from pathlib import Path
from typing import List, Optional, Union
import json
import numpy as np

from ..config.settings import get_settings
from ..utils.logging_config import get_logger

logger = get_logger("embedding_generator")


class EmbeddingGenerator:
    """
    Génère des embeddings haute qualité pour le domaine médical.
    
    Utilise PubMedBERT par défaut, optimisé pour les textes biomédicaux.
    Supporte le batch processing et le caching des embeddings.
    
    Example:
        >>> generator = EmbeddingGenerator()
        >>> embeddings = generator.embed_texts(["skin lesion diagnosis", "melanoma treatment"])
        >>> print(embeddings.shape)  # (2, 768)
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        cache_enabled: Optional[bool] = None
    ):
        """
        Initialise le générateur d'embeddings.
        
        Args:
            model_name: Nom du modèle HuggingFace (défaut: PubMedBERT)
            device: Device cible ('cuda', 'cpu', ou None pour auto-detect)
            cache_enabled: Activer le cache des embeddings
        """
        settings = get_settings()
        
        self.model_name = model_name or settings.embedding.model_name
        self.device = device or settings.embedding.device
        self.dimension = settings.embedding.dimension
        self.max_seq_length = settings.embedding.max_seq_length
        self.batch_size = settings.embedding.batch_size
        self.normalize = settings.embedding.normalize_embeddings
        
        # Cache
        self.cache_enabled = cache_enabled if cache_enabled is not None else settings.cache.embeddings_cache
        self.cache_dir = settings.cache.cache_dir / "embeddings"
        
        # Lazy loading du modèle
        self._model = None
        self._tokenizer = None
        
        logger.info(f"EmbeddingGenerator configuré avec {self.model_name}")
    
    @property
    def model(self):
        """Lazy loading du modèle."""
        if self._model is None:
            self._load_model()
        return self._model
    
    @property
    def tokenizer(self):
        """Lazy loading du tokenizer."""
        if self._tokenizer is None:
            self._load_model()
        return self._tokenizer
    
    def _load_model(self) -> None:
        """Charge le modèle et le tokenizer."""
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
            
            logger.info(f"Chargement du modèle: {self.model_name}")
            
            # Auto-detect device
            if self.device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            logger.info(f"Device utilisé: {self.device}")
            
            # Charger tokenizer et modèle
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name)
            self._model.to(self.device)
            self._model.eval()
            
            logger.info("Modèle chargé avec succès")
            
        except ImportError as e:
            raise ImportError(
                "Installez les dépendances: pip install torch transformers"
            ) from e
    
    def embed_texts(
        self,
        texts: List[str],
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Génère les embeddings pour une liste de textes.
        
        Args:
            texts: Liste de textes à encoder
            show_progress: Afficher une barre de progression
        
        Returns:
            Array numpy de shape (n_texts, dimension)
        """
        import torch
        
        if not texts:
            return np.array([])
        
        # Vérifier le cache
        cache_key = self._get_cache_key(texts)
        if self.cache_enabled:
            cached = self._load_from_cache(cache_key)
            if cached is not None:
                logger.debug(f"Embeddings chargés du cache: {len(texts)} textes")
                return cached
        
        logger.info(f"Génération de {len(texts)} embeddings")
        
        all_embeddings = []
        
        # Batch processing
        iterator = range(0, len(texts), self.batch_size)
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(iterator, desc="Embeddings", unit="batch")
            except ImportError:
                pass
        
        for i in iterator:
            batch_texts = texts[i:i + self.batch_size]
            batch_embeddings = self._embed_batch(batch_texts)
            all_embeddings.append(batch_embeddings)
        
        embeddings = np.vstack(all_embeddings)
        
        # Normalisation
        if self.normalize:
            embeddings = self._normalize_embeddings(embeddings)
        
        # Sauvegarder dans le cache
        if self.cache_enabled:
            self._save_to_cache(cache_key, embeddings)
        
        logger.info(f"Embeddings générés: shape={embeddings.shape}")
        return embeddings
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Génère l'embedding pour un seul texte.
        
        Args:
            text: Texte à encoder
        
        Returns:
            Array numpy de shape (dimension,)
        """
        embeddings = self.embed_texts([text])
        return embeddings[0]
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Génère l'embedding pour une requête.
        
        Peut appliquer un traitement spécifique aux requêtes
        par rapport aux documents.
        
        Args:
            query: Requête utilisateur
        
        Returns:
            Array numpy de shape (dimension,)
        """
        # Pour certains modèles, on peut préfixer la requête
        # Exemple: "query: " pour les modèles E5
        return self.embed_text(query)
    
    def _embed_batch(self, texts: List[str]) -> np.ndarray:
        """Génère les embeddings pour un batch de textes."""
        import torch
        
        # Tokenization
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt"
        )
        
        # Move to device
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        
        # Inference
        with torch.no_grad():
            outputs = self.model(**encoded)
            
            # Mean pooling sur les tokens
            attention_mask = encoded["attention_mask"]
            token_embeddings = outputs.last_hidden_state
            
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1).clamp(min=1e-9)
            embeddings = sum_embeddings / sum_mask
        
        return embeddings.cpu().numpy()
    
    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Normalise les embeddings en norme L2."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return embeddings / norms
    
    def _get_cache_key(self, texts: List[str]) -> str:
        """Génère une clé de cache unique pour les textes."""
        content = json.dumps(texts, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()
    
    def _load_from_cache(self, cache_key: str) -> Optional[np.ndarray]:
        """Charge les embeddings depuis le cache."""
        cache_file = self.cache_dir / f"{cache_key}.npy"
        
        if cache_file.exists():
            try:
                return np.load(cache_file)
            except Exception as e:
                logger.warning(f"Erreur lecture cache: {e}")
        
        return None
    
    def _save_to_cache(self, cache_key: str, embeddings: np.ndarray) -> None:
        """Sauvegarde les embeddings dans le cache."""
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = self.cache_dir / f"{cache_key}.npy"
            np.save(cache_file, embeddings)
            logger.debug(f"Embeddings sauvegardés: {cache_file}")
        except Exception as e:
            logger.warning(f"Erreur sauvegarde cache: {e}")
    
    def clear_cache(self) -> None:
        """Vide le cache des embeddings."""
        import shutil
        
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            logger.info("Cache embeddings vidé")
