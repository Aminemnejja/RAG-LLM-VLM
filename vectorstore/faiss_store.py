"""
Vector Store FAISS pour la recherche de similarité.

Fournit un stockage efficace et une recherche rapide des embeddings
avec support de plusieurs types d'index et persistance.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import json
import numpy as np

from ..config.settings import get_settings
from ..utils.logging_config import get_logger

logger = get_logger("faiss_store")


@dataclass
class SearchResult:
    """
    Résultat d'une recherche de similarité.
    
    Attributes:
        id: Identifiant du document
        score: Score de similarité (0-1 pour cosine)
        content: Contenu textuel
        metadata: Métadonnées du document
    """
    id: str
    score: float
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "score": self.score,
            "content": self.content,
            "metadata": self.metadata
        }


class FAISSVectorStore:
    """
    Vector store basé sur FAISS pour recherche de similarité rapide.
    
    Supporte plusieurs types d'index (Flat, IVF, HNSW) avec
    persistance et métadonnées.
    
    Example:
        >>> store = FAISSVectorStore(dimension=768)
        >>> store.add_documents(texts, embeddings, ids)
        >>> results = store.search(query_embedding, top_k=5)
    """
    
    def __init__(
        self,
        dimension: Optional[int] = None,
        index_type: Optional[str] = None,
        metric: Optional[str] = None
    ):
        """
        Initialise le vector store.
        
        Args:
            dimension: Dimension des embeddings
            index_type: Type d'index ('flat', 'ivf', 'hnsw')
            metric: Métrique de distance ('cosine', 'l2', 'ip')
        """
        settings = get_settings()
        
        self.dimension = dimension or settings.embedding.dimension
        self.index_type = index_type or settings.faiss.index_type
        self.metric = metric or settings.faiss.metric
        self.nlist = settings.faiss.nlist
        self.nprobe = settings.faiss.nprobe
        
        # Index FAISS
        self._index = None
        
        # Stockage des documents et métadonnées
        self._documents: Dict[int, Dict[str, Any]] = {}
        self._id_to_idx: Dict[str, int] = {}
        self._idx_to_id: Dict[int, str] = {}
        self._current_idx = 0
        
        logger.info(
            f"FAISSVectorStore initialisé: dim={self.dimension}, "
            f"type={self.index_type}, metric={self.metric}"
        )
    
    @property
    def index(self):
        """Lazy loading de l'index FAISS."""
        if self._index is None:
            self._create_index()
        return self._index
    
    def _create_index(self) -> None:
        """Crée l'index FAISS selon la configuration."""
        try:
            import faiss
        except ImportError:
            raise ImportError("Installez FAISS: pip install faiss-cpu (ou faiss-gpu)")
        
        logger.info(f"Création de l'index {self.index_type}")
        
        # Créer l'index de base selon le type
        if self.index_type == "flat":
            if self.metric == "cosine" or self.metric == "ip":
                self._index = faiss.IndexFlatIP(self.dimension)
            else:
                self._index = faiss.IndexFlatL2(self.dimension)
        
        elif self.index_type == "ivf":
            # Quantizer pour IVF
            quantizer = faiss.IndexFlatIP(self.dimension) if self.metric == "cosine" else faiss.IndexFlatL2(self.dimension)
            
            if self.metric == "cosine" or self.metric == "ip":
                self._index = faiss.IndexIVFFlat(quantizer, self.dimension, self.nlist, faiss.METRIC_INNER_PRODUCT)
            else:
                self._index = faiss.IndexIVFFlat(quantizer, self.dimension, self.nlist, faiss.METRIC_L2)
        
        elif self.index_type == "hnsw":
            settings = get_settings()
            self._index = faiss.IndexHNSWFlat(self.dimension, settings.faiss.ef_construction)
            self._index.hnsw.efSearch = settings.faiss.ef_search
        
        else:
            raise ValueError(f"Type d'index non supporté: {self.index_type}")
        
        # Wrapper pour conserver les IDs
        self._index = faiss.IndexIDMap(self._index)
        
        logger.info("Index FAISS créé")
    
    def add_documents(
        self,
        contents: List[str],
        embeddings: np.ndarray,
        ids: List[str],
        metadata_list: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """
        Ajoute des documents au vector store.
        
        Args:
            contents: Liste des contenus textuels
            embeddings: Embeddings correspondants (n_docs, dimension)
            ids: Identifiants uniques des documents
            metadata_list: Métadonnées optionnelles pour chaque document
        """
        if len(contents) != len(embeddings) or len(contents) != len(ids):
            raise ValueError("Le nombre de contenus, embeddings et IDs doit être identique")
        
        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"Dimension attendue: {self.dimension}, reçue: {embeddings.shape[1]}")
        
        logger.info(f"Ajout de {len(contents)} documents")
        
        # Normaliser pour cosine similarity
        if self.metric == "cosine":
            embeddings = self._normalize(embeddings)
        
        # Préparer les métadonnées
        if metadata_list is None:
            metadata_list = [{} for _ in contents]
        
        # Mapper les IDs
        faiss_ids = []
        for i, doc_id in enumerate(ids):
            internal_idx = self._current_idx
            self._id_to_idx[doc_id] = internal_idx
            self._idx_to_id[internal_idx] = doc_id
            
            self._documents[internal_idx] = {
                "id": doc_id,
                "content": contents[i],
                "metadata": metadata_list[i]
            }
            
            faiss_ids.append(internal_idx)
            self._current_idx += 1
        
        # Ajouter à l'index
        faiss_ids_array = np.array(faiss_ids, dtype=np.int64)
        
        # Pour IVF, entraîner si nécessaire
        if self.index_type == "ivf" and not self.index.is_trained:
            logger.info("Entraînement de l'index IVF")
            self.index.train(embeddings)
        
        self.index.add_with_ids(embeddings.astype(np.float32), faiss_ids_array)
        
        logger.info(f"Documents ajoutés. Total: {self.index.ntotal}")
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        threshold: Optional[float] = None
    ) -> List[SearchResult]:
        """
        Recherche les documents les plus similaires.
        
        Args:
            query_embedding: Embedding de la requête (dimension,) ou (1, dimension)
            top_k: Nombre de résultats à retourner
            threshold: Score minimum (optionnel)
        
        Returns:
            Liste de SearchResult triés par score décroissant
        """
        if self.index.ntotal == 0:
            logger.warning("Index vide, aucun résultat")
            return []
        
        # Reshape si nécessaire
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Normaliser pour cosine
        if self.metric == "cosine":
            query_embedding = self._normalize(query_embedding)
        
        # Configurer nprobe pour IVF
        if self.index_type == "ivf":
            self.index.nprobe = self.nprobe
        
        # Recherche
        top_k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_embedding.astype(np.float32), top_k)
        
        # Construire les résultats
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            
            # Convertir le score selon la métrique
            if self.metric == "l2":
                # Pour L2, plus petit = meilleur, convertir en similarité
                normalized_score = 1 / (1 + score)
            else:
                # Pour IP/cosine, le score est déjà une similarité
                normalized_score = float(score)
            
            if threshold and normalized_score < threshold:
                continue
            
            doc = self._documents.get(idx)
            if doc:
                results.append(SearchResult(
                    id=doc["id"],
                    score=normalized_score,
                    content=doc["content"],
                    metadata=doc["metadata"]
                ))
        
        logger.debug(f"Recherche: {len(results)} résultats")
        return results
    
    def batch_search(
        self,
        query_embeddings: np.ndarray,
        top_k: int = 10
    ) -> List[List[SearchResult]]:
        """
        Recherche pour plusieurs requêtes en batch.
        
        Args:
            query_embeddings: Embeddings des requêtes (n_queries, dimension)
            top_k: Nombre de résultats par requête
        
        Returns:
            Liste de listes de SearchResult
        """
        results = []
        for query_emb in query_embeddings:
            results.append(self.search(query_emb, top_k))
        return results
    
    def _normalize(self, embeddings: np.ndarray) -> np.ndarray:
        """Normalise les embeddings en norme L2."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return embeddings / norms
    
    def save(self, path: Path) -> None:
        """
        Sauvegarde le vector store sur disque.
        
        Args:
            path: Répertoire de sauvegarde
        """
        import faiss
        
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Sauvegarder l'index FAISS
        index_path = path / "index.faiss"
        faiss.write_index(self.index, str(index_path))
        
        # Sauvegarder les documents et mappings
        metadata_path = path / "metadata.json"
        metadata = {
            "dimension": self.dimension,
            "index_type": self.index_type,
            "metric": self.metric,
            "current_idx": self._current_idx,
            "documents": {str(k): v for k, v in self._documents.items()},
            "id_to_idx": self._id_to_idx,
            "idx_to_id": {str(k): v for k, v in self._idx_to_id.items()}
        }
        
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Vector store sauvegardé: {path}")
    
    @classmethod
    def load(cls, path: Path) -> "FAISSVectorStore":
        """
        Charge un vector store depuis le disque.
        
        Args:
            path: Répertoire de sauvegarde
        
        Returns:
            Instance de FAISSVectorStore
        """
        import faiss
        
        path = Path(path)
        
        # Charger les métadonnées
        metadata_path = path / "metadata.json"
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        
        # Créer l'instance
        store = cls(
            dimension=metadata["dimension"],
            index_type=metadata["index_type"],
            metric=metadata["metric"]
        )
        
        # Charger l'index
        index_path = path / "index.faiss"
        store._index = faiss.read_index(str(index_path))
        
        # Restaurer les mappings
        store._current_idx = metadata["current_idx"]
        store._documents = {int(k): v for k, v in metadata["documents"].items()}
        store._id_to_idx = metadata["id_to_idx"]
        store._idx_to_id = {int(k): v for k, v in metadata["idx_to_id"].items()}
        
        logger.info(f"Vector store chargé: {store.index.ntotal} documents")
        return store
    
    def delete(self, ids: List[str]) -> int:
        """
        Supprime des documents par leur ID.
        
        Note: FAISS ne supporte pas nativement la suppression,
        donc on marque les documents comme supprimés.
        
        Args:
            ids: Liste des IDs à supprimer
        
        Returns:
            Nombre de documents supprimés
        """
        deleted = 0
        for doc_id in ids:
            if doc_id in self._id_to_idx:
                idx = self._id_to_idx[doc_id]
                del self._documents[idx]
                del self._id_to_idx[doc_id]
                del self._idx_to_id[idx]
                deleted += 1
        
        logger.info(f"{deleted} documents marqués comme supprimés")
        return deleted
    
    @property
    def count(self) -> int:
        """Retourne le nombre de documents dans l'index."""
        return len(self._documents)
    
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Récupère un document par son ID."""
        idx = self._id_to_idx.get(doc_id)
        if idx is not None:
            return self._documents.get(idx)
        return None
