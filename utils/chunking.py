"""
Module de chunking pour les documents médicaux.

Découpe les documents en chunks de taille optimale pour l'indexation
tout en préservant le contexte et les métadonnées.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator
import hashlib
import re

from ..config.settings import get_settings
from .logging_config import get_logger

logger = get_logger("chunking")


@dataclass
class DocumentChunk:
    """
    Représente un chunk de document avec ses métadonnées.
    
    Attributes:
        content: Contenu textuel du chunk
        chunk_id: Identifiant unique du chunk
        document_id: Identifiant du document source
        chunk_index: Index du chunk dans le document
        metadata: Métadonnées additionnelles
    """
    content: str
    chunk_id: str
    document_id: str
    chunk_index: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def token_count(self) -> int:
        """Estimation du nombre de tokens (approximatif)."""
        return len(self.content.split())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit le chunk en dictionnaire."""
        return {
            "content": self.content,
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "chunk_index": self.chunk_index,
            "metadata": self.metadata,
            "token_count": self.token_count
        }


@dataclass
class Document:
    """
    Représente un document source à découper.
    
    Attributes:
        content: Contenu textuel complet
        source: Chemin ou identifiant de la source
        metadata: Métadonnées du document
    """
    content: str
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def document_id(self) -> str:
        """Génère un ID unique basé sur le contenu."""
        return hashlib.md5(self.content.encode()).hexdigest()[:12]


class DocumentChunker:
    """
    Découpe les documents en chunks optimisés pour RAG.
    
    Supporte plusieurs stratégies de découpage avec préservation
    du contexte via l'overlap.
    
    Example:
        >>> chunker = DocumentChunker(chunk_size=512, overlap=50)
        >>> doc = Document(content="...", source="medical_book.pdf")
        >>> chunks = chunker.chunk_document(doc)
    """
    
    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        min_chunk_size: Optional[int] = None,
        separator: Optional[str] = None
    ):
        """
        Initialise le chunker avec les paramètres de configuration.
        
        Args:
            chunk_size: Taille cible des chunks en caractères
            chunk_overlap: Chevauchement entre chunks consécutifs
            min_chunk_size: Taille minimale d'un chunk valide
            separator: Séparateur pour le découpage initial
        """
        settings = get_settings()
        
        self.chunk_size = chunk_size or settings.chunking.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunking.chunk_overlap
        self.min_chunk_size = min_chunk_size or settings.chunking.min_chunk_size
        self.separator = separator or settings.chunking.separator
        
        logger.info(
            f"DocumentChunker initialisé: size={self.chunk_size}, "
            f"overlap={self.chunk_overlap}, min={self.min_chunk_size}"
        )
    
    def chunk_document(self, document: Document) -> List[DocumentChunk]:
        """
        Découpe un document en chunks.
        
        Args:
            document: Document à découper
        
        Returns:
            Liste de DocumentChunk
        """
        logger.debug(f"Chunking document: {document.source}")
        
        # Prétraitement du texte
        text = self._preprocess_text(document.content)
        
        # Découpage par paragraphes d'abord
        paragraphs = self._split_by_separator(text)
        
        # Fusion/découpage selon la taille cible
        chunks_content = self._merge_or_split_paragraphs(paragraphs)
        
        # Création des chunks avec métadonnées
        chunks = []
        for idx, content in enumerate(chunks_content):
            if len(content.strip()) < self.min_chunk_size:
                continue
            
            chunk = DocumentChunk(
                content=content.strip(),
                chunk_id=f"{document.document_id}_{idx:04d}",
                document_id=document.document_id,
                chunk_index=idx,
                metadata={
                    **document.metadata,
                    "source": document.source,
                    "total_chunks": len(chunks_content)
                }
            )
            chunks.append(chunk)
        
        logger.info(f"Document découpé en {len(chunks)} chunks")
        return chunks
    
    def chunk_documents(self, documents: List[Document]) -> List[DocumentChunk]:
        """
        Découpe plusieurs documents en chunks.
        
        Args:
            documents: Liste de documents
        
        Returns:
            Liste plate de tous les chunks
        """
        all_chunks = []
        for doc in documents:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)
        
        logger.info(f"Total: {len(all_chunks)} chunks pour {len(documents)} documents")
        return all_chunks
    
    def _preprocess_text(self, text: str) -> str:
        """Nettoie et normalise le texte."""
        # Normaliser les sauts de ligne
        text = re.sub(r'\r\n', '\n', text)
        # Supprimer les espaces multiples
        text = re.sub(r'[ \t]+', ' ', text)
        # Normaliser les paragraphes
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()
    
    def _split_by_separator(self, text: str) -> List[str]:
        """Découpe le texte par le séparateur configuré."""
        if self.separator:
            parts = text.split(self.separator)
            return [p.strip() for p in parts if p.strip()]
        return [text]
    
    def _merge_or_split_paragraphs(self, paragraphs: List[str]) -> List[str]:
        """
        Fusionne les petits paragraphes et découpe les grands.
        
        Utilise l'overlap pour préserver le contexte.
        """
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            # Si le paragraphe seul est trop grand, le découper
            if len(para) > self.chunk_size:
                # Sauver le chunk courant si non vide
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""
                
                # Découper le grand paragraphe
                para_chunks = self._split_large_text(para)
                chunks.extend(para_chunks)
            
            # Si ajouter le paragraphe dépasse la taille
            elif len(current_chunk) + len(para) + 1 > self.chunk_size:
                chunks.append(current_chunk)
                
                # Garder l'overlap du chunk précédent
                overlap_text = self._get_overlap_text(current_chunk)
                current_chunk = overlap_text + para
            
            else:
                # Ajouter le paragraphe au chunk courant
                if current_chunk:
                    current_chunk += self.separator + para
                else:
                    current_chunk = para
        
        # Ne pas oublier le dernier chunk
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _split_large_text(self, text: str) -> List[str]:
        """Découpe un texte trop grand en chunks avec overlap."""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Essayer de couper à une limite de phrase
            if end < len(text):
                # Chercher un point ou une fin de phrase
                for sep in ['. ', '.\n', '? ', '! ', '\n']:
                    last_sep = text[start:end].rfind(sep)
                    if last_sep > self.chunk_size // 2:
                        end = start + last_sep + len(sep)
                        break
            
            chunk = text[start:end]
            chunks.append(chunk.strip())
            
            # Avancer avec overlap
            start = end - self.chunk_overlap
        
        return chunks
    
    def _get_overlap_text(self, text: str) -> str:
        """Extrait le texte d'overlap de la fin d'un chunk."""
        if len(text) <= self.chunk_overlap:
            return text + " "
        return text[-self.chunk_overlap:] + " "


def load_document_from_file(file_path: Path) -> Document:
    """
    Charge un document depuis un fichier.
    
    Args:
        file_path: Chemin vers le fichier
    
    Returns:
        Document chargé
    
    Raises:
        ValueError: Si le format de fichier n'est pas supporté
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Fichier non trouvé: {file_path}")
    
    suffix = file_path.suffix.lower()
    
    if suffix == ".txt":
        content = file_path.read_text(encoding="utf-8")
    elif suffix == ".md":
        content = file_path.read_text(encoding="utf-8")
    elif suffix == ".pdf":
        content = _extract_pdf_text(file_path)
    else:
        raise ValueError(f"Format non supporté: {suffix}")
    
    return Document(
        content=content,
        source=str(file_path),
        metadata={
            "filename": file_path.name,
            "file_type": suffix,
            "file_size": file_path.stat().st_size
        }
    )


def _extract_pdf_text(file_path: Path) -> str:
    """Extrait le texte d'un PDF."""
    try:
        import pymupdf as fitz  # PyMuPDF
        
        doc = fitz.open(str(file_path))
        text_parts = []
        
        for page_num, page in enumerate(doc):
            text = page.get_text()
            text_parts.append(f"[Page {page_num + 1}]\n{text}")
        
        doc.close()
        return "\n\n".join(text_parts)
    
    except ImportError:
        logger.warning("PyMuPDF non installé, tentative avec pdfplumber")
        
        try:
            import pdfplumber
            
            with pdfplumber.open(file_path) as pdf:
                text_parts = []
                for page_num, page in enumerate(pdf.pages):
                    text = page.extract_text() or ""
                    text_parts.append(f"[Page {page_num + 1}]\n{text}")
                
                return "\n\n".join(text_parts)
        
        except ImportError:
            raise ImportError(
                "Aucune bibliothèque PDF installée. "
                "Installer PyMuPDF: pip install pymupdf"
            )
