"""
Script d'indexation des documents médicaux.

Charge les documents depuis un répertoire, les découpe en chunks,
génère les embeddings, et construit le vector store FAISS.

Usage:
    python -m rag_pipeline.scripts.index_documents --input ./documents --output ./index
"""

import argparse
from pathlib import Path
from typing import List, Optional
import sys

# Ajouter le parent au path pour les imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rag_pipeline.config.settings import get_settings
from rag_pipeline.utils.logging_config import setup_logging, get_logger
from rag_pipeline.utils.chunking import DocumentChunker, Document, load_document_from_file
from rag_pipeline.embeddings.embedding_generator import EmbeddingGenerator
from rag_pipeline.vectorstore.faiss_store import FAISSVectorStore

logger = get_logger("index_documents")


def find_documents(input_dir: Path, extensions: List[str]) -> List[Path]:
    """
    Trouve tous les documents dans un répertoire.
    
    Args:
        input_dir: Répertoire d'entrée
        extensions: Extensions de fichiers à traiter
    
    Returns:
        Liste des chemins de fichiers
    """
    files = []
    for ext in extensions:
        files.extend(input_dir.glob(f"**/*.{ext}"))
    return sorted(files)


def index_documents(
    input_dir: Path,
    output_dir: Path,
    extensions: Optional[List[str]] = None,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    batch_size: int = 32
) -> None:
    """
    Indexe les documents d'un répertoire.
    
    Args:
        input_dir: Répertoire contenant les documents
        output_dir: Répertoire de sortie pour l'index
        extensions: Extensions de fichiers à traiter
        chunk_size: Taille des chunks
        chunk_overlap: Chevauchement entre chunks
        batch_size: Taille des batches pour les embeddings
    """
    if extensions is None:
        extensions = ["txt", "md", "pdf"]
    
    logger.info(f"Indexation des documents depuis: {input_dir}")
    logger.info(f"Extensions: {extensions}")
    
    # Trouver les documents
    doc_files = find_documents(input_dir, extensions)
    
    if not doc_files:
        logger.warning("Aucun document trouvé")
        return
    
    logger.info(f"{len(doc_files)} fichiers trouvés")
    
    # Charger et chunker les documents
    chunker = DocumentChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    all_chunks = []
    
    try:
        from tqdm import tqdm
        doc_iterator = tqdm(doc_files, desc="Chargement")
    except ImportError:
        doc_iterator = doc_files
    
    for file_path in doc_iterator:
        try:
            doc = load_document_from_file(file_path)
            chunks = chunker.chunk_document(doc)
            all_chunks.extend(chunks)
            logger.debug(f"Chargé: {file_path.name} ({len(chunks)} chunks)")
        except Exception as e:
            logger.error(f"Erreur pour {file_path}: {e}")
            continue
    
    logger.info(f"Total: {len(all_chunks)} chunks")
    
    if not all_chunks:
        logger.warning("Aucun chunk généré")
        return
    
    # Générer les embeddings
    logger.info("Génération des embeddings...")
    
    embedding_generator = EmbeddingGenerator()
    
    contents = [chunk.content for chunk in all_chunks]
    embeddings = embedding_generator.embed_texts(contents, show_progress=True)
    
    logger.info(f"Embeddings générés: shape={embeddings.shape}")
    
    # Créer le vector store
    logger.info("Construction du vector store...")
    
    vector_store = FAISSVectorStore(dimension=embeddings.shape[1])
    
    ids = [chunk.chunk_id for chunk in all_chunks]
    metadata_list = [chunk.metadata for chunk in all_chunks]
    
    vector_store.add_documents(
        contents=contents,
        embeddings=embeddings,
        ids=ids,
        metadata_list=metadata_list
    )
    
    # Sauvegarder
    output_dir.mkdir(parents=True, exist_ok=True)
    vector_store.save(output_dir)
    
    logger.info(f"Index sauvegardé: {output_dir}")
    logger.info(f"Documents indexés: {vector_store.count}")


def main():
    """Point d'entrée du script."""
    parser = argparse.ArgumentParser(
        description="Indexe les documents médicaux pour le RAG pipeline"
    )
    
    parser.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="Répertoire contenant les documents à indexer"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=Path,
        required=True,
        help="Répertoire de sortie pour l'index FAISS"
    )
    
    parser.add_argument(
        "--extensions", "-e",
        nargs="+",
        default=["txt", "md", "pdf"],
        help="Extensions de fichiers à traiter (défaut: txt md pdf)"
    )
    
    parser.add_argument(
        "--chunk-size", "-c",
        type=int,
        default=None,
        help="Taille des chunks en caractères"
    )
    
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=None,
        help="Chevauchement entre chunks"
    )
    
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=32,
        help="Taille des batches pour les embeddings"
    )
    
    parser.add_argument(
        "--log-level", "-l",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Niveau de log"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level=args.log_level)
    
    # Vérifier le répertoire d'entrée
    if not args.input.exists():
        logger.error(f"Répertoire d'entrée non trouvé: {args.input}")
        sys.exit(1)
    
    # Lancer l'indexation
    try:
        index_documents(
            input_dir=args.input,
            output_dir=args.output,
            extensions=args.extensions,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            batch_size=args.batch_size
        )
        logger.info("Indexation terminée avec succès")
    except Exception as e:
        logger.error(f"Erreur lors de l'indexation: {e}")
        raise


if __name__ == "__main__":
    main()
