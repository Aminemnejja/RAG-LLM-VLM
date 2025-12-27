"""
Configuration du système de logging pour le pipeline RAG.

Fournit un logging structuré avec rotation des fichiers,
support JSON pour l'analyse, et niveaux configurables par module.
"""

import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Optional
import json
from datetime import datetime


class JSONFormatter(logging.Formatter):
    """Formatter pour output JSON structuré."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        if hasattr(record, "extra_data"):
            log_data["extra"] = record.extra_data
        
        return json.dumps(log_data, ensure_ascii=False)


def setup_logging(
    level: str = "INFO",
    log_dir: Optional[Path] = None,
    json_format: bool = False,
    max_file_size_mb: int = 10,
    backup_count: int = 5
) -> None:
    """
    Configure le système de logging global.
    
    Args:
        level: Niveau de log (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Répertoire pour les fichiers de log
        json_format: Utiliser le format JSON pour les logs
        max_file_size_mb: Taille maximale des fichiers de log en MB
        backup_count: Nombre de fichiers de backup à conserver
    
    Example:
        >>> setup_logging(level="DEBUG", json_format=True)
    """
    # Créer le logger racine
    root_logger = logging.getLogger("rag_pipeline")
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Nettoyer les handlers existants
    root_logger.handlers.clear()
    
    # Format du log
    if json_format:
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
    
    # Handler console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.DEBUG)
    root_logger.addHandler(console_handler)
    
    # Handler fichier (si log_dir spécifié)
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / "rag_pipeline.log"
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_file_size_mb * 1024 * 1024,
            backupCount=backup_count,
            encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        root_logger.addHandler(file_handler)
        
        # Fichier séparé pour les erreurs
        error_file = log_dir / "rag_pipeline_errors.log"
        error_handler = RotatingFileHandler(
            error_file,
            maxBytes=max_file_size_mb * 1024 * 1024,
            backupCount=backup_count,
            encoding="utf-8"
        )
        error_handler.setFormatter(formatter)
        error_handler.setLevel(logging.ERROR)
        root_logger.addHandler(error_handler)
    
    root_logger.info("Logging configuré avec succès", extra={"level": level})


def get_logger(name: str) -> logging.Logger:
    """
    Retourne un logger configuré pour un module spécifique.
    
    Args:
        name: Nom du module (ex: "embedding_generator")
    
    Returns:
        Logger configuré
    
    Example:
        >>> logger = get_logger("embedding_generator")
        >>> logger.info("Embeddings générés", extra={"count": 100})
    """
    return logging.getLogger(f"rag_pipeline.{name}")


class LoggerAdapter(logging.LoggerAdapter):
    """Adapter pour ajouter des données contextuelles aux logs."""
    
    def process(self, msg, kwargs):
        extra = kwargs.get("extra", {})
        extra.update(self.extra)
        kwargs["extra"] = extra
        return msg, kwargs
