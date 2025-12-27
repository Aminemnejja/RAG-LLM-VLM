"""
Pipeline VLM pour diagnostic dermatologique avec analyse d'image directe.

Utilise un Vision Language Model (GPT-4V, LLaVA, etc.) pour analyser
directement l'image avec le contexte RAG.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import base64
import mimetypes

from ..config.settings import get_settings
from ..retriever.rag_retriever import RAGRetriever, RetrievalResult
from ..utils.logging_config import get_logger
from .llm_pipeline import DiagnosticResult

logger = get_logger("vlm_pipeline")


class VLMPipeline:
    """
    Pipeline VLM: Image directe + RAG + Vision Language Model.
    
    Analyse directement l'image médicale avec un VLM comme GPT-4V
    et enrichit l'analyse avec le contexte RAG.
    
    Example:
        >>> pipeline = VLMPipeline.from_index("./data/index")
        >>> result = pipeline.diagnose("./lesion.jpg", "Analyse cette lésion cutanée")
        >>> print(result.diagnosis)
    """
    
    def __init__(
        self,
        retriever: RAGRetriever,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        detail: Optional[str] = None
    ):
        """
        Initialise le pipeline VLM.
        
        Args:
            retriever: RAG Retriever configuré
            model_name: Nom du modèle VLM
            temperature: Température de génération
            detail: Niveau de détail pour l'image ('low', 'high', 'auto')
        """
        settings = get_settings()
        
        self.retriever = retriever
        self.model_name = model_name or settings.vlm.model_name
        self.temperature = temperature if temperature is not None else settings.vlm.temperature
        self.max_tokens = settings.vlm.max_tokens
        self.detail = detail or settings.vlm.detail
        self.max_image_size = settings.vlm.max_image_size
        self.system_prompt = settings.vlm.system_prompt
        self.api_key = settings.vlm.api_key
        self.api_base = settings.vlm.api_base
        
        # Client (lazy loading)
        self._client = None
        
        logger.info(f"VLMPipeline initialisé avec {self.model_name}")
    
    @classmethod
    def from_index(
        cls,
        index_path: str,
        model_name: Optional[str] = None
    ) -> "VLMPipeline":
        """
        Crée un pipeline à partir d'un index existant.
        
        Args:
            index_path: Chemin vers l'index RAG
            model_name: Nom du modèle VLM
        
        Returns:
            Instance de VLMPipeline
        """
        from ..reranker.cross_encoder import CrossEncoderReranker
        
        reranker = CrossEncoderReranker()
        retriever = RAGRetriever.from_index(index_path, reranker=reranker)
        
        return cls(retriever=retriever, model_name=model_name)
    
    @property
    def client(self):
        """Lazy loading du client OpenAI."""
        if self._client is None:
            self._init_client()
        return self._client
    
    def _init_client(self) -> None:
        """Initialise le client OpenAI."""
        try:
            from openai import OpenAI
            
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.api_base
            )
            logger.info("Client OpenAI initialisé pour VLM")
            
        except ImportError:
            raise ImportError("Installez openai: pip install openai")
    
    def diagnose(
        self,
        image_path: Union[str, Path],
        query: str,
        use_rag: bool = True
    ) -> DiagnosticResult:
        """
        Génère un diagnostic en analysant directement l'image.
        
        Args:
            image_path: Chemin vers l'image de la lésion
            query: Question de l'utilisateur
            use_rag: Utiliser le contexte RAG
        
        Returns:
            DiagnosticResult avec le diagnostic structuré
        """
        logger.info(f"Diagnostic VLM demandé: {query[:50]}...")
        
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image non trouvée: {image_path}")
        
        # Préparer l'image en base64
        image_data = self._prepare_image(image_path)
        
        # Récupérer le contexte RAG
        rag_result = None
        rag_context = ""
        if use_rag:
            rag_result = self.retriever.retrieve(query)
            rag_context = rag_result.context
        
        # Construire le prompt avec contexte
        prompt = self._build_prompt(query, rag_context)
        
        # Appeler le VLM
        response = self._call_vlm(prompt, image_data)
        
        # Parser la réponse
        result = self._parse_response(response, query, rag_context)
        
        # Ajouter les sources
        if rag_result:
            result.sources = [
                doc.metadata.get("source", doc.id) 
                for doc in rag_result.documents
            ]
        
        result.metadata["image_path"] = str(image_path)
        result.metadata["model"] = self.model_name
        
        logger.info(f"Diagnostic VLM généré: {result.diagnosis[:100]}...")
        return result
    
    def _prepare_image(self, image_path: Path) -> Dict[str, str]:
        """
        Prépare l'image pour l'envoi au VLM.
        
        Args:
            image_path: Chemin vers l'image
        
        Returns:
            Dictionnaire avec type et données base64
        """
        # Déterminer le type MIME
        mime_type, _ = mimetypes.guess_type(str(image_path))
        if mime_type is None:
            mime_type = "image/jpeg"
        
        # Optionnel: redimensionner si trop grande
        image_bytes = self._resize_if_needed(image_path)
        
        # Encoder en base64
        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        
        return {
            "type": "base64",
            "media_type": mime_type,
            "data": base64_image
        }
    
    def _resize_if_needed(self, image_path: Path) -> bytes:
        """Redimensionne l'image si elle dépasse la taille maximale."""
        try:
            import cv2
            import numpy as np
            
            img = cv2.imread(str(image_path))
            if img is None:
                # Fallback: lire directement les bytes
                return image_path.read_bytes()
            
            h, w = img.shape[:2]
            max_size = self.max_image_size
            
            if max(h, w) > max_size:
                scale = max_size / max(h, w)
                new_w = int(w * scale)
                new_h = int(h * scale)
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                logger.debug(f"Image redimensionnée: {w}x{h} -> {new_w}x{new_h}")
            
            # Encoder en JPEG
            _, buffer = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            return buffer.tobytes()
            
        except ImportError:
            # Si OpenCV n'est pas disponible, lire directement
            return image_path.read_bytes()
    
    def _build_prompt(self, query: str, rag_context: str) -> str:
        """Construit le prompt pour le VLM."""
        prompt_parts = []
        
        # Ajouter le contexte RAG si disponible
        if rag_context:
            prompt_parts.append(f"""## Contexte médical de référence
Les informations suivantes proviennent de la littérature dermatologique et peuvent aider à l'analyse:

{rag_context}""")
        
        # Ajouter la question
        prompt_parts.append(f"""## Question
{query}""")
        
        # Instructions de réponse
        prompt_parts.append("""## Instructions
Analyse l'image de la lésion cutanée fournie et réponds à la question en suivant ce format:

1. **Description visuelle**: Décris les caractéristiques observées (couleur, forme, bords, taille, texture)
2. **Diagnostic principal**: Propose le diagnostic le plus probable avec ton niveau de confiance
3. **Diagnostics différentiels**: Liste 2-3 autres diagnostics possibles
4. **Recommandations**: Suggère les prochaines étapes (examens, consultation spécialisée)

Important: Rappelle que cette analyse est une aide à la décision et ne remplace pas un examen médical.""")
        
        return "\n\n".join(prompt_parts)
    
    def _call_vlm(self, prompt: str, image_data: Dict[str, str]) -> str:
        """
        Appelle le VLM avec l'image et le prompt.
        
        Args:
            prompt: Texte du prompt
            image_data: Données de l'image encodée
        
        Returns:
            Réponse du modèle
        """
        try:
            # Construire le message avec l'image
            content = [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{image_data['media_type']};base64,{image_data['data']}",
                        "detail": self.detail
                    }
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": content}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Erreur appel VLM: {e}")
            raise
    
    def _parse_response(
        self,
        response: str,
        query: str,
        rag_context: str
    ) -> DiagnosticResult:
        """Parse la réponse du VLM en résultat structuré."""
        # Extraction du diagnostic principal
        diagnosis = ""
        if "**diagnostic principal**" in response.lower():
            parts = response.lower().split("**diagnostic principal**")
            if len(parts) > 1:
                diagnosis_section = parts[1].split("**")[0]
                diagnosis = diagnosis_section.strip().strip(":").strip()
        
        if not diagnosis:
            lines = response.split("\n")
            diagnosis = lines[0] if lines else "Non déterminé"
        
        # Extraction des diagnostics différentiels
        differential = []
        if "différentiel" in response.lower():
            lines = response.split("\n")
            capture = False
            for line in lines:
                if "différentiel" in line.lower():
                    capture = True
                    continue
                if capture and line.strip().startswith(("-", "*", "•", "1", "2", "3")):
                    diag = line.strip().lstrip("-*•0123456789. ")
                    if diag:
                        differential.append(diag)
                elif capture and line.startswith("**"):
                    break
        
        # Extraction des recommandations
        recommendations = []
        if "recommand" in response.lower():
            lines = response.split("\n")
            capture = False
            for line in lines:
                if "recommand" in line.lower():
                    capture = True
                    continue
                if capture and line.strip().startswith(("-", "*", "•", "1", "2", "3")):
                    rec = line.strip().lstrip("-*•0123456789. ")
                    if rec:
                        recommendations.append(rec)
                elif capture and line.startswith("**"):
                    break
        
        # Extraction du niveau de confiance si mentionné
        confidence = 0.0
        if "confiance" in response.lower():
            import re
            match = re.search(r'(\d+)\s*%', response)
            if match:
                confidence = int(match.group(1)) / 100.0
        
        return DiagnosticResult(
            query=query,
            diagnosis=diagnosis,
            confidence=confidence,
            differential_diagnoses=differential[:5],
            features_extracted={},  # VLM n'utilise pas d'extraction de features séparée
            rag_context=rag_context,
            recommendations=recommendations[:5],
            raw_response=response
        )
    
    def analyze_batch(
        self,
        image_paths: List[Union[str, Path]],
        query: str,
        use_rag: bool = True
    ) -> List[DiagnosticResult]:
        """
        Analyse plusieurs images avec la même question.
        
        Args:
            image_paths: Liste des chemins d'images
            query: Question commune
            use_rag: Utiliser le contexte RAG
        
        Returns:
            Liste de DiagnosticResult
        """
        results = []
        for image_path in image_paths:
            try:
                result = self.diagnose(image_path, query, use_rag)
                results.append(result)
            except Exception as e:
                logger.error(f"Erreur pour {image_path}: {e}")
                results.append(DiagnosticResult(
                    query=query,
                    diagnosis=f"Erreur: {str(e)}",
                    metadata={"image_path": str(image_path), "error": True}
                ))
        return results
    
    def compare_lesions(
        self,
        image_path_1: Union[str, Path],
        image_path_2: Union[str, Path],
        comparison_query: Optional[str] = None
    ) -> DiagnosticResult:
        """
        Compare deux images de lésions.
        
        Args:
            image_path_1: Première image
            image_path_2: Deuxième image
            comparison_query: Question de comparaison
        
        Returns:
            DiagnosticResult avec l'analyse comparative
        """
        if comparison_query is None:
            comparison_query = "Compare ces deux lésions cutanées et analyse leurs différences."
        
        # Préparer les deux images
        image_data_1 = self._prepare_image(Path(image_path_1))
        image_data_2 = self._prepare_image(Path(image_path_2))
        
        # Construire le prompt de comparaison
        prompt = f"""## Comparaison de lésions cutanées

{comparison_query}

Analyse les deux images fournies et:
1. Décris les caractéristiques de chaque lésion
2. Identifie les similitudes et différences
3. Propose un diagnostic pour chacune
4. Recommande des actions si nécessaire"""
        
        try:
            content = [
                {
                    "type": "text",
                    "text": "Image 1:"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{image_data_1['media_type']};base64,{image_data_1['data']}",
                        "detail": self.detail
                    }
                },
                {
                    "type": "text",
                    "text": "Image 2:"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{image_data_2['media_type']};base64,{image_data_2['data']}",
                        "detail": self.detail
                    }
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": content}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            raw_response = response.choices[0].message.content
            
            return DiagnosticResult(
                query=comparison_query,
                diagnosis=raw_response.split("\n")[0] if raw_response else "Comparaison effectuée",
                raw_response=raw_response,
                metadata={
                    "image_1": str(image_path_1),
                    "image_2": str(image_path_2),
                    "comparison": True
                }
            )
            
        except Exception as e:
            logger.error(f"Erreur comparaison: {e}")
            raise
