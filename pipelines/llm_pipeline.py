"""
Pipeline LLM pour diagnostic dermatologique avec features OpenCV.

Combine l'extraction de features visuelles avec RAG et LLM
pour générer des diagnostics médicaux.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import hashlib
import json
import base64

from ..config.settings import get_settings
from ..retriever.rag_retriever import RAGRetriever, RetrievalResult
from ..utils.logging_config import get_logger

logger = get_logger("llm_pipeline")


@dataclass
class DiagnosticResult:
    """
    Résultat de diagnostic médical structuré.
    
    Attributes:
        query: Question posée
        diagnosis: Diagnostic principal proposé
        confidence: Score de confiance (0-1)
        differential_diagnoses: Diagnostics différentiels
        features_extracted: Features extraites de l'image
        rag_context: Contexte RAG utilisé
        recommendations: Recommandations
        sources: Sources médicales utilisées
        raw_response: Réponse brute du LLM
    """
    query: str
    diagnosis: str
    confidence: float = 0.0
    differential_diagnoses: List[str] = field(default_factory=list)
    features_extracted: Dict[str, Any] = field(default_factory=dict)
    rag_context: str = ""
    recommendations: List[str] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    raw_response: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "diagnosis": self.diagnosis,
            "confidence": self.confidence,
            "differential_diagnoses": self.differential_diagnoses,
            "features_extracted": self.features_extracted,
            "recommendations": self.recommendations,
            "sources": self.sources,
            "metadata": self.metadata
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


class FeatureExtractor:
    """
    Extracteur de features visuelles avec OpenCV.
    
    Extrait des caractéristiques pertinentes pour le diagnostic
    dermatologique: couleur, texture, forme, asymétrie, etc.
    """
    
    def __init__(self, cache_enabled: bool = True):
        """
        Initialise l'extracteur de features.
        
        Args:
            cache_enabled: Activer le cache des features
        """
        settings = get_settings()
        self.cache_enabled = cache_enabled and settings.cache.features_cache
        self.cache_dir = settings.cache.cache_dir / "features"
        
        logger.info("FeatureExtractor initialisé")
    
    def extract(self, image_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Extrait les features d'une image.
        
        Args:
            image_path: Chemin vers l'image
        
        Returns:
            Dictionnaire des features extraites
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image non trouvée: {image_path}")
        
        # Vérifier le cache
        if self.cache_enabled:
            cached = self._load_from_cache(image_path)
            if cached:
                logger.debug("Features chargées du cache")
                return cached
        
        logger.info(f"Extraction des features: {image_path.name}")
        
        try:
            import cv2
            import numpy as np
        except ImportError:
            raise ImportError("Installez OpenCV: pip install opencv-python")
        
        # Charger l'image
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Impossible de charger l'image: {image_path}")
        
        features = {}
        
        # Caractéristiques de couleur
        features["color"] = self._extract_color_features(img)
        
        # Caractéristiques de texture
        features["texture"] = self._extract_texture_features(img)
        
        # Caractéristiques de forme
        features["shape"] = self._extract_shape_features(img)
        
        # Métadonnées de l'image
        features["image_info"] = {
            "width": img.shape[1],
            "height": img.shape[0],
            "channels": img.shape[2] if len(img.shape) > 2 else 1,
            "filename": image_path.name
        }
        
        # Sauvegarder dans le cache
        if self.cache_enabled:
            self._save_to_cache(image_path, features)
        
        logger.debug(f"Features extraites: {list(features.keys())}")
        return features
    
    def _extract_color_features(self, img) -> Dict[str, Any]:
        """Extrait les caractéristiques de couleur."""
        import cv2
        import numpy as np
        
        # Convertir en différents espaces de couleur
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        
        # Statistiques HSV
        h_mean, s_mean, v_mean = np.mean(hsv, axis=(0, 1))
        h_std, s_std, v_std = np.std(hsv, axis=(0, 1))
        
        # Statistiques LAB
        l_mean, a_mean, b_mean = np.mean(lab, axis=(0, 1))
        
        # Histogramme de couleur dominant
        hist_h = cv2.calcHist([hsv], [0], None, [12], [0, 180])
        dominant_hue_idx = np.argmax(hist_h)
        
        # Mapping des teintes
        hue_names = ["rouge", "orange", "jaune", "vert-jaune", "vert", "cyan",
                     "bleu", "violet", "magenta", "rose", "rouge-rose", "rouge"]
        dominant_color = hue_names[dominant_hue_idx]
        
        return {
            "hsv_mean": [float(h_mean), float(s_mean), float(v_mean)],
            "hsv_std": [float(h_std), float(s_std), float(v_std)],
            "lab_mean": [float(l_mean), float(a_mean), float(b_mean)],
            "dominant_color": dominant_color,
            "saturation_level": "élevée" if s_mean > 100 else "moyenne" if s_mean > 50 else "faible",
            "brightness_level": "claire" if v_mean > 170 else "moyenne" if v_mean > 85 else "sombre"
        }
    
    def _extract_texture_features(self, img) -> Dict[str, Any]:
        """Extrait les caractéristiques de texture."""
        import cv2
        import numpy as np
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Laplacien pour la variance (netteté)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Gradient (Sobel)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        # Entropie locale (mesure de complexité)
        entropy = self._compute_entropy(gray)
        
        return {
            "sharpness": float(laplacian_var),
            "gradient_mean": float(np.mean(gradient_magnitude)),
            "gradient_std": float(np.std(gradient_magnitude)),
            "entropy": float(entropy),
            "texture_type": "lisse" if laplacian_var < 100 else "rugueuse" if laplacian_var < 500 else "très texturée"
        }
    
    def _extract_shape_features(self, img) -> Dict[str, Any]:
        """Extrait les caractéristiques de forme."""
        import cv2
        import numpy as np
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Seuillage adaptatif pour trouver les contours
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {
                "area": 0,
                "perimeter": 0,
                "circularity": 0,
                "asymmetry_index": 0,
                "border_regularity": "non déterminée"
            }
        
        # Prendre le plus grand contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        
        # Circularité
        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
        
        # Asymétrie (via moments)
        moments = cv2.moments(largest_contour)
        if moments["m00"] > 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
            
            # Calculer l'asymétrie horizontale/verticale
            rect = cv2.boundingRect(largest_contour)
            asymmetry = abs(cx - (rect[0] + rect[2]/2)) / (rect[2]/2) if rect[2] > 0 else 0
        else:
            asymmetry = 0
        
        # Classification de la régularité des bords
        hull = cv2.convexHull(largest_contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        border_regularity = "régulier" if solidity > 0.9 else "irrégulier" if solidity > 0.7 else "très irrégulier"
        
        return {
            "area": float(area),
            "perimeter": float(perimeter),
            "circularity": float(circularity),
            "asymmetry_index": float(asymmetry),
            "solidity": float(solidity),
            "border_regularity": border_regularity,
            "shape_description": "circulaire" if circularity > 0.8 else "ovale" if circularity > 0.5 else "irrégulière"
        }
    
    def _compute_entropy(self, gray) -> float:
        """Calcule l'entropie de l'image."""
        import numpy as np
        
        hist, _ = np.histogram(gray.flatten(), bins=256, range=(0, 256))
        hist = hist / hist.sum()
        hist = hist[hist > 0]
        return -np.sum(hist * np.log2(hist))
    
    def _get_cache_key(self, image_path: Path) -> str:
        """Génère une clé de cache unique."""
        stat = image_path.stat()
        content = f"{image_path}_{stat.st_mtime}_{stat.st_size}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _load_from_cache(self, image_path: Path) -> Optional[Dict[str, Any]]:
        """Charge les features depuis le cache."""
        cache_key = self._get_cache_key(image_path)
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, "r") as f:
                    return json.load(f)
            except Exception:
                pass
        return None
    
    def _save_to_cache(self, image_path: Path, features: Dict[str, Any]) -> None:
        """Sauvegarde les features dans le cache."""
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache_key = self._get_cache_key(image_path)
            cache_file = self.cache_dir / f"{cache_key}.json"
            
            with open(cache_file, "w") as f:
                json.dump(features, f)
        except Exception as e:
            logger.warning(f"Erreur sauvegarde cache: {e}")


class LLMPipeline:
    """
    Pipeline complet: Features OpenCV + RAG + LLM.
    
    Extrait les features visuelles de l'image, récupère le contexte
    médical pertinent, et génère un diagnostic via LLM.
    
    Example:
        >>> pipeline = LLMPipeline.from_index("./data/index")
        >>> result = pipeline.diagnose("./lesion.jpg", "Quel est ce type de lésion?")
        >>> print(result.diagnosis)
    """
    
    def __init__(
        self,
        retriever: RAGRetriever,
        feature_extractor: Optional[FeatureExtractor] = None,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None
    ):
        """
        Initialise le pipeline LLM.
        
        Args:
            retriever: RAG Retriever configuré
            feature_extractor: Extracteur de features (optionnel)
            model_name: Nom du modèle LLM
            temperature: Température de génération
        """
        settings = get_settings()
        
        self.retriever = retriever
        self.feature_extractor = feature_extractor or FeatureExtractor()
        self.model_name = model_name or settings.llm.model_name
        self.temperature = temperature if temperature is not None else settings.llm.temperature
        self.max_tokens = settings.llm.max_tokens
        self.system_prompt = settings.llm.system_prompt
        self.api_key = settings.llm.api_key
        self.api_base = settings.llm.api_base
        
        # Client LLM (lazy loading)
        self._client = None
        
        logger.info(f"LLMPipeline initialisé avec {self.model_name}")
    
    @classmethod
    def from_index(
        cls,
        index_path: str,
        model_name: Optional[str] = None
    ) -> "LLMPipeline":
        """
        Crée un pipeline à partir d'un index existant.
        
        Args:
            index_path: Chemin vers l'index RAG
            model_name: Nom du modèle LLM
        
        Returns:
            Instance de LLMPipeline
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
            logger.info("Client OpenAI initialisé")
            
        except ImportError:
            raise ImportError("Installez openai: pip install openai")
    
    def diagnose(
        self,
        image_path: Union[str, Path],
        query: str,
        use_rag: bool = True,
        use_features: bool = True
    ) -> DiagnosticResult:
        """
        Génère un diagnostic pour une image et une question.
        
        Args:
            image_path: Chemin vers l'image de la lésion
            query: Question de l'utilisateur
            use_rag: Utiliser le contexte RAG
            use_features: Extraire les features de l'image
        
        Returns:
            DiagnosticResult avec le diagnostic structuré
        """
        logger.info(f"Diagnostic demandé: {query[:50]}...")
        
        image_path = Path(image_path)
        
        # Extraire les features
        features = {}
        if use_features:
            try:
                features = self.feature_extractor.extract(image_path)
            except Exception as e:
                logger.warning(f"Erreur extraction features: {e}")
        
        # Récupérer le contexte RAG
        rag_result = None
        rag_context = ""
        if use_rag:
            # Enrichir la requête avec les features pour meilleure recherche
            enriched_query = self._enrich_query_with_features(query, features)
            rag_result = self.retriever.retrieve(enriched_query)
            rag_context = rag_result.context
        
        # Générer le prompt
        prompt = self._build_prompt(query, features, rag_context)
        
        # Appeler le LLM
        response = self._call_llm(prompt)
        
        # Parser la réponse
        result = self._parse_response(response, query, features, rag_context)
        
        # Ajouter les sources
        if rag_result:
            result.sources = [
                doc.metadata.get("source", doc.id) 
                for doc in rag_result.documents
            ]
        
        logger.info(f"Diagnostic généré: {result.diagnosis[:100]}...")
        return result
    
    def _enrich_query_with_features(
        self,
        query: str,
        features: Dict[str, Any]
    ) -> str:
        """Enrichit la requête avec les features extraites."""
        if not features:
            return query
        
        enrichments = []
        
        if "color" in features:
            color = features["color"]
            enrichments.append(f"couleur {color.get('dominant_color', '')}")
            enrichments.append(color.get("saturation_level", ""))
        
        if "shape" in features:
            shape = features["shape"]
            enrichments.append(shape.get("border_regularity", ""))
            enrichments.append(f"forme {shape.get('shape_description', '')}")
        
        if "texture" in features:
            texture = features["texture"]
            enrichments.append(f"texture {texture.get('texture_type', '')}")
        
        enriched = f"{query} (caractéristiques: {', '.join(filter(None, enrichments))})"
        return enriched
    
    def _build_prompt(
        self,
        query: str,
        features: Dict[str, Any],
        rag_context: str
    ) -> str:
        """Construit le prompt pour le LLM."""
        prompt_parts = []
        
        # Ajouter les features
        if features:
            features_text = self._format_features(features)
            prompt_parts.append(f"## Caractéristiques extraites de l'image\n{features_text}")
        
        # Ajouter le contexte RAG
        if rag_context:
            prompt_parts.append(f"## Contexte médical de référence\n{rag_context}")
        
        # Ajouter la question
        prompt_parts.append(f"## Question du patient\n{query}")
        
        # Instructions de réponse
        prompt_parts.append("""
## Instructions
Basé sur les caractéristiques de l'image et le contexte médical:
1. Propose un diagnostic principal avec un niveau de confiance
2. Liste les diagnostics différentiels possibles
3. Donne des recommandations (examens complémentaires, consultation)
4. Cite tes sources si disponibles

Rappel: Ceci est une aide à la décision. Un examen médical professionnel est toujours nécessaire.
""")
        
        return "\n\n".join(prompt_parts)
    
    def _format_features(self, features: Dict[str, Any]) -> str:
        """Formate les features en texte lisible."""
        lines = []
        
        if "color" in features:
            c = features["color"]
            lines.append(f"- Couleur dominante: {c.get('dominant_color', 'N/A')}")
            lines.append(f"- Saturation: {c.get('saturation_level', 'N/A')}")
            lines.append(f"- Luminosité: {c.get('brightness_level', 'N/A')}")
        
        if "shape" in features:
            s = features["shape"]
            lines.append(f"- Forme: {s.get('shape_description', 'N/A')}")
            lines.append(f"- Bords: {s.get('border_regularity', 'N/A')}")
            lines.append(f"- Indice d'asymétrie: {s.get('asymmetry_index', 0):.2f}")
        
        if "texture" in features:
            t = features["texture"]
            lines.append(f"- Texture: {t.get('texture_type', 'N/A')}")
        
        return "\n".join(lines)
    
    def _call_llm(self, prompt: str) -> str:
        """Appelle le LLM avec le prompt."""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Erreur appel LLM: {e}")
            raise
    
    def _parse_response(
        self,
        response: str,
        query: str,
        features: Dict[str, Any],
        rag_context: str
    ) -> DiagnosticResult:
        """Parse la réponse du LLM en résultat structuré."""
        # Extraction simple (peut être amélioré avec un parser plus robuste)
        diagnosis = response.split("\n")[0] if response else "Non déterminé"
        
        # Rechercher les diagnostics différentiels
        differential = []
        if "différentiel" in response.lower():
            lines = response.split("\n")
            capture = False
            for line in lines:
                if "différentiel" in line.lower():
                    capture = True
                    continue
                if capture and line.strip().startswith(("-", "*", "•", "1", "2", "3")):
                    differential.append(line.strip().lstrip("-*•0123456789. "))
                elif capture and not line.strip():
                    break
        
        # Rechercher les recommandations
        recommendations = []
        if "recommand" in response.lower():
            lines = response.split("\n")
            capture = False
            for line in lines:
                if "recommand" in line.lower():
                    capture = True
                    continue
                if capture and line.strip().startswith(("-", "*", "•", "1", "2", "3")):
                    recommendations.append(line.strip().lstrip("-*•0123456789. "))
                elif capture and not line.strip():
                    break
        
        return DiagnosticResult(
            query=query,
            diagnosis=diagnosis,
            confidence=0.0,  # À améliorer avec extraction du score de confiance
            differential_diagnoses=differential[:5],
            features_extracted=features,
            rag_context=rag_context,
            recommendations=recommendations[:5],
            raw_response=response
        )
