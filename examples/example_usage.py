"""
Exemples d'utilisation du pipeline RAG médical.

Ce script montre comment utiliser les différentes fonctionnalités
du pipeline pour le diagnostic dermatologique.
"""

from pathlib import Path
import sys

# Ajouter le chemin du projet
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def example_index_documents():
    """
    Exemple: Indexer des documents médicaux.
    
    Usage:
        python -m rag_pipeline.scripts.index_documents \
            --input ./data/documents \
            --output ./data/index
    """
    from rag_pipeline.utils.chunking import DocumentChunker, Document
    from rag_pipeline.embeddings.embedding_generator import EmbeddingGenerator
    from rag_pipeline.vectorstore.faiss_store import FAISSVectorStore
    
    print("=== Exemple: Indexation de documents ===\n")
    
    # Créer quelques documents médicaux de test
    documents = [
        Document(
            content="""
            Le mélanome est une forme de cancer de la peau qui se développe 
            à partir des mélanocytes. Les signes d'alerte incluent la règle ABCDE:
            - Asymétrie de la lésion
            - Bords irréguliers
            - Couleur non homogène
            - Diamètre supérieur à 6mm
            - Évolution rapide
            """,
            source="melanoma_guide.txt",
            metadata={"type": "oncology", "chapter": "melanoma"}
        ),
        Document(
            content="""
            Le psoriasis est une maladie inflammatoire chronique de la peau.
            Il se manifeste par des plaques rouges recouvertes de squames blanches.
            Les zones les plus touchées sont les coudes, les genoux et le cuir chevelu.
            Le traitement comprend des topiques, la photothérapie et les biologiques.
            """,
            source="psoriasis_manual.txt",
            metadata={"type": "dermatology", "chapter": "psoriasis"}
        ),
        Document(
            content="""
            L'eczéma atopique est caractérisé par une peau sèche, des démangeaisons
            intenses et des lésions érythémateuses. Il touche particulièrement 
            les plis de flexion chez l'enfant. Le traitement repose sur l'hydratation
            et les dermocorticoïdes en cas de poussée.
            """,
            source="eczema_overview.txt",
            metadata={"type": "dermatology", "chapter": "eczema"}
        ),
    ]
    
    # Chunker les documents
    chunker = DocumentChunker(chunk_size=300, chunk_overlap=30)
    all_chunks = []
    for doc in documents:
        chunks = chunker.chunk_document(doc)
        all_chunks.extend(chunks)
        print(f"  {doc.source}: {len(chunks)} chunks")
    
    print(f"\nTotal: {len(all_chunks)} chunks")
    
    # Générer les embeddings (simulé pour l'exemple)
    print("\nGénération des embeddings...")
    
    # Note: En production, utiliser le vrai EmbeddingGenerator
    import numpy as np
    fake_embeddings = np.random.randn(len(all_chunks), 768).astype(np.float32)
    fake_embeddings = fake_embeddings / np.linalg.norm(fake_embeddings, axis=1, keepdims=True)
    
    # Créer le vector store
    store = FAISSVectorStore(dimension=768)
    store.add_documents(
        contents=[c.content for c in all_chunks],
        embeddings=fake_embeddings,
        ids=[c.chunk_id for c in all_chunks],
        metadata_list=[c.metadata for c in all_chunks]
    )
    
    print(f"Vector store créé avec {store.count} documents")
    
    # Sauvegarder (optionnel)
    # store.save(Path("./data/index"))


def example_search():
    """
    Exemple: Recherche dans le vector store.
    """
    from rag_pipeline.vectorstore.faiss_store import FAISSVectorStore
    import numpy as np
    
    print("=== Exemple: Recherche vectorielle ===\n")
    
    # Créer un store de démonstration
    store = FAISSVectorStore(dimension=128)
    
    # Ajouter des documents
    docs = [
        "Melanoma treatment options include surgery and immunotherapy",
        "Psoriasis can be treated with topical corticosteroids",
        "Eczema requires moisturizing and avoiding triggers",
        "Skin cancer screening should be done annually",
        "Acne treatment includes retinoids and antibiotics",
    ]
    
    embeddings = np.random.randn(len(docs), 128).astype(np.float32)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    ids = [f"doc_{i}" for i in range(len(docs))]
    
    store.add_documents(docs, embeddings, ids)
    
    # Rechercher
    query_embedding = np.random.randn(128).astype(np.float32)
    query_embedding = query_embedding / np.linalg.norm(query_embedding)
    
    results = store.search(query_embedding, top_k=3)
    
    print("Résultats de recherche:")
    for i, result in enumerate(results, 1):
        print(f"  {i}. [{result.score:.4f}] {result.content[:60]}...")


def example_llm_pipeline():
    """
    Exemple: Utilisation du pipeline LLM complet.
    
    Note: Nécessite une clé API OpenAI configurée.
    """
    print("=== Exemple: Pipeline LLM ===\n")
    
    print("""
    # Usage avec le pipeline LLM:
    
    from rag_pipeline import query_llm, init_pipelines
    
    # Initialiser (une seule fois)
    init_pipelines(index_path="./data/index")
    
    # Diagnostic
    result = query_llm(
        image_path="./patient_lesion.jpg",
        query="Cette lésion est-elle suspecte de mélanome?"
    )
    
    print(f"Diagnostic: {result.diagnosis}")
    print(f"Différentiels: {result.differential_diagnoses}")
    print(f"Recommandations: {result.recommendations}")
    """)


def example_vlm_pipeline():
    """
    Exemple: Utilisation du pipeline VLM.
    
    Note: Nécessite une clé API OpenAI avec accès à GPT-4V.
    """
    print("=== Exemple: Pipeline VLM ===\n")
    
    print("""
    # Usage avec le pipeline VLM:
    
    from rag_pipeline import query_vlm, init_pipelines
    
    # Initialiser
    init_pipelines(index_path="./data/index")
    
    # Diagnostic avec analyse visuelle directe
    result = query_vlm(
        image_path="./patient_lesion.jpg",
        query="Analyse cette lésion cutanée"
    )
    
    print(f"Diagnostic: {result.diagnosis}")
    print(f"Confiance: {result.confidence}")
    print(f"Réponse complète: {result.raw_response}")
    """)


def example_evaluation():
    """
    Exemple: Évaluation avec ground truth.
    """
    print("=== Exemple: Évaluation du pipeline ===\n")
    
    print("""
    # Structure pour l'évaluation:
    
    ground_truth = [
        {
            "image": "./images/case_001.jpg",
            "query": "Quel est le diagnostic?",
            "expected_diagnosis": "Mélanome",
            "expected_category": "malignant"
        },
        # ... autres cas
    ]
    
    results = []
    for case in ground_truth:
        result = query_vlm(case["image"], case["query"])
        
        # Comparer avec le ground truth
        match = case["expected_diagnosis"].lower() in result.diagnosis.lower()
        results.append({
            "case_id": case["image"],
            "predicted": result.diagnosis,
            "expected": case["expected_diagnosis"],
            "match": match,
            "score": result.confidence
        })
    
    # Calculer les métriques
    accuracy = sum(r["match"] for r in results) / len(results)
    print(f"Accuracy: {accuracy:.2%}")
    """)


if __name__ == "__main__":
    print("=" * 60)
    print("RAG Pipeline Medical - Exemples d'utilisation")
    print("=" * 60)
    print()
    
    example_index_documents()
    print("\n" + "=" * 60 + "\n")
    
    example_search()
    print("\n" + "=" * 60 + "\n")
    
    example_llm_pipeline()
    print("\n" + "=" * 60 + "\n")
    
    example_vlm_pipeline()
    print("\n" + "=" * 60 + "\n")
    
    example_evaluation()
