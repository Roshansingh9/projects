import pandas as pd
import yaml
from tqdm import tqdm
import os

from sklearn.model_selection import train_test_split

from llm.groq_client import GroqClient
from pathway_pipeline.ingest import NovelIngestor
from pathway_pipeline.index import NovelIndexer
from retrieval.retrieve import EvidenceRetriever
from reasoning.debate import DebateOrchestrator
from scoring.scorer import BackstoryScorer


# --------------------------------------------------
# Utilities
# --------------------------------------------------

def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def normalize_label(label_str: str) -> int:
    """Convert string label to int for evaluation only."""
    return 1 if label_str.strip().lower() == "consistent" else 0


# --------------------------------------------------
# Main Inference
# --------------------------------------------------

def main():
    print("=" * 80)
    print("KHARAGPUR DATA SCIENCE HACKATHON 2026")
    print("TRACK A — SYSTEMS REASONING WITH NLP & GENAI")
    print("=" * 80)

    # -----------------------------
    # Load config
    # -----------------------------
    config = load_config()
    print("✓ Configuration loaded")

    # -----------------------------
    # Initialize LLM (FREE API)
    # -----------------------------
    llm_client = GroqClient()
    print("✓ LLM client initialized")

    # -----------------------------
    # Load train.csv and split 80/20
    # -----------------------------
    full_df = pd.read_csv("data/train.csv")
    print(f"✓ Loaded {len(full_df)} samples from train.csv")

    train_df, val_df = train_test_split(
        full_df,
        test_size=0.2,
        random_state=42,
        stratify=full_df["label"]
    )

    print(f"✓ Training split (unused): {len(train_df)}")
    print(f"✓ Validation split (inference): {len(val_df)}")

    # -----------------------------
    # PATHWAY PIPELINE
    # -----------------------------
    print("\n" + "=" * 80)
    print("PATHWAY PIPELINE — INGEST & INDEX NOVELS")
    print("=" * 80)

    indexer = NovelIndexer(config)
    index_cache = "pathway_index.pkl"

    if os.path.exists(index_cache) and indexer.load_index(index_cache):
        print("✓ Loaded cached Pathway index")
    else:
        print("⚙️ Building Pathway index from scratch...")
        ingestor = NovelIngestor(config)
        book_chunks = ingestor.ingest_books("data/Books/")
        indexer.build_index(book_chunks)
        indexer.save_index(index_cache)
        print("✓ Pathway index built and cached")

    # -----------------------------
    # Initialize components
    # -----------------------------
    retriever = EvidenceRetriever(indexer, config)
    debate_orchestrator = DebateOrchestrator(llm_client, retriever, config)
    scorer = BackstoryScorer(config)

    print("✓ Retrieval + Reasoning components ready")

    # -----------------------------
    # Inference on validation set
    # -----------------------------
    print("\n" + "=" * 80)
    print("RUNNING INFERENCE ON VALIDATION SPLIT")
    print("=" * 80)

    results = []

    for _, row in tqdm(val_df.iterrows(), total=len(val_df), desc="Inference"):
        sample_id = row["id"]
        book_name = row["book_name"]
        character = row["char"]
        backstory = row["content"]
        true_label = normalize_label(row["label"])

        try:
            deliberations = debate_orchestrator.deliberate_on_backstory(
                backstory=backstory,
                book_name=book_name,
                character=character
            )

            pred_label, rationale = scorer.compute_score(deliberations)

            results.append({
                "id": sample_id,
                "book_name": book_name,
                "character": character,
                "prediction": pred_label,
                "true_label": true_label,
                "rationale": rationale
            })

        except Exception as e:
            print(f"\n❌ Error on sample {sample_id}: {e}")
            results.append({
                "id": sample_id,
                "book_name": book_name,
                "character": character,
                "prediction": 0,
                "true_label": true_label,
                "rationale": f"Error during inference: {str(e)}"
            })

    # -----------------------------
    # Save results
    # -----------------------------
    results_df = pd.DataFrame(results)
    results_df.to_csv("results_val.csv", index=False)

    print("\n" + "=" * 80)
    print("INFERENCE COMPLETE")
    print("=" * 80)
    print("✓ Results saved to results_val.csv")

    # -----------------------------
    # Validation metrics (DEV ONLY)
    # -----------------------------
    accuracy = (results_df["prediction"] == results_df["true_label"]).mean()
    print(f"\n📊 Validation Accuracy: {accuracy:.4f}")

    # -----------------------------
    # LLM usage stats
    # -----------------------------
    llm_client.print_stats()

    print("\n✓ Done.")


if __name__ == "__main__":
    main()
