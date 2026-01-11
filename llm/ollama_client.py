import requests
import time
from typing import Optional, Dict, Any
import yaml


class OllamaClient:
    """
    Local LLM client using Ollama.

    Optimized for:
    - Windows
    - CPU inference
    - Multi-agent pipelines
    - Long-running hackathon workflows
    """

    def __init__(self, config_path: str = "config.yaml"):
        # Load config
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Ollama endpoint
        self.base_url = self.config["llm"].get(
            "ollama_url", "http://localhost:11434"
        )

        # Model mapping
        self.models = self.config["llm"]["models"]

        # Generation params
        self.temperature = self.config["llm"].get("temperature", 0.1)
        self.max_tokens = self.config["llm"].get("max_tokens", 256)
        self.max_retries = self.config["llm"].get("max_retries", 1)

        # Stats
        self.call_count = 0
        self.call_history = {m: 0 for m in set(self.models.values())}

        # Test connection
        self._test_connection()

        # Warm up models (CRITICAL on Windows)
        self._warmup_models()

    # ------------------------------------------------------------------
    # Connection check
    # ------------------------------------------------------------------
    def _test_connection(self):
        try:
            response = requests.get(
                f"{self.base_url}/api/tags", timeout=5
            )
            response.raise_for_status()

            models = response.json().get("models", [])
            print(f"✓ Connected to Ollama ({len(models)} models available)")

            available = [m["name"] for m in models]
            for task, model_name in self.models.items():
                if model_name not in available:
                    print(
                        f"⚠️  Model '{model_name}' not found. Pull it with:\n"
                        f"   ollama pull {model_name}"
                    )

        except Exception:
            print("❌ Cannot connect to Ollama.")
            print("   Make sure Ollama is running:")
            print("   ollama pull llama3.1:8b")
            raise

    # ------------------------------------------------------------------
    # Warm-up (prevents first-call timeout)
    # ------------------------------------------------------------------
    def _warmup_models(self):
        print("🔥 Warming up Ollama models (one-time cost)...")
        for model in set(self.models.values()):
            try:
                requests.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": model,
                        "prompt": "Say OK.",
                        "stream": False,
                        "options": {"num_predict": 5},
                    },
                    timeout=180,
                )
                print(f"   ✓ Warmed {model}")
            except Exception as e:
                print(f"   ⚠️ Warmup failed for {model}: {e}")

    # ------------------------------------------------------------------
    # Text generation
    # ------------------------------------------------------------------
    def generate(
        self,
        prompt: str,
        task_type: str = "general",
    ) -> Optional[str]:
        """
        Generate text using the model mapped to task_type.

        task_type ∈ {extractor, prosecutor, defense, judge, general}
        """

        model_name = self.models.get(
            task_type, self.models.get("judge")
        )

        for attempt in range(self.max_retries):
            try:
                payload = {
                    "model": model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.temperature,
                        # HARD CAP to avoid OOM / slow alloc
                        "num_predict": min(self.max_tokens, 256),
                    },
                }

                response = requests.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    # First call slower, later calls faster
                    timeout=180 if self.call_count == 0 else 90,
                )

                if response.status_code != 200:
                    print(f"⚠️ Ollama returned status {response.status_code}")
                    return None

                result = response.json()
                self.call_count += 1
                self.call_history[model_name] += 1

                return result.get("response", "").strip()

            except requests.exceptions.Timeout:
                print(
                    f"❌ Timeout on attempt {attempt + 1}/{self.max_retries}"
                )
                if attempt < self.max_retries - 1:
                    time.sleep(2)
                else:
                    return None

            except Exception as e:
                print(
                    f"❌ Generation failed on attempt {attempt + 1}: {e}"
                )
                if attempt < self.max_retries - 1:
                    time.sleep(2)
                else:
                    return None

        return None

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------
    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_calls": self.call_count,
            "calls_by_model": self.call_history,
            "estimated_cost": 0.0,  # Local = FREE
        }

    def print_stats(self):
        stats = self.get_stats()
        print("\n" + "=" * 60)
        print("OLLAMA USAGE STATISTICS")
        print("=" * 60)
        print(f"Total Calls: {stats['total_calls']}")
        print("\nCalls by Model:")
        for model, count in stats["calls_by_model"].items():
            print(f"  {model}: {count}")
        print("\nCost: FREE (local inference)")
        print("=" * 60)
