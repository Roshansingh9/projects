import time
from groq import Groq
from typing import Optional, Dict, Any
import yaml

class GroqClient:
    """
    Centralized Groq API client with multi-model support.
    
    Free tier limits (as of 2024):
    - llama-3.3-70b-versatile: 30 req/min, 6000 req/day
    - llama-3.1-8b-instant: 30 req/min, 14400 req/day
    - mixtral-8x7b-32768: 30 req/min, 14400 req/day
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize Groq client
        self.client = Groq(
            api_key=self.config['llm']['api_key']
        )
        
        # Model assignments
        self.models = self.config['llm']['models']
        
        # Rate limiting
        self.last_call_time = 0
        self.min_interval = 60.0 / self.config['llm']['requests_per_minute']
        self.call_count = 0
        self.call_history = {model: 0 for model in self.models.values()}
    
    def _rate_limit(self):
        """Ensure we respect rate limits."""
        elapsed = time.time() - self.last_call_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_call_time = time.time()
    
    def generate(
        self, 
        prompt: str, 
        task_type: str = "general",
        max_retries: int = None
    ) -> Optional[str]:
        """
        Generate response using appropriate model for task.
        
        Args:
            prompt: Input prompt
            task_type: One of ['claim_extraction', 'prosecutor', 'defense', 'judge', 'general']
            max_retries: Number of retry attempts
        
        Returns:
            Generated text or None if all retries fail
        """
        if max_retries is None:
            max_retries = self.config['llm']['max_retries']
        
        # Select appropriate model
        model_name = self.models.get(task_type, self.models['judge'])
        
        self._rate_limit()
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a precise reasoning system. Follow instructions exactly and output only the requested format."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=self.config['llm']['temperature'],
                    max_tokens=self.config['llm']['max_tokens'],
                    top_p=self.config['llm']['top_p']
                )
                
                self.call_count += 1
                self.call_history[model_name] += 1
                
                # Extract text from response
                if response.choices and len(response.choices) > 0:
                    return response.choices[0].message.content.strip()
                else:
                    print(f"⚠️ Empty response from {model_name}")
                    return None
            
            except Exception as e:
                print(f"❌ Attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    # Exponential backoff
                    sleep_time = 2 ** attempt
                    print(f"   Retrying in {sleep_time}s...")
                    time.sleep(sleep_time)
                else:
                    print(f"   All retries exhausted for {task_type}")
                    return None
        
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Return usage statistics."""
        return {
            "total_calls": self.call_count,
            "calls_by_model": self.call_history,
            "estimated_cost": 0.0  # Free tier
        }
    
    def print_stats(self):
        """Print detailed usage statistics."""
        stats = self.get_stats()
        print("\n" + "=" * 60)
        print("GROQ API USAGE STATISTICS")
        print("=" * 60)
        print(f"Total Calls: {stats['total_calls']}")
        print("\nCalls by Model:")
        for model, count in stats['calls_by_model'].items():
            print(f"  {model}: {count}")
        print("=" * 60)