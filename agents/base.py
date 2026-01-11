from typing import List, Dict, Optional
from llm.ollama_client import OllamaClient  # CHANGED: Use Ollama

class BaseAgent:
    """Base class for all reasoning agents."""
    
    def __init__(self, llm_client: OllamaClient, config: dict, task_type: str):  # CHANGED
        self.llm = llm_client
        self.config = config
        self.task_type = task_type
    
    def format_evidence(self, evidence_chunks: List[Dict], max_chunks: int = 3) -> str:
        """Format evidence chunks for prompt."""
        if not evidence_chunks:
            return "No relevant evidence found."
        
        formatted = []
        for i, chunk in enumerate(evidence_chunks[:max_chunks]):
            # Truncate very long chunks
            text = chunk['text']
            if len(text) > 800:
                text = text[:800] + "..."
            
            formatted.append(
                f"[Evidence {i+1}] (Similarity: {chunk['similarity']:.2f})\n{text}\n"
            )
        
        return "\n".join(formatted)
    
    def extract_judgment(self, response: str) -> Dict:
        """Parse LLM response into structured judgment."""
        if not response:
            return {
                'verdict': 'INSUFFICIENT',
                'confidence': 0.0,
                'reasoning': 'LLM returned empty response'
            }
        
        lines = response.split('\n')
        judgment = {
            'verdict': 'INSUFFICIENT',
            'confidence': 0.0,
            'reasoning': response
        }
        
        for line in lines:
            line_upper = line.upper()
            
            if line_upper.startswith('VERDICT:'):
                verdict = line.split(':', 1)[1].strip().upper()
                if verdict in ['CONSISTENT', 'CONTRADICTORY', 'INSUFFICIENT']:
                    judgment['verdict'] = verdict
            
            elif line_upper.startswith('CONFIDENCE:'):
                try:
                    conf_str = line.split(':', 1)[1].strip()
                    conf_str = conf_str.replace('%', '').strip()
                    confidence = float(conf_str)
                    if confidence > 1.0:
                        confidence = confidence / 100.0
                    judgment['confidence'] = max(0.0, min(1.0, confidence))
                except:
                    pass
            
            elif line_upper.startswith('REASONING:'):
                reasoning = line.split(':', 1)[1].strip()
                if reasoning:
                    judgment['reasoning'] = reasoning
        
        return judgment
    
    def analyze_claim(self, claim: str, evidence: List[Dict]) -> Dict:
        """Abstract method - must be implemented by subclasses."""
        raise NotImplementedError