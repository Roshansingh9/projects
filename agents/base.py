from typing import List, Dict, Optional
from llm.groq_client import GroqClient

class BaseAgent:
    """Base class for all reasoning agents."""
    
    def __init__(self, llm_client: GroqClient, config: dict, task_type: str):
        self.llm = llm_client
        self.config = config
        self.task_type = task_type  # For model selection
    
    def format_evidence(self, evidence_chunks: List[Dict], max_chunks: int = 5) -> str:
        """Format evidence chunks for prompt."""
        if not evidence_chunks:
            return "No relevant evidence found."
        
        formatted = []
        for i, chunk in enumerate(evidence_chunks[:max_chunks]):
            formatted.append(
                f"[Evidence {i+1}] (Position: {chunk['position']}, "
                f"Similarity: {chunk['similarity']:.2f})\n{chunk['text']}\n"
            )
        
        return "\n".join(formatted)
    
    def extract_judgment(self, response: str) -> Dict:
        """
        Parse LLM response into structured judgment.
        
        Expected format:
        VERDICT: CONSISTENT|CONTRADICTORY|INSUFFICIENT
        CONFIDENCE: 0.0-1.0
        REASONING: ...
        """
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
                    # Handle both 0.X and X% formats
                    conf_str = conf_str.replace('%', '').strip()
                    confidence = float(conf_str)
                    # Normalize to 0-1 range
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
        """
        Abstract method - must be implemented by subclasses.
        
        Returns: {verdict, confidence, reasoning, evidence_used}
        """
        raise NotImplementedError