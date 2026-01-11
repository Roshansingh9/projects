from agents.base import BaseAgent
from typing import List, Dict

class DefenseAgent(BaseAgent):
    """Finds consistency paths between backstory claims and novel evidence."""
    
    def __init__(self, llm_client, config: dict):
        super().__init__(llm_client, config, task_type='defense')
    
    def analyze_claim(self, claim: str, evidence: List[Dict]) -> Dict:
        """Search for consistency using Groq's llama-3.1-8b-instant (faster)."""
        if not evidence:
            return {
                'verdict': 'INSUFFICIENT',
                'confidence': 0.0,
                'reasoning': 'No evidence available for analysis',
                'evidence_used': []
            }
        
        evidence_text = self.format_evidence(evidence)
        
        prompt = f"""You are a DEFENSE attorney analyzing whether a backstory claim is CONSISTENT with a novel.

BACKSTORY CLAIM:
{claim}

NOVEL EVIDENCE:
{evidence_text}

YOUR TASK:
1. Find ANY plausible interpretation where the claim fits the evidence
2. Look for:
   - Compatible causal pathways (claim → evidence makes sense)
   - Consistent character development (claim explains later behavior)
   - No explicit contradictions

PERMISSIVE RULES:
- If claim doesn't contradict evidence, it's CONSISTENT
- Unstated details can be assumed if plausible
- Coincidences are acceptable unless impossible
- Benefit of doubt favors CONSISTENT

OUTPUT FORMAT (MUST FOLLOW EXACTLY):
VERDICT: CONSISTENT|CONTRADICTORY|INSUFFICIENT
CONFIDENCE: [0.0-1.0]
REASONING: [Explain your verdict in 2-3 sentences]

Think step-by-step, but output ONLY the format above."""

        response = self.llm.generate(prompt, task_type=self.task_type)
        
        if not response:
            return {
                'verdict': 'INSUFFICIENT',
                'confidence': 0.0,
                'reasoning': 'LLM call failed',
                'evidence_used': []
            }
        
        judgment = self.extract_judgment(response)
        judgment['evidence_used'] = [e['chunk_id'] for e in evidence[:5]]
        
        return judgment