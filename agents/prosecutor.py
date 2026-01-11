from agents.base import BaseAgent
from typing import List, Dict

class ProsecutorAgent(BaseAgent):
    """Finds contradictions between backstory claims and novel evidence."""
    
    def __init__(self, llm_client, config: dict):
        super().__init__(llm_client, config, task_type='prosecutor')
    
    def analyze_claim(self, claim: str, evidence: List[Dict]) -> Dict:
        """Search for contradictions using Groq's llama-3.3-70b model."""
        if not evidence:
            return {
                'verdict': 'INSUFFICIENT',
                'confidence': 0.0,
                'reasoning': 'No evidence available for analysis',
                'evidence_used': []
            }
        
        evidence_text = self.format_evidence(evidence)
        
        prompt = f"""You are a PROSECUTOR analyzing whether a backstory claim CONTRADICTS a novel.

BACKSTORY CLAIM:
{claim}

NOVEL EVIDENCE:
{evidence_text}

YOUR TASK:
1. Identify ANY direct contradictions between the claim and evidence
2. Look for:
   - Temporal impossibilities (events that couldn't happen in claimed order)
   - Logical contradictions (claim states X, novel shows NOT X)
   - Causal violations (claim's preconditions prevent novel's events)

STRICT RULES:
- A contradiction must be EXPLICIT and DIRECT
- Absence of confirmation is NOT contradiction
- Unexplained events are NOT contradictions
- Only flag HARD contradictions, not soft implausibilities

OUTPUT FORMAT (MUST FOLLOW EXACTLY):
VERDICT: CONTRADICTORY|CONSISTENT|INSUFFICIENT
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