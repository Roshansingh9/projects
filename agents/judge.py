from agents.base import BaseAgent
from typing import Dict

class JudgeAgent(BaseAgent):
    """Adjudicates between prosecutor and defense arguments."""
    
    def __init__(self, llm_client, config: dict):
        super().__init__(llm_client, config, task_type='judge')
    
    def deliberate(self, claim: str, prosecutor_judgment: Dict, defense_judgment: Dict) -> Dict:
        """
        Make final judgment on a single claim using Groq's llama-3.3-70b.
        
        Decision logic:
        1. If prosecutor finds HARD contradiction → CONTRADICTORY
        2. If both agree → follow consensus
        3. If disagree → weigh confidence and evidence quality
        """
        
        # Handle insufficient evidence
        if (prosecutor_judgment['verdict'] == 'INSUFFICIENT' and 
            defense_judgment['verdict'] == 'INSUFFICIENT'):
            return {
                'verdict': 'INSUFFICIENT',
                'confidence': 0.0,
                'reasoning': 'Both sides lack sufficient evidence'
            }
        
        # Prosecutor found strong contradiction
        if (prosecutor_judgment['verdict'] == 'CONTRADICTORY' and 
            prosecutor_judgment['confidence'] > 0.7):
            return {
                'verdict': 'CONTRADICTORY',
                'confidence': prosecutor_judgment['confidence'],
                'reasoning': f"Hard contradiction found: {prosecutor_judgment['reasoning']}"
            }
        
        # Both agree it's consistent
        if (prosecutor_judgment['verdict'] == 'CONSISTENT' and 
            defense_judgment['verdict'] == 'CONSISTENT'):
            avg_conf = (prosecutor_judgment['confidence'] + defense_judgment['confidence']) / 2
            return {
                'verdict': 'CONSISTENT',
                'confidence': avg_conf,
                'reasoning': 'Both sides agree: consistent'
            }
        
        # Disagreement - use LLM to adjudicate
        prompt = f"""You are a JUDGE evaluating conflicting arguments about a backstory claim.

CLAIM:
{claim}

PROSECUTOR (finds contradictions):
Verdict: {prosecutor_judgment['verdict']}
Confidence: {prosecutor_judgment['confidence']:.2f}
Reasoning: {prosecutor_judgment['reasoning']}

DEFENSE (finds consistency):
Verdict: {defense_judgment['verdict']}
Confidence: {defense_judgment['confidence']:.2f}
Reasoning: {defense_judgment['reasoning']}

YOUR TASK:
Determine which side has the stronger argument based on:
1. Strength of evidence cited
2. Logical soundness of reasoning
3. Conservative principle: contradictions override weak consistency

OUTPUT FORMAT (MUST FOLLOW EXACTLY):
VERDICT: CONSISTENT|CONTRADICTORY|INSUFFICIENT
CONFIDENCE: [0.0-1.0]
REASONING: [Explain your final judgment in 2-3 sentences]

Think step-by-step, but output ONLY the format above."""

        response = self.llm.generate(prompt, task_type=self.task_type)
        
        if not response:
            # Fallback: trust prosecutor more (conservative)
            return prosecutor_judgment
        
        return self.extract_judgment(response)