from typing import List, Dict
from agents.prosecutor import ProsecutorAgent
from agents.defense import DefenseAgent
from agents.judge import JudgeAgent
from llm.groq_client import GroqClient

class DebateOrchestrator:
    """Orchestrates multi-agent deliberation on backstory consistency."""
    
    def __init__(self, llm_client: GroqClient, retriever, config: dict):
        self.llm = llm_client
        self.retriever = retriever
        self.config = config
        
        self.prosecutor = ProsecutorAgent(llm_client, config)
        self.defense = DefenseAgent(llm_client, config)
        self.judge = JudgeAgent(llm_client, config)
    
    def extract_claims(self, backstory: str) -> List[str]:
        """
        Extract atomic claims from backstory using Groq's llama-3.3-70b.
        """
        prompt = f"""Extract atomic claims from this character backstory. Each claim should be:
- A single verifiable statement
- Specific enough to check against evidence
- Free of compound statements

BACKSTORY:
{backstory}

OUTPUT FORMAT:
Return ONLY a numbered list of claims, one per line:
1. [First claim]
2. [Second claim]
...

Limit to {self.config['agents']['max_claims_per_backstory']} most important claims."""

        response = self.llm.generate(prompt, task_type='claim_extraction')
        
        if not response:
            # Fallback: split by sentences
            return [s.strip() + '.' for s in backstory.split('.') if len(s.strip()) > 20][:10]
        
        # Parse numbered list
        claims = []
        for line in response.split('\n'):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                # Remove numbering/bullets
                claim = line.lstrip('0123456789.-•)').strip()
                if claim:
                    claims.append(claim)
        
        return claims[:self.config['agents']['max_claims_per_backstory']]
    
    def deliberate_on_backstory(self, backstory: str, book_id: str) -> List[Dict]:
        """
        Run full debate on all claims.
        
        Returns: List of {claim, prosecutor_judgment, defense_judgment, final_judgment}
        """
        print(f"\n🔍 Extracting claims from backstory...")
        claims = self.extract_claims(backstory)
        print(f"   → {len(claims)} claims identified")
        
        # Retrieve evidence for all claims at once
        print(f"\n📚 Retrieving evidence for claims...")
        evidence_map = self.retriever.retrieve_for_claims(claims, book_id)
        
        deliberations = []
        
        for i, claim in enumerate(claims):
            print(f"\n⚖️  Claim {i+1}/{len(claims)}: {claim[:80]}...")
            
            evidence = evidence_map[claim]
            print(f"   → {len(evidence)} evidence chunks retrieved")
            
            # Prosecutor analyzes
            print(f"   → Prosecutor analyzing...")
            prosecutor_judgment = self.prosecutor.analyze_claim(claim, evidence)
            
            # Defense analyzes
            print(f"   → Defense analyzing...")
            defense_judgment = self.defense.analyze_claim(claim, evidence)
            
            # Judge deliberates
            print(f"   → Judge deliberating...")
            final_judgment = self.judge.deliberate(claim, prosecutor_judgment, defense_judgment)
            
            deliberations.append({
                'claim': claim,
                'prosecutor': prosecutor_judgment,
                'defense': defense_judgment,
                'final': final_judgment
            })
            
            print(f"   ✓ Verdict: {final_judgment['verdict']} "
                  f"(confidence: {final_judgment['confidence']:.2f})")
        
        return deliberations