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
        Extract high-quality claims from backstory using Groq's llama-3.3-70b.
        
        IMPROVED: Focus on verifiable, specific claims that can be checked against novel.
        """
        max_claims = min(5, self.config['agents']['max_claims_per_backstory'])
        
        prompt = f"""Extract the MOST IMPORTANT and VERIFIABLE claims from this character backstory.

Focus on claims that are:
✓ Specific events (battles, meetings, deaths, discoveries)
✓ Concrete relationships (who helped/betrayed/knew whom)  
✓ Factual details (locations, objects, dates, actions taken)
✓ Checkable against the novel's plot

AVOID claims that are:
✗ Generic statements ("The character was skilled")
✗ Emotional/internal states ("felt betrayed")
✗ Redundant sub-claims
✗ Too vague to verify

BACKSTORY:
{backstory}

OUTPUT FORMAT:
Return ONLY a numbered list of {max_claims} key claims:
1. [First specific, verifiable claim]
2. [Second specific, verifiable claim]
...

Extract exactly {max_claims} claims, prioritizing the most fact-checkable ones."""

        response = self.llm.generate(prompt, task_type='claim_extraction')
        
        if not response:
            # Fallback: split by sentences, take first N
            sentences = [s.strip() + '.' for s in backstory.split('.') if len(s.strip()) > 20]
            return sentences[:max_claims]
        
        # Parse numbered list
        claims = []
        for line in response.split('\n'):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                # Remove numbering/bullets
                claim = line.lstrip('0123456789.-•)').strip()
                if claim and len(claim) > 15:  # Filter out empty/trivial claims
                    claims.append(claim)
        
        # Ensure we have at least 1 claim
        if not claims:
            claims = [backstory[:200]]  # Use first 200 chars as fallback
        
        return claims[:max_claims]
    
    def deliberate_on_backstory(self, backstory: str, book_id: str) -> List[Dict]:
        """
        Run full debate on all claims.
        
        Returns: List of {claim, prosecutor_judgment, defense_judgment, final_judgment}
        """
        print(f"\n🔍 Extracting claims from backstory...")
        claims = self.extract_claims(backstory)
        print(f"   → {len(claims)} high-quality claims identified")
        
        # Retrieve evidence for all claims at once
        print(f"\n📚 Retrieving evidence for claims...")
        evidence_map = self.retriever.retrieve_for_claims(claims, book_id)
        
        # DEBUG: Show retrieval stats
        total_evidence = sum(len(ev) for ev in evidence_map.values())
        print(f"   → {total_evidence} total evidence chunks retrieved")
        
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
        
        # DEBUG: Show overall statistics
        verdicts = [d['final']['verdict'] for d in deliberations]
        print(f"\n📊 Deliberation Summary:")
        print(f"   CONSISTENT: {verdicts.count('CONSISTENT')}")
        print(f"   CONTRADICTORY: {verdicts.count('CONTRADICTORY')}")
        print(f"   INSUFFICIENT: {verdicts.count('INSUFFICIENT')}")
        
        return deliberations