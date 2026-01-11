from typing import List, Dict

class ConstraintTracker:
    """Tracks hard and soft constraints from deliberations."""
    
    def classify_constraints(self, deliberations: List[Dict]) -> Dict:
        """
        Classify verdicts into hard contradictions vs soft evidence.
        
        Returns: {
            'hard_contradictions': count,
            'soft_contradictions': count,
            'consistent_claims': count,
            'insufficient_evidence': count
        }
        """
        classification = {
            'hard_contradictions': 0,
            'soft_contradictions': 0,
            'consistent_claims': 0,
            'insufficient_evidence': 0
        }
        
        for delib in deliberations:
            verdict = delib['final']['verdict']
            confidence = delib['final']['confidence']
            
            if verdict == 'CONTRADICTORY':
                if confidence > 0.7:
                    classification['hard_contradictions'] += 1
                else:
                    classification['soft_contradictions'] += 1
            
            elif verdict == 'CONSISTENT':
                classification['consistent_claims'] += 1
            
            else:  # INSUFFICIENT
                classification['insufficient_evidence'] += 1
        
        return classification
    
    def has_critical_violations(self, classification: Dict) -> bool:
        """
        Check if hard constraints are violated.
        
        Even ONE hard contradiction → INCONSISTENT backstory
        """
        return classification['hard_contradictions'] > 0
    
    def get_evidence_coverage(self, classification: Dict) -> float:
        """
        Calculate what fraction of claims have sufficient evidence.
        
        Low coverage → conservative judgment needed
        """
        total = sum(classification.values())
        if total == 0:
            return 0.0
        
        sufficient = total - classification['insufficient_evidence']
        return sufficient / total