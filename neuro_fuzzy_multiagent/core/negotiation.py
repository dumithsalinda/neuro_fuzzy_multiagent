import os
from neuro_fuzzy_multiagent.core.fuzzy.fuzzy_controller import FuzzyController

class NegotiatorAgent:
    """
    Generic multi-agent negotiation and aggregation agent.
    Agents must implement propose_decision(topic, context) -> {proposal, confidence, explanation}
    """
    def __init__(self, agents, fuzzy_rule_dir=None):
        self.agents = agents  # Dict[str, agent]
        self.fuzzy_rule_dir = fuzzy_rule_dir or os.path.dirname(__file__)

    def negotiate(self, topic, context=None, strategy="majority"):
        context = context or {}
        proposals, confidences = {}, {}
        for name, agent in self.agents.items():
            resp = agent.propose_decision(topic, context)
            proposals[name] = resp.get("proposal")
            confidences[name] = resp.get("confidence", 1.0)
        if strategy == "majority":
            result = self.majority_vote(proposals)
            agg_expl = "Majority vote"
        elif strategy == "weighted":
            result = self.weighted_vote(proposals, confidences)
            agg_expl = "Weighted by confidence"
        elif strategy == "fuzzy":
            fuzzy_result = self.fuzzy_aggregate(topic, proposals, confidences, context)
            result = fuzzy_result["decision"]
            agg_expl = fuzzy_result.get("fuzzy_trace", "Fuzzy aggregation")
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        return {"decision": result, "aggregation": agg_expl, "proposals": proposals, "confidences": confidences}

    def majority_vote(self, proposals):
        from collections import Counter
        counts = Counter(proposals.values())
        return counts.most_common(1)[0][0] if counts else None

    def weighted_vote(self, proposals, confidences):
        from collections import defaultdict
        score = defaultdict(float)
        for name, proposal in proposals.items():
            score[proposal] += confidences.get(name, 1.0)
        return max(score, key=score.get) if score else None

    def fuzzy_aggregate(self, topic, proposals, confidences, context):
        yaml_path = os.path.join(self.fuzzy_rule_dir, "rules", f"fuzzy_{topic}.yaml")
        fuzzy_ctrl = FuzzyController.from_yaml(yaml_path)
        # Example for resource_allocation
        if topic == "resource_allocation":
            urgency = context.get("urgency", "medium")
            privacy_risk = context.get("privacy_risk", "medium")
            utility = max(confidences.values())
            if utility >= 0.8:
                utility_label = "high"
            elif utility >= 0.5:
                utility_label = "medium"
            else:
                utility_label = "low"
            fuzzy_input = dict(urgency=urgency, privacy_risk=privacy_risk, utility=utility_label)
            best_out = None
            for rule in fuzzy_ctrl.rules.get("rules", []):
                rule_if = rule.get("if", {})
                match = all(fuzzy_input.get(k) == v for k, v in rule_if.items())
                if match:
                    best_out = rule["then"]["allocation_score"]
                    break
            if not best_out:
                best_out = "maybe"
            fuzzy_explanation = fuzzy_ctrl.explain(fuzzy_input, best_out)
            return {"decision": best_out, "fuzzy_trace": fuzzy_explanation}
        if topic == "notification_suppression":
            urgency = context.get("urgency", "medium")
            user_mode = context.get("user_mode", "normal")
            interruption_cost = context.get("interruption_cost", "medium")
            fuzzy_input = dict(urgency=urgency, user_mode=user_mode, interruption_cost=interruption_cost)
            best_out = None
            for rule in fuzzy_ctrl.rules.get("rules", []):
                rule_if = rule.get("if", {})
                match = all(fuzzy_input.get(k) == v for k, v in rule_if.items())
                if match:
                    best_out = rule["then"]["notification_action"]
                    break
            if not best_out:
                best_out = "delay"
            fuzzy_explanation = fuzzy_ctrl.explain(fuzzy_input, best_out)
            return {"decision": best_out, "fuzzy_trace": fuzzy_explanation}
        # Default fallback
        return {"decision": self.weighted_vote(proposals, confidences), "fuzzy_trace": "Used weighted vote fallback."}
