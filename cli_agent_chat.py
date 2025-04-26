"""
CLI Chat Interface for Human-Agent Collaboration
Allows user to interact with NeuroFuzzyFusionAgent via natural language commands/questions.
"""

import numpy as np
from src.core.neuro_fuzzy_fusion_agent import NeuroFuzzyFusionAgent

# Initialize agent (example config)
input_dims = [4, 3]
hidden_dim = 16
output_dim = 5
fusion_type = "concat"
fusion_alpha = 0.6
fuzzy_config = None
agent = NeuroFuzzyFusionAgent(
    input_dims=input_dims,
    hidden_dim=hidden_dim,
    output_dim=output_dim,
    fusion_type=fusion_type,
    fuzzy_config=fuzzy_config,
    fusion_alpha=fusion_alpha,
)

print("\n=== NeuroFuzzyFusionAgent CLI Chat ===")
print("Type 'help' for commands. Type 'quit' to exit.\n")

last_obs = [np.random.rand(4), np.random.rand(3)]  # Example observation
last_action = None

while True:
    user_input = input("You: ").strip().lower()
    if user_input in ("quit", "exit"):
        break
    if user_input == "help":
        print("Commands:")
        print("  explain              - Explain last action")
        print("  act                  - Take action on new random obs")
        print("  set fusion alpha X   - Set fusion alpha (0-1)")
        print("  show rules           - Show fuzzy rules")
        print("  feedback [text]      - Provide feedback to agent")
        print("  help                 - Show this help message")
        print("  quit                 - Exit chat")
        continue
    if user_input.startswith("set fusion alpha"):
        try:
            alpha = float(user_input.split()[-1])
            agent.set_fusion_alpha(alpha)
            print(f"Fusion alpha set to {alpha}")
        except Exception:
            print("Usage: set fusion alpha X (where X is a float between 0 and 1)")
        continue
    if user_input == "act":
        last_obs = [np.random.rand(4), np.random.rand(3)]
        last_action = agent.act(last_obs)
        print(f"Agent action: {last_action} (on random obs: {last_obs})")
        continue
    if user_input == "explain":
        if last_obs is None:
            print("No last observation. Use 'act' first.")
        else:
            exp = agent.explain_action(last_obs)
            print("Explanation:")
            for k, v in exp.items():
                print(f"  {k}: {v}")
        continue
    if user_input == "show rules":
        rules = getattr(agent.fuzzy_system, "rules", [])
        if not rules:
            print("No fuzzy rules defined.")
        else:
            print("Fuzzy Rules:")
            for i, rule in enumerate(rules):
                print(
                    f"  Rule {i}: Antecedents: {rule.antecedents}, Consequent: {rule.consequent}"
                )
        continue
    if user_input.startswith("feedback"):
        feedback = user_input[len("feedback") :].strip()
        if feedback.startswith("rule") and "obs=" in feedback and "action=" in feedback:
            # Example: feedback rule obs=[0.1,0.2,0.3,0.4,0.5,0.6,0.7] action=2
            import re

            obs_match = re.search(r"obs=\[(.*?)\]", feedback)
            action_match = re.search(r"action=([0-9]+)", feedback)
            if obs_match and action_match:
                obs_str = obs_match.group(1)
                obs_vals = [float(x.strip()) for x in obs_str.split(",") if x.strip()]
                action_val = int(action_match.group(1))
                # Assume two modalities: obs[0:4], obs[4:7]
                obs_modalities = [obs_vals[:4], obs_vals[4:]]
                # Demo fuzzy sets: for each input, Low/High sets
                from src.core.fuzzy_system import FuzzySet

                fuzzy_sets_per_input = [
                    [FuzzySet("Low", [0, 1]), FuzzySet("High", [1, 1])]
                    for _ in obs_vals
                ]
                agent.add_fuzzy_rule_from_feedback(
                    obs_vals, action_val, fuzzy_sets_per_input
                )
                print(f"Fuzzy rule added for obs={obs_vals}, action={action_val}")
                # Show updated rules
                rules = getattr(agent.fuzzy_system, "rules", [])
                print("Updated Fuzzy Rules:")
                for i, rule in enumerate(rules):
                    print(
                        f"  Rule {i}: Antecedents: {rule.antecedents}, Consequent: {rule.consequent}"
                    )
            else:
                print("Usage: feedback rule obs=[x1,...,xN] action=X")
        else:
            print(f"Feedback received: '{feedback}' (not yet used for learning)")
        continue
    print("Unknown command. Type 'help' for options.")

print("Exiting chat.")
