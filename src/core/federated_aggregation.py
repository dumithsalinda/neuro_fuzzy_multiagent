"""
federated_aggregation.py

Federated aggregation utilities for Q-tables, neural network weights, and fuzzy rules.
"""
import numpy as np
import ray

def federated_update(ray_agents):
    """
    Perform federated aggregation and broadcast to all agents.
    Args:
        ray_agents: list of Ray agent actor handles (RayAgentWrapper)
    Returns:
        aggregated_knowledge (dict)
    """
    knowledges = ray.get([agent.get_knowledge.remote() for agent in ray_agents])
    # Determine type
    if all(_is_nn_weights(k) for k in knowledges):
        agg = _aggregate_nn_weights(knowledges)
    elif all(isinstance(k, dict) for k in knowledges):
        # Q-table (dict of state->action values)
        agg = _aggregate_qtables(knowledges)
    elif all(_is_fuzzy_rules(k) for k in knowledges):
        agg = _aggregate_fuzzy_rules(knowledges)
    else:
        raise ValueError("Unsupported or mixed knowledge types for aggregation.")
    # Broadcast back
    ray.get([agent.set_knowledge.remote(agg) for agent in ray_agents])
    return agg

def _aggregate_qtables(qtables):
    # Average Q-values for each (state, action)
    all_keys = set().union(*(qt.keys() for qt in qtables))
    agg_q = {}
    for key in all_keys:
        vals = [qt[key] for qt in qtables if key in qt]
        agg_q[key] = float(np.mean(vals))
    return agg_q

def _is_nn_weights(obj):
    return isinstance(obj, dict) and all(isinstance(v, np.ndarray) for v in obj.values())

def _aggregate_nn_weights(weights_list):
    # FedAvg: average each weight array (elementwise, preserve shape)
    agg = {}
    for k in weights_list[0]:
        arrs = [np.array(w[k]) for w in weights_list]
        arr_types = [type(a) for a in arrs]
        arr_shapes = [a.shape for a in arrs]
        print(f"NN aggregation for key '{k}': arrs = {arrs}, types = {arr_types}, shapes = {arr_shapes}")
        if not all(s == arr_shapes[0] for s in arr_shapes):
            raise ValueError(f"Shape mismatch for NN weight '{k}': {arr_shapes}")
        if arr_shapes[0] == ():
            # All are scalars
            arrs_stacked = np.array(arrs)
            print(f"arrs_stacked for '{k}': shape = {arrs_stacked.shape}, values =\n{arrs_stacked}")
            mean = float(np.mean(arrs_stacked))
            print(f"mean for '{k}' (scalar): {mean}, shape: scalar")
            agg[k] = mean
        else:
            # All are arrays of same shape (including shape (1,), (2,), (n,))
            arrs_stacked = np.stack(arrs, axis=0)
            print(f"arrs_stacked for '{k}': shape = {arrs_stacked.shape}, values =\n{arrs_stacked}")
            mean = np.mean(arrs_stacked, axis=0)
            print(f"mean for '{k}': {mean}, shape: {getattr(mean, 'shape', type(mean))}")
            if k == 'w1':
                print(f"DEBUG: arrs_stacked for 'w1' = {arrs_stacked}, mean = {mean}, mean.shape = {getattr(mean, 'shape', type(mean))}")
            agg[k] = mean
    return agg

def _is_fuzzy_rules(obj):
    return isinstance(obj, list) and all(isinstance(r, dict) for r in obj)

def _aggregate_fuzzy_rules(rules_list):
    # For each rule index, average numeric params, majority vote for discrete
    n_rules = len(rules_list[0])
    agg_rules = []
    for i in range(n_rules):
        params = [rules[i] for rules in rules_list]
        agg = {}
        for k in params[0]:
            vals = [p[k] for p in params]
            if all(isinstance(v, (int, float, np.number)) for v in vals):
                agg[k] = float(np.mean(vals))
            else:
                # Discrete: majority vote
                from collections import Counter
                agg[k] = Counter(vals).most_common(1)[0][0]
        agg_rules.append(agg)
    return agg_rules
