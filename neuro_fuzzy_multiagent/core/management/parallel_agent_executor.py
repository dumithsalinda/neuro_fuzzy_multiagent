"""
parallel_agent_executor.py

Utility for running agent steps in parallel (threaded or process-based).
"""

from concurrent.futures import ThreadPoolExecutor, as_completed


def run_agents_parallel(agents, observations, max_workers=4):
    """
    Run agent.act(obs) for each agent/observation pair in parallel.
    Args:
        agents: list of agent objects with an 'act' method
        observations: list of observations (one per agent)
        max_workers: number of threads
    Returns:
        actions: list of actions (same order as agents)
    """
    results = [None] * len(agents)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(agent.act, obs): idx
            for idx, (agent, obs) in enumerate(zip(agents, observations))
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            results[idx] = future.result()
    return results
