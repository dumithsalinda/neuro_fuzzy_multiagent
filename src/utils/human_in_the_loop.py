"""
Human-in-the-loop control utility for interactive experiment management.
Allows pausing, resuming, stopping, and adjusting agent/env parameters during runs.
"""
def human_in_the_loop_control(step, agent=None, env=None):
    print("\n[Human-in-the-Loop] Step {}: [p]ause, [r]esume, [s]top, [a]djust agent/env, [c]ontinue?".format(step))
    user_input = input("Enter command: ").strip().lower()
    if user_input == 'p':
        input("Paused. Press Enter to resume...")
        return 'pause'
    elif user_input == 's':
        print("Experiment stopped by user.")
        return 'stop'
    elif user_input == 'a':
        if agent is not None:
            try:
                new_epsilon = input("Enter new epsilon for agent (or blank to skip): ").strip()
                if new_epsilon:
                    agent.epsilon = float(new_epsilon)
                    print(f"Set agent epsilon to {agent.epsilon}")
            except Exception:
                print("Invalid epsilon input.")
        if env is not None:
            try:
                new_diff = input("Enter new difficulty for env (or blank to skip): ").strip()
                if new_diff and hasattr(env, 'set_difficulty'):
                    env.set_difficulty(float(new_diff))
                    print(f"Set environment difficulty to {new_diff}")
            except Exception:
                print("Invalid difficulty input.")
        return 'adjust'
    elif user_input == 'r':
        print("Resumed.")
        return 'resume'
    else:
        return 'continue'
