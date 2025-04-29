# env.py
import logging
import random # Added for seeding and example env variation

logger = logging.getLogger(__name__)

class BaseEnvironment:
    """Abstract base class for task environments."""
    def get_task(self) -> str:
        """
        Returns the description/input for the current task the agent needs to solve.
        """
        raise NotImplementedError

    def evaluate(self, agent_answer: str) -> float:
        """
        Evaluates the agent's final answer string against the ground truth.

        Args:
            agent_answer: The answer string extracted from the agent's solver output.

        Returns:
            A score between 0.0 (completely wrong) and 1.0 (perfectly correct).
            Partial credit is possible depending on the environment implementation.
        """
        raise NotImplementedError

# --- Example Implementation ---

class SimpleMathEnv(BaseEnvironment):
    """A simple environment asking for the sum of two numbers in text."""
    def __init__(self, num1=None, num2=None, seed=None): # Added seed parameter
        # Seed the random number generator for reproducibility if a seed is provided
        if seed is not None:
            random.seed(seed)

        # Allow specific numbers or generate random ones
        self.number1 = num1 if num1 is not None else random.randint(1, 100)
        self.number2 = num2 if num2 is not None else random.randint(1, 100)
        self.task = f"Calculate the sum of {self.number1} and {self.number2}."
        self.correct_answer = str(self.number1 + self.number2)
        logger.debug(f"Initialized SimpleMathEnv (Seed: {seed}): Task='{self.task}', Answer='{self.correct_answer}'")

    def get_task(self) -> str:
        return self.task

    def evaluate(self, agent_answer: str) -> float:
        """Score is 1.0 if the answer is exactly correct, 0.0 otherwise."""
        clean_answer = str(agent_answer).strip()

        # Example: Basic check for numeric format before comparison
        if not re.match(r'^-?\d+(\.\d+)?$', clean_answer): # Check if it looks like a number
            logger.warning(f"Evaluation: Agent answer '{clean_answer}' is not in a standard numeric format.")
            # Optionally return 0 here, or let the comparison handle it
            # return 0.0

        if clean_answer == self.correct_answer:
            logger.debug(f"Evaluation: Correct! ('{clean_answer}' == '{self.correct_answer}')")
            return 1.0
        else:
            # --- Potential Area for Partial Credit ---
            # try:
            #     numeric_answer = float(clean_answer)
            #     numeric_correct = float(self.correct_answer)
            #     diff = abs(numeric_answer - numeric_correct)
            #     if diff <= 1: return 0.8 # Near miss
            #     if diff <= 5: return 0.5 # Wider miss
            # except ValueError:
            #     pass # Handle non-numeric answers if strict parsing above is removed
            # --- End Partial Credit Example ---

            logger.warning(f"Evaluation: Incorrect. Agent answered '{clean_answer}', Expected '{self.correct_answer}'")
            return 0.0

# Re-add re here as it's used in the improved evaluate method
import re

# Example Usage (if run directly)
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    # Test with seed
    env1 = SimpleMathEnv(seed=42)
    print(f"[Seed 42] Task: {env1.get_task()}")
    print(f"[Seed 42] Correct Answer: {env1.correct_answer}")
    print(f"Evaluating '{env1.correct_answer}': {env1.evaluate(env1.correct_answer)}")
    # Test without seed (will likely differ)
    env2 = SimpleMathEnv()
    print(f"[No Seed] Task: {env2.get_task()}")