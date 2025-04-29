# tools.py
import logging
from datetime import datetime, timezone
from pathlib import Path
import traceback
import json
import ast # For basic sanity check
import time
from typing import Dict, Any, Callable, Optional, List

# Assume llm_interface provides OllamaInterface
try:
    from llm_interface import OllamaInterface
except ImportError:
    # Define a dummy class if llm_interface is somehow unavailable during tool definition
    # This allows the file to be imported but tools requiring LLM will fail gracefully later
    class OllamaInterface: pass
    print("WARNING: llm_interface.py not found, LLM-dependent tools may fail.")

# Assume env provides BaseEnvironment
try:
    from env import BaseEnvironment
except ImportError:
    class BaseEnvironment: pass
    print("WARNING: env.py not found, Environment-dependent tools may fail.")


logger = logging.getLogger("agent_tools")

# --- Constants ---
# List of tool names that modify code strings in the state dict
CODE_MODIFYING_TOOLS: List[str] = ['adjust_logic', 'improve_decision_logic']

# Allowed tools list for reference or potential future validation
ALLOWED_TOOLS_LIST: List[str] = [
    'self_inspect', 'adjust_logic', 'evaluate', 'display_analysis',
    'improve_decision_logic', 'analyze_failures', 'tune_hyperparams'
]


# --- Tool Implementations ---

def self_inspect(state: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """Returns the current code state strings."""
    logger.info("Executing: self_inspect")
    # This operation is generally safe and should succeed
    try:
        result_data = {
            "solver_code": state.get("solver_code", "# Solver code missing"),
            # Ensure correct key name consistent with agent.py's state
            "decision_code": state.get("decision_logic_code", "# Decision logic code missing"),
        }
        return {
            "success": True,
            "result": result_data,
            "error": None
        }
    except Exception as e:
        logger.error(f"Unexpected error during self_inspect: {e}", exc_info=True)
        return {"success": False, "result": {}, "error": f"Internal error: {e}"}


def adjust_logic(state: Dict[str, Any], target: str, code: str, **kwargs) -> Dict[str, Any]:
    """Validates code, saves snapshot, updates code string in state."""
    logger.info(f"Executing: adjust_logic for target '{target}'")
    snapshot_dir = Path(state.get('snapshot_dir_config', 'snapshots')) # Get dir from state

    # --- Parameter Validation ---
    # Ensure target uses the correct key name format used in state
    state_key_target = f"{target}_code"
    if target not in ['solver', 'decision_logic']:
        err = f"Invalid target '{target}'. Must be 'solver' or 'decision_logic'."
        logger.error(err)
        return {"success": False, "updated": False, "error": err}
    if not isinstance(code, str) or len(code.strip()) < 10:
        err = "Invalid or effectively empty code provided"
        logger.error(f"{err} for target {target}")
        return {"success": False, "updated": False, "error": err}

    # --- Basic AST Check ---
    try:
        ast.parse(code)
        logger.debug(f"AST check passed for {target} update.")
    except SyntaxError as e:
        err = f"Proposed code for {target} has SyntaxError: {e}"
        logger.error(err)
        return {"success": False, "updated": False, "error": err} # Fail before snapshotting bad code
    except Exception as e:
        logger.warning(f"Unexpected error during AST check for {target}: {e}. Proceeding cautiously.")
        # Potentially return success=False here if AST check must pass? For now, just warn.

    # --- Snapshot ---
    iteration = state.get('iteration', 0)
    snapshot_path = None
    snapshot_error = None
    try:
        snapshot_dir.mkdir(exist_ok=True)
        ts = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        # Use the 'target' name (solver/decision_logic) for the filename consistency
        filename = f"{target}_iter{iteration}_{ts}.py"
        filepath = snapshot_dir / filename
        # Add snapshot headers
        header = f"# Snapshot at Iteration: {iteration} (Before Apply)\n"
        header += f"# Target Component: {target}\n"
        header += f"# Timestamp (UTC): {ts}\n\n"
        filepath.write_text(header + code, encoding='utf-8')
        snapshot_path = str(filepath)
        logger.info(f"Snapshot saved to: {snapshot_path}")
    except Exception as e:
        snapshot_error = f"Failed to save snapshot for {target} to {snapshot_dir}: {e}"
        logger.error(snapshot_error)
        # Non-fatal error, but report it

    # --- Update State ---
    # Use the derived state_key_target (e.g., "decision_logic_code")
    state[state_key_target] = code
    logger.info(f"Code string for '{target}' (key: '{state_key_target}') updated in state. Reload needed.")
    return {
        "success": True, # State was updated successfully
        "updated": True, # Signal code changed
        "snapshot_path": snapshot_path,
        "error": snapshot_error # Report only snapshot error if that occurred
    }


def evaluate(state: Dict[str, Any], env: BaseEnvironment, solver_fn: Optional[Callable], **kwargs) -> Dict[str, Any]:
    """Runs the current solver function and returns evaluation results."""
    logger.info("Executing: evaluate")
    task_start_time = time.time()

    if not solver_fn or not callable(solver_fn):
        err = "Evaluate tool called but solver_fn is not loaded/callable."
        logger.error(err)
        return {"success": False, "score": state.get('score', -1.0), "error": err, "duration": 0}

    try:
        task = env.get_task()
        logger.debug(f"Running solver for task: {task[:100]}...")
        # Assume solver_fn itself raises exceptions on internal errors
        output_dict = solver_fn(task_input=task)

        if not isinstance(output_dict, dict) or 'answer' not in output_dict:
             # Raise specific error for agent loop to catch if needed, or handle here
             raise ValueError("Solver failed result contract: did not return a dictionary with an 'answer' key.")

        answer = output_dict.get('answer', '') # Use .get for safety
        logger.debug(f"Solver returned answer: {str(answer)[:100]}")

        # Environment evaluates the answer
        score = env.evaluate(str(answer))

        duration = time.time() - task_start_time
        logger.info(f"Evaluation complete. Score: {score:.4f} (Duration: {duration:.2f}s)")
        # Return results; Agent loop will update state['score'], ['history'], etc.
        return {"success": True, "score": score, "duration": duration, "error": None}

    except Exception as e:
        logger.error(f"Error during solver execution or evaluation: {e}", exc_info=True)
        error_info = f"Evaluation Error: {type(e).__name__}: {e}\n{traceback.format_exc(limit=1)}"
        duration = time.time() - task_start_time
        # Return failure result; Agent loop updates state score to 0
        return {"success": False, "score": 0.0, "error": error_info, "duration": duration}


def display_analysis(state: Dict[str, Any], analysis: str, **kwargs) -> Dict[str, Any]:
    """Logs the analysis and returns it."""
    logger.info("Executing: display_analysis")
    if not isinstance(analysis, str):
        err = f"Invalid analysis format received (type: {type(analysis)})"
        logger.warning(err)
        # Agent loop can store this error in state['last_error'] if needed
        return {"success": False, "analysis_summary": None, "error": err}
    else:
        analysis_summary = analysis[:2000] # Limit stored length slightly more
        logger.info(f"Analysis Provided:\n------\n{analysis_summary}\n------")
        # Return summary; Agent loop stores it in state['last_analysis']
        return {"success": True, "analysis_summary": analysis_summary, "error": None}


def improve_decision_logic(state: Dict[str, Any], llm: OllamaInterface, analysis: str, **kwargs) -> Dict[str, Any]:
    """Asks LLM to improve decision logic, then uses adjust_logic logic."""
    logger.info("Executing: improve_decision_logic")
    if not isinstance(analysis, str) or not analysis.strip():
         err = "Missing or empty analysis parameter required"
         logger.error(err)
         return {"success": False, "updated": False, "error": err}

    logger.info(f"Analysis provided for improving decision logic: {analysis[:200]}...")

    # Construct prompt using current state info
    current_decision_code = state.get('decision_logic_code', '# Decision Code Missing') # Use correct key
    prompt = f"Improve the decision-making logic ('make_decision' function) based on the following analysis:\n"
    prompt += f"Analysis: {analysis}\n\n"
    prompt += f"Current Decision Code Snippet:\n```python\n{current_decision_code[:1000]}\n...\n```\n\n"
    prompt += "Provide the **complete, improved** Python code for the `make_decision(prompt_context: str, llm_interface)` function.\n"
    prompt += "Key requirements:\n"
    prompt += "- Signature: `make_decision(prompt_context: str, llm_interface)`\n"
    prompt += "- Input `prompt_context` includes goal and JSON state summary.\n"
    prompt += "- Must use `llm_interface.query_for_json()` and return a JSON list of tool calls.\n"
    prompt += "- Avoid forbidden imports (os, sys, etc.).\n"
    prompt += "Respond ONLY with the Python code block enclosed in ```python ... ``` markers. Ensure the code is runnable."

    logger.debug("Querying LLM for new decision logic code...")
    # Explicitly pass the llm interface instance to the tool
    if not isinstance(llm, OllamaInterface):
         return {"success": False, "updated": False, "error": "LLM interface object not provided correctly"}

    new_code = llm.query_for_code(prompt, temperature=0.6, context="Improve Decision Logic Code")

    if new_code and not new_code.startswith("Error:") and 'def make_decision' in new_code:
        logger.info("Received new decision logic code proposal. Attempting update via adjust_logic...")
        # Call adjust_logic *logic* (validation, snapshotting, state update)
        # Pass necessary args (state, target, code)
        update_result = adjust_logic(state, target='decision_logic', code=new_code) # Use correct target name

        # Check if adjust_logic itself failed (e.g., AST error in new code)
        if not update_result.get("success"):
             return update_result # Propagate the error dict from adjust_logic

        # SUCCESS: Report success from perspective of this tool
        return {
            "success": True,
            "updated": True, # Code was successfully updated in state
            "new_code_received": True,
            "error": update_result.get("error") # Include snapshot error if any
        }
    else:
        # FAILURE: LLM didn't provide good code
        err_msg = f"LLM failed to provide valid new decision logic code. Response prefix: {str(new_code)[:200]}..."
        logger.error(err_msg)
        return {"success": False, "updated": False, "new_code_received": False, "error": err_msg}


def analyze_failures(state: Dict[str, Any], llm: OllamaInterface, **kwargs) -> Dict[str, Any]:
     """Analyzes recent failures using LLM."""
     logger.info("Executing: analyze_failures")
     if not isinstance(llm, OllamaInterface):
          return {"success": False, "analysis_summary": None, "error": "LLM interface object not provided"}

     # Gather context from state
     recent_history = state.get('history', [])
     last_error = state.get('last_error', 'None')
     solver_snippet = state.get('solver_code','')[:500]
     decision_snippet = state.get('decision_logic_code','')[:500] # Use correct key

     analysis_context = (
         f"Recent Scores: {recent_history}\n"
         f"Last Error: {last_error}\n"
         f"Solver Code Snippet:\n{solver_snippet}...\n"
         f"Decision Code Snippet:\n{decision_snippet}..."
     )
     prompt = f"Analyze the recent agent performance based on the context provided.\n"
     prompt += f"Context:\n{analysis_context}\n\n"
     prompt += "What are the likely reasons for failure or low scores? Provide a concise analysis (max 3 sentences)."

     logger.debug("Querying LLM for failure analysis...")
     # Use standard query method for text analysis
     analysis_result = llm.query(prompt, temperature=0.7)

     if analysis_result.startswith("Error:"):
          summary = f"LLM error during failure analysis: {analysis_result}"
          logger.error(summary)
          return {"success": False, "analysis_summary": None, "error": summary}
     else:
          summary = analysis_result
          logger.info(f"Failure analysis result: {summary}")
          # Return analysis; Agent loop stores it in state['last_analysis']
          return {"success": True, "analysis_summary": summary, "error": None}


def tune_hyperparams(state: Dict[str, Any], param_name: str, new_value: Any, **kwargs) -> Dict[str, Any]:
     """Updates a hyperparameter directly in the agent's state dict."""
     logger.info(f"Executing: tune_hyperparams for '{param_name}' to '{new_value}'")
     # Define valid tunable params and potentially their types/ranges
     tunable_params = {
         "llm_temperature": float,
         "improvement_threshold": float,
         # Add others here, e.g., "max_stagnation": int
     }

     if param_name in tunable_params:
         expected_type = tunable_params[param_name]
         try:
             # Attempt casting to the expected type
             # Handle bool explicitly if needed
             if expected_type is bool and isinstance(new_value, str):
                  casted_value = new_value.lower() in ['true', '1', 'yes']
             else:
                  casted_value = expected_type(new_value)

             # Add specific validation ranges
             if param_name == 'llm_temperature' and not (0.0 <= casted_value <= 1.5):
                 raise ValueError("Temperature must be between 0.0 and 1.5")
             if param_name == 'improvement_threshold' and not (0.0 < casted_value < 0.5):
                  raise ValueError("Improvement threshold seems out of reasonable range (0<x<0.5)")

             state[param_name] = casted_value # Update state directly
             logger.info(f"Hyperparameter '{param_name}' updated to {casted_value} (Type: {expected_type.__name__}).")
             return {"success": True, "updated": True, "param_name": param_name, "new_value": casted_value, "error": None}

         except (ValueError, TypeError) as e:
             err = f"Invalid value '{new_value}' for hyperparameter '{param_name}' (expected {expected_type.__name__}): {e}"
             logger.error(err)
             return {"success": False, "updated": False, "error": err}
     else:
         err = f"Unknown or non-tunable hyperparameter: '{param_name}'. Allowed: {list(tunable_params.keys())}"
         logger.warning(err)
         return {"success": False, "updated": False, "error": err}