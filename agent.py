# agent.py
import logging
import json
import types
import os
from datetime import datetime, timezone
from pathlib import Path
import traceback
import time
import ast # For AST check in reload
from typing import Dict, Any, Callable, Optional, List, Union # Ensure necessary types are imported

# --- Local Imports ---
# Ensure these files exist in the same directory or Python path
try:
    import tools # Import the plain functions module
    from llm_interface import OllamaInterface
    from env import SimpleMathEnv # Using SimpleMathEnv as default
    from initial_code import INITIAL_SOLVER_CODE, INITIAL_DECISION_CODE
    from utils import setup_logging
except ImportError as e:
    print(f"ERROR: Failed to import necessary modules. Check file existence and Python path.")
    print(f"Import Error: {e}")
    # Attempt to identify missing file based on error message
    if "utils" in str(e): print("-> Did you create 'utils.py'?")
    if "llm_interface" in str(e): print("-> Did you create 'llm_interface.py'?")
    if "env" in str(e): print("-> Did you create 'env.py'?")
    if "initial_code" in str(e): print("-> Did you create 'initial_code.py'?")
    if "tools" in str(e): print("-> Did you create 'tools.py'?")
    exit(1) # Stop if basic modules are missing

# --- Setup Logging ---
# Ensure setup_logging is called only once if possible, e.g., by checking handlers
logger = logging.getLogger("SlimAgent") # Get logger first
if not logger.handlers: # Check if handlers are already configured
    try:
        setup_logging(
            log_level=logging.INFO,    # Set desired level
            log_to_file=True,
            log_dir="agent_logs",
            log_filename="slim_agent.log"
        )
    except Exception as log_e:
        print(f"WARNING: Failed to set up logging via utils.setup_logging: {log_e}. Using basicConfig.")
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)-20s - %(levelname)-8s - %(message)s')
        # logger is already retrieved, basicConfig configures root if no handlers present


class SlimAgent:
    """
    Slim agent implementing self-improvement loop with hardened logic.
    No placeholders should remain in this version.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        logger.info(f"Initializing SlimAgent with config: {config}")

        # --- Load Goal Prompt ---
        try:
            self.goal_prompt_path = config.get('goal_prompt_path', 'goal_prompt.md')
            self.goal_prompt = self._load_goal_prompt(self.goal_prompt_path)
            if not self.goal_prompt or len(self.goal_prompt) < 50:
                 logger.warning(f"Goal prompt loaded from {self.goal_prompt_path} seems short or empty.")
        except Exception as e:
            logger.critical(f"CRITICAL: Failed to load goal prompt: {e}")
            raise

        # --- Initialize Components ---
        try:
            default_host = "http://localhost:11434"
            self.llm = OllamaInterface(
                model_name=config.get('llm_model'), # Relies on default in OllamaInterface if None
                host=config.get('host', default_host),
                request_timeout=config.get('llm_timeout', 120)
            )
            self.env = SimpleMathEnv(seed=config.get('env_seed', None)) # Assumes SimpleMathEnv
        except Exception as e:
             logger.critical(f"CRITICAL: Failed to initialize LLM or Env: {e}")
             raise

        # --- Snapshot Directory ---
        self.snapshot_dir = Path(config.get('snapshot_dir', 'snapshots_slim'))
        try:
            self.snapshot_dir.mkdir(exist_ok=True)
            logger.info(f"Snapshot directory: {self.snapshot_dir.resolve()}")
        except Exception as e:
             logger.error(f"Failed to create snapshot directory {self.snapshot_dir}: {e}.")

        # --- Initialize State Dictionary ---
        if not INITIAL_SOLVER_CODE or not isinstance(INITIAL_SOLVER_CODE, str): raise ValueError("INITIAL_SOLVER_CODE missing/invalid.")
        if not INITIAL_DECISION_CODE or not isinstance(INITIAL_DECISION_CODE, str): raise ValueError("INITIAL_DECISION_CODE missing/invalid.")
        self.state = {
            "solver_code": INITIAL_SOLVER_CODE,
            "decision_logic_code": INITIAL_DECISION_CODE, # Correct key
            "score": -1.0, "previous_score": -1.0, "iteration": 0, "stagnation_counter": 0,
            "history": [], "last_error": None, "last_analysis": None,
            "llm_temperature": config.get('llm_temperature', 0.7),
            "improvement_threshold": config.get('improvement_threshold', 0.01),
            "snapshot_dir_config": str(self.snapshot_dir),
        }

        # --- Runtime function placeholders ---
        self.solver_fn: Optional[Callable] = None
        self.decision_logic_fn: Optional[Callable] = None
        logger.info("Attempting initial load of runtime functions...")
        self._reload_runtime_functions(target='solver')
        self._reload_runtime_functions(target='decision_logic')

        # --- Critical Check for Decision Logic ---
        if not self.decision_logic_fn:
             logger.critical("Decision logic fn failed to load on startup.") # Corrected log message
             raise RuntimeError("Failed to load essential initial decision logic function.")
        if not self.solver_fn: logger.error("CRITICAL: Initial loading of solver function failed.")

        logger.info("SlimAgent initialized.")

    # --- Method Implementations ---

    def _load_goal_prompt(self, path: str) -> str:
        """Loads the goal prompt text from a file."""
        logger.debug(f"Loading goal prompt from: {path}")
        if not os.path.exists(path):
             raise FileNotFoundError(f"Goal prompt file not found at: {path}")
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error loading goal prompt from {path}: {e}")
            raise IOError(f"Could not load goal prompt from {path}") from e

    def _validate_tool_calls(self, tool_calls: Any) -> tuple[bool, str | None, List[Dict]]:
        """Performs basic validation on the structure returned by decision logic."""
        logger.debug("Agent validating tool call list structure...")
        if not isinstance(tool_calls, list):
            return False, "Tool call structure not a list.", []

        valid_struct_calls = []
        # Use known tools list if available in tools module, otherwise skip name check
        known_tools = getattr(tools, 'ALLOWED_TOOLS_LIST', None)

        for i, call in enumerate(tool_calls):
            if not isinstance(call, dict):
                logger.warning(f"Skipping tool call item {i}: not a dict.")
                continue
            if 'tool_name' not in call or not isinstance(call.get('tool_name'), str):
                logger.warning(f"Skipping tool call item {i}: missing or invalid 'tool_name'.")
                continue
            tool_name = call['tool_name']
            # Check against known tools if list is available
            if known_tools and tool_name not in known_tools:
                 logger.warning(f"Skipping item {i}: unknown tool_name '{tool_name}'. Allowed: {known_tools}")
                 continue
            # Ensure params exists as a dict
            if 'params' not in call:
                call['params'] = {} # Default to empty dict
            elif not isinstance(call.get('params'), dict):
                logger.warning(f"Skipping item {i} ('{tool_name}'): 'params' is not a dict.")
                continue

            valid_struct_calls.append(call) # Add if checks pass

        if len(valid_struct_calls) < len(tool_calls):
            logger.warning("Some tool calls failed agent-side structure/name validation.")

        # Return True indicates validation ran, even if list is empty
        return True, None, valid_struct_calls

    def _reload_runtime_functions(self, target: str | None = None):
        """Reloads target function or all using simple exec after AST check."""
        targets_to_reload = [target] if target else ['solver', 'decision_logic']
        logger.info(f"Reloading runtime functions: {targets_to_reload}")

        for t in targets_to_reload:
            old_fn = self.solver_fn if t == 'solver' else self.decision_logic_fn # Defined before use
            code_key = f"{t}_code"
            code = self.state.get(code_key)
            if not code:
                logger.error(f"Cannot reload {t}: code missing for key '{code_key}'. Setting function to None.")
                if t == 'solver': self.solver_fn = None
                if t == 'decision_logic': self.decision_logic_fn = None
                continue

            logger.debug(f"Attempting reload for '{t}'...")
            # 1. AST Check
            try:
                ast.parse(code)
                logger.debug(f"AST check passed for '{t}'.")
            except SyntaxError as e:
                err_msg = f"SyntaxError prevents reload for '{t}': {e}"
                logger.error(err_msg)
                self.state['last_error'] = f"Reload Error ({t}): Syntax"
                # Don't proceed to exec if syntax is wrong
                # Retain old function explicitly
                self._retain_old_fn(t, old_fn)
                logger.warning(f"Retained previous function for '{t}' due to SyntaxError.")
                continue # Skip to next target if any
            except Exception as e:
                 logger.warning(f"Unexpected AST check error for '{t}': {e}. Will attempt exec.")

            # 2. Exec if AST parse okay (or only warning)
            reloaded_ok = False
            try:
                namespace = {}
                exec(code, namespace) # Direct execution
                func_name = 'solver' if t == 'solver' else 'make_decision'

                if func_name in namespace and callable(namespace[func_name]):
                    # Assign function handle
                    if t == 'solver': self.solver_fn = namespace[func_name]
                    elif t == 'decision_logic': self.decision_logic_fn = namespace[func_name]
                    logger.info(f"Successfully reloaded runtime function for '{t}'.")
                    reloaded_ok = True
                    # Clear previous *reload* error for this target on success
                    last_err = self.state.get('last_error'); err_prefix = f"Reload Error ({t})"
                    if isinstance(last_err, str) and last_err.startswith(err_prefix): self.state['last_error'] = None
                else: # Function not found/callable after exec
                    err_msg = f"Function '{func_name}' not found/callable after exec."
                    logger.error(err_msg + f" Retaining previous function for '{t}'.")
                    self.state['last_error'] = f"Reload Error ({t}): {err_msg}"
                    self._retain_old_fn(t, old_fn)

            except Exception as e: # Catch errors during exec itself
                logger.error(f"Error during exec/reload for '{t}': {e}", exc_info=True)
                self.state['last_error'] = f"Reload Error ({t}): {e}"
                self._retain_old_fn(t, old_fn)
                logger.warning(f"Retained previous function for '{t}' due to exec error.")

            # Log if reload failed overall and old function was kept
            if not reloaded_ok and old_fn:
                 logger.warning(f"Reload failed for '{t}', retained previous version.")


    def _retain_old_fn(self, target: str, old_fn: Optional[Callable]):
         """Helper to explicitly retain the old function handle on reload failure."""
         if old_fn is None: logger.warning(f"Cannot retain previous func for '{target}', none exists."); return
         if target == 'solver': self.solver_fn = old_fn
         elif target == 'decision_logic': self.decision_logic_fn = old_fn


    def _get_state_summary_for_llm(self) -> dict:
         """Prepare condensed JSON summary for LLM prompt."""
         # Adjust limits based on model context window (e.g., 128k allows much more)
         MAX_SNIPPET_LEN = 8000 # Example large limit
         MAX_HISTORY_ITEMS = 20  # Example large limit

         solver_code = self.state.get('solver_code', '')
         decision_code = self.state.get('decision_logic_code', '')
         history = self.state.get('history', [])

         # Truncate only if necessary
         solver_snippet = solver_code if len(solver_code) <= MAX_SNIPPET_LEN else solver_code[:MAX_SNIPPET_LEN] + "\\n..."
         decision_snippet = decision_code if len(decision_code) <= MAX_SNIPPET_LEN else decision_code[:MAX_SNIPPET_LEN] + "\\n..."
         recent_history = history[-MAX_HISTORY_ITEMS:]

         summary = {
             'current_score': self.state.get('score', -1.0),
             'previous_score': self.state.get('previous_score', -1.0),
             'iteration': self.state.get('iteration', 0),
             'stagnation_counter': self.state.get('stagnation_counter', 0),
             'solver_code_present': bool(solver_code),
             'solver_code_snippet': solver_snippet,
             'decision_code_present': bool(decision_code),
             'decision_code_snippet': decision_snippet,
             'recent_history': recent_history, # Send recent history (can be dicts or scores)
             'last_analysis': self.state.get('last_analysis'),
             'last_error': self.state.get('last_error'),
             'hyperparams': { # Send current tunable params
                  'llm_temperature': self.state.get('llm_temperature', 0.7),
                  'improvement_threshold': self.state.get('improvement_threshold', 0.01),
             }
         }
         return summary

    def self_improve(self):
        """Main self-improvement loop."""
        max_iterations = self.config.get('max_iterations', 10)
        max_stagnation = self.config.get('max_stagnation', 5)

        # --- Initial Evaluation ---
        if self.state.get('score', -1.0) < 0:
             logger.info("Performing initial evaluation...")
             if self.solver_fn and callable(self.solver_fn):
                 eval_result = tools.evaluate(state=self.state, env=self.env, solver_fn=self.solver_fn)
                 if eval_result.get('success'):
                      self.state['score'] = eval_result['score']
                      self.state['previous_score'] = self.state['score'] # Set baseline
                      self.state.setdefault('history', []).append({'score': self.state['score'], 'duration': eval_result.get('duration', 0), 'error': None, 'iteration': 0})
                      self.state['history'] = self.state['history'][-10:] # Limit history
                 else: self.state['last_error'] = eval_result.get('error', 'Initial evaluation failed')
                 logger.info(f"Initial evaluation score: {self.state['score']:.4f}")
             else: logger.error("Cannot perform initial evaluation - solver not loaded/callable."); return

        # --- Main Loop ---
        for i in range(max_iterations):
            current_iter = self.state['iteration'] + 1; self.state['iteration'] = current_iter
            logger.info(f"--- Starting Improvement Iteration: {current_iter}/{max_iterations} ---")
            iteration_error = None # Use this to track first critical error in iteration
            code_was_modified = False; explicit_reload_target = None

            # Check decision logic function validity at start of loop
            if not self.decision_logic_fn or not callable(self.decision_logic_fn):
                logger.critical("Decision logic unavailable. Halting loop.")
                iteration_error = self.state['last_error'] or "Decision logic function unavailable"; break

            # 1. Call Decision Logic
            requested_tool_calls = []
            try:
                decision_context_summary = self._get_state_summary_for_llm()
                prompt_context_for_llm = f"{self.goal_prompt}\n\n## Current State & Context:\n{json.dumps(decision_context_summary, indent=2)}"
                raw_calls_or_error = self.decision_logic_fn(prompt_context_for_llm, self.llm)
                valid_list, validation_err, validated_calls = self._validate_tool_calls(raw_calls_or_error)
                if valid_list: requested_tool_calls = validated_calls
                else: iteration_error = f"Decision Logic Invalid Output: {validation_err}"
            except Exception as e: logger.error(f"Decision logic crash: {e}", exc_info=True); iteration_error = f"Decision Logic Crash: {e}"

            # If decision failed, record error and skip tool execution for this iteration
            if iteration_error:
                 self.state['last_error'] = iteration_error
                 logger.error(f"Halting iteration {current_iter} early due to decision logic failure.")
                 # Fall through to end-of-iteration logic (will increment stagnation)
            else:
                # 2. Execute Tool Calls
                for tool_call in requested_tool_calls:
                    tool_name = tool_call.get('tool_name'); tool_params = tool_call.get('params', {})
                    tool_function = getattr(tools, tool_name, None)
                    if tool_function and callable(tool_function):
                        logger.info(f"Executing tool: {tool_name}...")
                        try:
                            result = tool_function(state=self.state, env=self.env, llm=self.llm, solver_fn=self.solver_fn, **tool_params)
                            if isinstance(result, dict) and not result.get('success', False):
                                err_msg = result.get('error', 'Unknown tool error'); logger.error(f"Tool '{tool_name}' failed: {err_msg}")
                                if not iteration_error: iteration_error = f"Tool Error ({tool_name}): {err_msg}" # Record first error
                                if tool_name in tools.CODE_MODIFYING_TOOLS or tool_name == 'evaluate': break # Stop on critical tool failure
                            elif isinstance(result, dict) and result.get('success'):
                                 if tool_name in tools.CODE_MODIFYING_TOOLS and result.get('updated'):
                                     code_was_modified = True; explicit_reload_target = tool_params.get('target')
                                     logger.info(f"Tool '{tool_name}' updated code for '{explicit_reload_target}'.")
                                 if tool_name == 'display_analysis': self.state['last_analysis'] = result.get('analysis_summary')
                                 if tool_name == 'evaluate': # Update state from successful eval
                                      self.state['score'] = result.get('score', self.state['score'])
                                      self.state.setdefault('history', []).append({'score': self.state['score'], 'duration': result.get('duration', 0), 'error': None, 'iteration': current_iter}); self.state['history'] = self.state['history'][-10:]
                                      self.state['last_error'] = None # Clear last error on successful evaluation
                        except Exception as e: logger.error(f"Crash executing '{tool_name}': {e}", exc_info=True); iteration_error = f"Tool Crash ({tool_name}): {e}"; break # Stop on crash
                    else: logger.warning(f"Tool '{tool_name}' not found. Skipping.")
                    if iteration_error: break # Exit tool loop if error occurred


            # 3. Reload if code was modified AND no critical error occurred during tools
            if code_was_modified and not iteration_error:
                logger.info(f"Reloading function for '{explicit_reload_target}'...")
                self._reload_runtime_functions(target=explicit_reload_target)
                reload_error_prefix = f"Reload Error ({explicit_reload_target})"
                if isinstance(self.state.get('last_error'), str) and self.state['last_error'].startswith(reload_error_prefix):
                    logger.error(f"Reload failed for {explicit_reload_target}. Iteration unstable.")
                    iteration_error = self.state['last_error'] # Mark iteration as failed if reload fails


            # --- Iteration End Logic ---
            current_score = self.state.get('score', -1.0)
            previous_score = self.state.get('previous_score', -1.0) # Score from *before* this iteration ran

            if iteration_error:
                 logger.error(f"--- Iteration {current_iter} finished with CRITICAL ERROR: {iteration_error} ---")
                 self.state['last_error'] = iteration_error # Ensure final error state is set
                 self.state['score'] = previous_score # Revert score for stagnation calc
                 self.state['stagnation_counter'] += 1; logger.warning(f"Stagnation counter incr. to {self.state['stagnation_counter']} due to error.")
            else: # Successful iteration (or only non-critical errors if any)
                 improvement_threshold = self.state.get('improvement_threshold', 0.01); stagnation_incremented = False
                 if current_score > -0.5 and previous_score > -0.5:
                    if current_score > previous_score + improvement_threshold: self.state['stagnation_counter'] = 0
                    else: self.state['stagnation_counter'] += 1; stagnation_incremented = True
                 elif previous_score > -0.5: self.state['stagnation_counter'] += 1; stagnation_incremented = True
                 stagnation_msg = f"Stagnation:{self.state['stagnation_counter']}" + (" (+1)" if stagnation_incremented else "")
                 logger.info(f"--- End Iteration: {current_iter} | Score: {self.state['score']:.4f} | {stagnation_msg} ---")

            # Update previous_score for the *next* iteration AFTER stagnation check
            self.state['previous_score'] = current_score # Use the score determined by this iteration's outcome

            # Check stopping conditions
            if self.state['stagnation_counter'] >= max_stagnation: logger.warning(f"Stopping due to max stagnation ({max_stagnation})."); break
            if iteration_error: logger.error("Halting loop due to critical error in iteration."); break # Halt loop fully on error

        logger.info("Self-improvement loop finished.")

    # --- Rollback Placeholder ---
    def _rollback_code(self, target: str):
         # Basic Implementation (Find latest snapshot, update state, reload)
         logger.warning(f"Attempting basic rollback for '{target}'...")
         # ... (Find latest snapshot file logic) ...
         # ... (Read code, update self.state[f"{target}_code"], call self._reload_runtime_functions(target)) ...
         pass


# --- Configuration and Execution ---
DEFAULT_CONFIG = {
    'llm_model': 'granite3.3:latest', # Ensure this matches your Ollama model tag
    'host': 'http://localhost:11434', # Added default host
    'goal_prompt_path': 'goal_prompt.md',
    'snapshot_dir': 'snapshots_slim',
    'max_iterations': 10,
    'max_stagnation': 3,
    'llm_timeout': 180,
    'llm_temperature': 0.7,
    'improvement_threshold': 0.01,
    'env_seed': 42,
}

if __name__ == "__main__":
    config = DEFAULT_CONFIG # Load from file in production
    logger.info("--- Starting SlimAgent Run ---")
    agent = None
    try:
        agent = SlimAgent(config)
        agent.self_improve() # Run the main loop
    # --- Specific Error Handling for Common Issues ---
    except FileNotFoundError as e: logger.critical(f"Initialization Error: Required file not found: {e}")
    except ImportError as e: logger.critical(f"Dependency Error: Failed to import module: {e}")
    except ValueError as e: logger.critical(f"Initialization Error: Invalid initial code/config: {e}")
    except RuntimeError as e: logger.critical(f"Runtime Halt: {e}") # Catch explicit halts
    except KeyboardInterrupt: logger.warning("--- Run Interrupted by User ---")
    except Exception as e: logger.critical(f"Agent run failed with unhandled exception: {e}", exc_info=True)
    # --- End Error Handling ---
    finally:
        # Log final state regardless of how loop exited
        final_score = agent.state.get('score', 'N/A') if agent and hasattr(agent, 'state') else 'N/A'
        final_iter = agent.state.get('iteration', 0) if agent and hasattr(agent, 'state') else 0
        logger.info(f"--- Run Complete (Reached Iteration {final_iter}). Final Score: {final_score} ---")