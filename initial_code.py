# initial_code.py
import logging
import json
# Ensure necessary types are imported AT THE MODULE LEVEL for type checking if running standalone
from typing import Dict, Any, List, Union

# --- Initial Solver Code (Using r""" for raw string literal) ---
INITIAL_SOLVER_CODE = r"""
# Imports needed for the solver logic (inside the string)
import logging
import re
# import ast # Not used in this version

# Logger for the solver runtime namespace
# Note: logger setup needs to happen in the main script before this is exec'd
# ThisgetLogger call will work if logging is already configured.
logger_solver = logging.getLogger('agent_solver_runtime')

def solver(task_input: str) -> dict:
    '''
    Initial naive solver: Tries to find two integers and sum them.
    Returns result in the required dictionary format {'answer': ...}.
    Includes basic robustness and correct regex pattern.
    '''
    # Add basic check for logger availability in exec environment
    if 'logger_solver' not in globals() or not hasattr(logger_solver, 'info'):
         # Fallback print if logger isn't set up in exec context
         print(f"Solver received task (logger unavailable): {task_input[:100]}...")
         log_info = lambda msg: print(f"SOLVER INFO: {msg}")
         log_debug = lambda msg: print(f"SOLVER DEBUG: {msg}")
         log_warning = lambda msg: print(f"SOLVER WARNING: {msg}")
         log_error = lambda msg, exc_info=False: print(f"SOLVER ERROR: {msg}")
    else:
         log_info = logger_solver.info
         log_debug = logger_solver.debug
         log_warning = logger_solver.warning
         log_error = logger_solver.error

    log_info(f"Initial solver received task: {task_input[:100]}...")
    final_answer = "Error: Calculation failed." # Default error answer
    try:
        # Use raw string r'' for regex pattern - embedded correctly now
        numbers = [int(n) for n in re.findall(r'-?\d+', task_input)]
        log_debug(f"Found numbers: {numbers}")

        if len(numbers) >= 2:
            # Simple sum of the first two numbers found
            result = numbers[0] + numbers[1]
            final_answer = str(result)
            log_info(f"Initial solver calculated sum: {result}")
        else:
            log_warning("Initial solver couldn't find at least two integers.")
            final_answer = "Error: Not enough integers found."

    except Exception as e:
        log_error(f"Exception in initial solver: {e}", exc_info=False) # Keep log concise
        final_answer = f"Error: Exception during calculation - {type(e).__name__}"

    # Return required structure with answer as string
    return {"answer": str(final_answer)}

""" # End of INITIAL_SOLVER_CODE string


# --- Initial Decision Logic Code ---
INITIAL_DECISION_CODE = """
import logging
import json
import re
# --- FIX: Added 'Any' to typing import ---
from typing import Dict, Any, List, Union
# --- END FIX ---
import ast # Needed for compile check in param validation

# Logger for the decision logic runtime
logger_decision = logging.getLogger('agent_decision_runtime')

# --- Robust JSON Context Parser ---
def parse_context_from_prompt_context(prompt_context: str) -> Dict[str, Any]:
    \"\"\"Extracts JSON dict embedded in prompt_context after marker.\"\"\"
    # Add basic check for logger availability
    if 'logger_decision' not in globals() or not hasattr(logger_decision, 'warning'):
        log_warning = lambda msg: print(f"DECISION WARNING: {msg}")
        log_error = lambda msg: print(f"DECISION ERROR: {msg}")
        log_debug = lambda msg: print(f"DECISION DEBUG: {msg}")
    else:
        log_warning = logger_decision.warning
        log_error = logger_decision.error
        log_debug = logger_decision.debug

    marker = "## Current State & Context:"
    idx = prompt_context.find(marker)
    if idx == -1: log_warning("Context marker not found."); return {}
    start = -1; str_after_marker = prompt_context[idx + len(marker):]
    first_brace = str_after_marker.find('{'); first_square = str_after_marker.find('[')
    if first_brace != -1 and (first_square == -1 or first_brace < first_square): start = idx + len(marker) + first_brace
    elif first_square != -1: log_warning("Found '[' before '{'. State summary should be dict."); return {}
    if start == -1: log_warning("No JSON start ('{') found after marker."); return {}
    open_ch, close_ch = ('{', '}') # Expecting dict
    balance = 0; end = -1
    for i, ch in enumerate(prompt_context[start:], start):
        if ch == open_ch: balance += 1
        elif ch == close_ch: balance -= 1;
        if balance == 0: end = i + 1; break
    if end == -1:
       log_warning(f"No matching closing '{close_ch}' found.")
       return {}
    json_str = prompt_context[start:end]
    try:
        parsed_json = json.loads(json_str)
        if not isinstance(parsed_json, dict): log_warning(f"Parsed context not dict ({type(parsed_json)})."); return {}
        log_debug("Parsed context JSON successfully.")
        return parsed_json
    except Exception as e: log_error(f"Failed parsing context JSON: {e}. Substr: {json_str[:100]}..."); return {}
# --- End Parser ---

# --- Tool Validation Helpers ---
def is_valid_tool_call_structure(call: dict) -> bool:
    \"\"\"Checks basic structure: dict with tool_name string and params dict.\"\"\"
    if not isinstance(call, dict): logger_decision.debug("Structure fail: Not dict"); return False
    if 'tool_name' not in call or not isinstance(call.get('tool_name'), str): logger_decision.debug("Structure fail: Bad tool_name"); return False
    if 'params' not in call: call['params'] = {}
    elif not isinstance(call.get('params'), dict): logger_decision.debug("Structure fail: Bad params"); return False
    return True

def validate_tool_params(call: dict) -> bool:
    \"\"\"Checks REQUIRED parameters and basic types/validity for specific tools.\"\"\"
    tool_name = call.get('tool_name'); params = call.get('params', {})
    # Known tools and their required parameters + types (use Any for values checked by tool)
    required_params_map = {
        'adjust_logic': {'target': str, 'code': str},
        'improve_decision_logic': {'analysis': str},
        'display_analysis': {'analysis': str},
        'tune_hyperparams': {'param_name': str, 'new_value': Any}
    }
    if tool_name in required_params_map:
        required = required_params_map[tool_name]
        for param, expected_type in required.items():
             if param not in params: logger_decision.warning(f"Param fail '{tool_name}': missing '{param}'."); return False
             param_value = params.get(param)
             if expected_type is not Any and not isinstance(param_value, expected_type): logger_decision.warning(f"Param fail '{tool_name}': '{param}' wrong type (exp {expected_type.__name__})."); return False
             if expected_type is str and not param_value.strip(): logger_decision.warning(f"Param fail '{tool_name}': required str '{param}' empty."); return False
             # Specific check for 'adjust_logic' code param
             if tool_name == 'adjust_logic' and param == 'code':
                 code_val = param_value.strip()
                 if len(code_val) < 10: logger_decision.warning(f"Param fail '{tool_name}': 'code' too short."); return False
                 try:
                     ast.parse(code_val); logger_decision.debug(f"Code param for '{tool_name}' passed compile check.") # Use built-in ast
                 except Exception as e: logger_decision.warning(f"Param fail '{tool_name}': code compile error: {e}"); return False
    # Passed all required checks for this tool
    return True
# --- End Validation Helpers ---

EXPECTED_TOOL_SCHEMA_INFO = ''' JSON: `[ {"tool_name": "...", "params": {...}} ]` OR `{"tool_name": "...", "params": {...}}` '''

def make_decision(prompt_context: str, llm_interface) -> List[Dict[str, Any]]:
    ''' Parses context, requests JSON, validates structure & PARAMS, uses fallback. '''
    logger_decision.info("Executing initial decision logic...")
    # Parse context
    context_data = parse_context_from_prompt_context(prompt_context)
    current_score = context_data.get('current_score', -1.0)
    score_threshold = context_data.get('hyperparams', {}).get('improvement_threshold', 0.95)
    last_error_from_state = context_data.get('last_error')
    logger_decision.debug(f"Context: Score:{current_score:.2f}, Thresh:{score_threshold:.2f}, LastErr:{bool(last_error_from_state)}")

    # Construct prompt
    llm_prompt = prompt_context + "\\n\\n## Action Request\\nAnalyze state/goal. Decide tools.\\n"
    if current_score < score_threshold: llm_prompt += f"Score low (<{score_threshold:.2f}). ACTION: Improve 'solver' via 'adjust_logic'(target='solver', code='<code>'), then 'evaluate'.\\n"
    else: llm_prompt += f"Score good (>= {score_threshold:.2f}). ACTION: 'self_inspect', then 'evaluate'.\\n"
    llm_prompt += f"Response MUST be ONLY valid JSON. Schema: {EXPECTED_TOOL_SCHEMA_INFO}\\n REQUIRED PARAMS: 'adjust_logic': ['target','code'] etc. NO commentary."

    # Call LLM
    logger_decision.info("Querying LLM for tool calls...")
    response = llm_interface.query_for_json(prompt=llm_prompt, temperature=0.4, retries=1)

    # Process Response
    structurally_valid_calls = []
    llm_query_failed = False; llm_error_msg = "LLM response OK."
    raw_calls_to_process = []
    if isinstance(response, list): raw_calls_to_process = response
    elif isinstance(response, dict): raw_calls_to_process = [response]
    elif isinstance(response, str) and response.startswith("Error:"): llm_query_failed = True; llm_error_msg = response
    else: llm_query_failed = True; llm_error_msg = f"LLM unexpected type: {type(response)}"

    if not llm_query_failed:
        structurally_valid_calls = [call for call in raw_calls_to_process if is_valid_tool_call_structure(call)]
        if len(structurally_valid_calls) < len(raw_calls_to_process): logger_decision.warning("Some raw calls failed structure validation.")
        if not structurally_valid_calls and raw_calls_to_process: llm_query_failed = True; llm_error_msg = "All raw calls failed structure validation."

    # Apply Fallback OR Parameter Validation
    final_tool_calls_list = []
    fallback_actions = [{"tool_name": "analyze_failures" if last_error_from_state else "self_inspect", "params": {}}, {"tool_name": "evaluate", "params": {}}]

    if llm_query_failed:
        logger_decision.warning(f"Using fallback logic: {llm_error_msg}")
        final_tool_calls_list = fallback_actions
    else:
        # Parameter Validation loop
        param_validated_calls = [call for call in structurally_valid_calls if validate_tool_params(call)]
        if len(param_validated_calls) < len(structurally_valid_calls): logger_decision.warning("Some calls failed parameter validation.")

        # Use fallback if all failed validation OR if LLM provided empty list initially
        if not param_validated_calls:
            if structurally_valid_calls: # Had valid structures, but params failed
                 logger_decision.warning("All valid structure calls failed param validation. Using fallback.")
            else: # LLM returned [] or only invalid structures
                 logger_decision.info("LLM provided no valid calls. Performing fallback evaluation.")
            final_tool_calls_list = fallback_actions
        else: # Param validation passed for at least one call
             final_tool_calls_list = param_validated_calls

    # Final Polish: Ensure self_inspect is first
    if final_tool_calls_list and (final_tool_calls_list[0].get('tool_name') != 'self_inspect'):
         logger_decision.debug("Prepending 'self_inspect' to start.")
         final_tool_calls_list.insert(0, {"tool_name": "self_inspect", "params": {}})

    logger_decision.info(f"Returning {len(final_tool_calls_list)} final validated tool calls: {[t.get('tool_name') for t in final_tool_calls_list]}")
    return final_tool_calls_list

""" # End of INITIAL_DECISION_CODE string