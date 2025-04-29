#!/usr/bin/env python3
# llm_interface.py

import requests
import json
import logging
import re
import time
from typing import Optional, Union, Dict, List, Any, Tuple

logger = logging.getLogger("llm_interface")


class OllamaInterface:
    """
    Interface for communicating with a local Ollama server.
    Supports large context windows, JSON/code extraction, retries, and robust error handling.
    """

    DEFAULT_HOST = "http://localhost:11434"
    DEFAULT_MODEL = "granite3.3:latest"
    DEFAULT_NUM_CTX = 131072  # 128 Ki tokens

    def __init__(
        self,
        model_name: Optional[str] = None,
        host: Optional[str] = None,
        request_timeout: int = 180,
        default_num_ctx: int = DEFAULT_NUM_CTX,
        test_connection_on_init: bool = True,
    ):
        # Host
        used_host = host or self.DEFAULT_HOST
        if not isinstance(used_host, str) or not used_host.startswith(("http://", "https://")):
            logger.warning(f"Invalid host ('{host}'). Falling back to {self.DEFAULT_HOST}.")
            used_host = self.DEFAULT_HOST
        self.host = used_host.rstrip("/")

        # Model
        used_model = model_name or self.DEFAULT_MODEL
        if not isinstance(used_model, str) or not used_model.strip():
            logger.warning(f"Invalid model ('{model_name}'). Falling back to {self.DEFAULT_MODEL}.")
            used_model = self.DEFAULT_MODEL
        self.model_name = used_model

        # API URL
        self.api_url = f"{self.host}/api/generate"
        self.request_timeout = request_timeout

        # Context window
        if not isinstance(default_num_ctx, int) or default_num_ctx <= 0:
            logger.warning(f"Invalid default_num_ctx ({default_num_ctx}). Using {self.DEFAULT_NUM_CTX}.")
            self.default_num_ctx = self.DEFAULT_NUM_CTX
        else:
            self.default_num_ctx = default_num_ctx

        logger.info(
            f"Initialized OllamaInterface(model={self.model_name}, api_url={self.api_url}, "
            f"timeout={self.request_timeout}s, num_ctx={self.default_num_ctx})"
        )

        if test_connection_on_init:
            if not self._test_connection():
                logger.warning("Failed initial connection test to Ollama server.")

    def _test_connection(self) -> bool:
        """Checks that the Ollama server is reachable and lists available models."""
        try:
            r = requests.get(self.host + "/", timeout=5)
            if r.status_code != 200:
                logger.warning(f"Ollama base URL returned {r.status_code}.")
                return False

            # Check /api/tags for model list
            tags = requests.get(f"{self.host}/api/tags", timeout=5).json().get("models", [])
            names = [m.get("name") for m in tags]
            if any(self.model_name.split(":")[0] == n.split(":")[0] for n in names):
                logger.info(f"Model '{self.model_name}' found in Ollama tags.")
            else:
                logger.warning(f"Model '{self.model_name}' not among {names}.")
            return True

        except Exception as e:
            logger.error(f"Ollama connection test error: {e}")
            return False

    def _strip_think_tags(self, text: str) -> str:
        """Remove any <think>…</think> blocks before JSON extraction."""
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()

    def _extract_json_block(self, text: str) -> Union[str, Tuple[None, str]]:
        """
        Try fenced ```json``` first, then fall back to matching outermost {…} or […].
        Returns JSON string or (None, error_message).
        """
        # fenced
        m = re.search(r"```json\s*(\{.*?\}|\[.*?\])\s*```", text, flags=re.DOTALL | re.IGNORECASE)
        if m:
            candidate = m.group(1).strip()
            try:
                json.loads(candidate)
                return candidate
            except Exception as e:
                logger.warning(f"Fenced JSON parse failed: {e}")

        # brace/bracket fallback
        first_obj = min(
            (i for i in (text.find("{"), text.find("[")) if i != -1),
            default=-1
        )
        if first_obj == -1:
            return None, "No JSON start marker found."

        # find matching end char by simple bracket counting
        open_ch = text[first_obj]
        close_ch = "}" if open_ch == "{" else "]"
        balance = 0
        for idx in range(first_obj, len(text)):
            if text[idx] == open_ch:
                balance += 1
            elif text[idx] == close_ch:
                balance -= 1
                if balance == 0:
                    candidate = text[first_obj : idx + 1]
                    try:
                        json.loads(candidate)
                        return candidate
                    except Exception as e:
                        return None, f"Extracted block parse failed: {e}"
        return None, "No matching closing bracket found."

    def _extract_code_block(self, text: str, language: str = "python") -> Union[str, Tuple[None, str]]:
        """
        Extract first fenced code block for given language, 
        fall back to generic ```…``` if none found.
        """
        # language-specific
        pattern = rf"```{language}\s*(.*?)\s*```"
        m = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)
        if m and m.group(1).strip():
            return m.group(1).strip()

        # fallback generic
        m2 = re.search(r"```\s*(.*?)\s*```", text, flags=re.DOTALL)
        if m2 and m2.group(1).strip():
            return m2.group(1).strip()

        return None, f"No ```{language}``` or generic code block found."

    def query(
        self,
        prompt: str,
        temperature: float = 0.7,
        num_predict: int = 1536,
        format: str = "",
        system_prompt: str = "",
        num_ctx: Optional[int] = None,
    ) -> str:
        """
        Send a raw prompt, return the generated text or an "Error: …" string.
        """
        ctx = num_ctx or self.default_num_ctx
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": num_predict, "num_ctx": ctx},
        }
        if system_prompt:
            payload["system"] = system_prompt
        if format:
            payload["format"] = format

        try:
            start = time.time()
            r = requests.post(self.api_url, json=payload, timeout=self.request_timeout)
            r.raise_for_status()
            data = r.json()
            text = data.get("response", "").strip()
            logger.debug(f"Ollama responded in {time.time()-start:.2f}s; tokens={data.get('eval_count','?')}")
            return text
        except requests.Timeout:
            return "Error: Ollama request timed out."
        except requests.RequestException as e:
            return f"Error: Ollama request failed: {e}"
        except json.JSONDecodeError as e:
            return f"Error: Ollama returned non-JSON: {e}"
        except Exception as e:
            return f"Error: Unexpected query error: {e}"

    def query_for_json(
        self,
        prompt: str,
        temperature: float = 0.5,
        retries: int = 1,
        system_prompt: str = "",
        num_ctx: Optional[int] = None,
    ) -> Union[dict, list, str]:
        """
        Query expecting a JSON dict or list. Retries once on parse or network error.
        Returns the parsed structure or an "Error: …" string.
        """
        suffix = "\n\nRespond ONLY with valid JSON object or list."
        full = prompt + suffix
        last_err = ""

        for attempt in range(retries + 1):
            logger.info(f"JSON query attempt {attempt+1}/{retries+1}")
            resp = self.query(full, temperature, format="json", system_prompt=system_prompt, num_ctx=num_ctx)
            if resp.startswith("Error:"):
                last_err = resp
                time.sleep(1.5 ** attempt)
                continue

            clean = self._strip_think_tags(resp)
            block, err = self._extract_json_block(clean) if not clean.startswith("{") and not clean.startswith("[") else (clean, None)
            if block:
                try:
                    return json.loads(block)
                except json.JSONDecodeError as e:
                    last_err = f"JSON decode failed: {e}"
                    full += f"\n\nFix JSON: {e}"
            else:
                last_err = err or "No JSON block found"

            time.sleep(1.5 ** attempt)

        return f"Error: Failed to get valid JSON after {retries+1} attempts: {last_err}"

    def query_for_code(
        self,
        prompt: str,
        context: str = "",
        temperature: float = 0.5,
        language: str = "python",
        system_prompt: str = "",
        num_ctx: Optional[int] = None,
    ) -> str:
        """
        Query expecting a fenced code block. Returns code string or "Error: …".
        """
        full = context + "\n\n" + prompt + f"\n\nRespond ONLY with valid {language} code in ```{language} ... ```."
        resp = self.query(full, temperature, num_ctx=num_ctx)
        if resp.startswith("Error:"):
            return resp

        code, err = self._extract_code_block(resp, language=language)
        if code:
            return code
        else:
            return f"Error: Code extraction failed: {err}"
