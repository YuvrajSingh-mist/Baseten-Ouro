"""
The `Model` class is an interface between the ML model that you're packaging and the model
server that you're running it on.

The main methods to implement here are:
* `load`: runs exactly once when the model server is spun up or patched and loads the
   model onto the model server. Include any logic for initializing your model, such
   as downloading model weights and loading the model into memory.
* `predict`: runs every time the model server is called. Include any logic for model
  inference and return the model output.

See https://truss.baseten.co/quickstart for more.
"""

import os
import sys
import time
import uuid
import yaml
import json
import hashlib
from typing import List, Dict, Any, Optional
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoConfig
import redis
from loguru import logger


class Model:
    def __init__(self, **kwargs):
        # Initialize with Truss parameters
        self._data_dir = kwargs.get("data_dir")
        self._config = kwargs.get("config", {})
        self._secrets = kwargs.get("secrets", {})
        self._environment = kwargs.get("environment", "development")
        self._lazy_data_resolver = kwargs.get("lazy_data_resolver")

        # Model components
        self._model = None
        self._tokenizer = None
        self._redis_client = None

        # Configuration
        self._model_name = None
        self._max_context_length = 20  # Max messages to keep in context
        self._cache_ttl = 3600  # Cache TTL in seconds
        # Configure logger for this module (stdout so Baseten captures it)
        logger.remove()
        log_level = os.getenv("LOG_LEVEL", "INFO")
        # Use a safe formatter function so missing `extra` keys don't raise KeyError
        def _format(record):
            # record is a dict-like mapping provided by Loguru
            try:
                ts = record.get("time")
                if hasattr(ts, "strftime"):
                    ts_str = ts.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                else:
                    ts_str = str(ts)
                levelname = record.get("level").name if record.get("level") is not None else "-"
                extra = record.get("extra", {}) or {}
                rid = extra.get("request_id") or "-"
                sid = extra.get("session_id") or "-"
                # message may be under "message" or "event" depending on record
                msg = record.get("message") if record.get("message") is not None else record.get("event", "")
                return f"{ts_str} | {levelname:<8} | {rid} | {sid} | {msg}"
            except Exception:
                # Fallback minimal format to avoid crashing the logger
                return f"{record.get('time')} | {record.get('level')} | - | - | {record.get('message')}"

        logger.add(
            sys.stdout,
            level=log_level,
            enqueue=True,
            backtrace=True,
            diagnose=False,
            format=_format,
        )

    def load(self):
        """Initialize the model with cached weights."""
        # Use config from kwargs instead of reading config.yaml
        config_data = self._config
        self._model_name = config_data.get('model_name', 'ByteDance/Ouro-1.4B')

        # # Prefetching the model weights
        logger.info("Waiting for model weights to be cached")
        self._lazy_data_resolver.block_until_download_complete()
        logger.info("Model weights cached and ready")


        # Get tokens from secrets
        hf_token = self._secrets.get("hf_access_token")

      
        redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
        try:
            
            self._redis_client = redis.from_url(redis_url)

            # Test connection
            self._redis_client.ping()
            logger.info("Redis connection established")
        except Exception as e:
            logger.warning("Redis connection failed; continuing without caching. error={error}", error=str(e))
            self._redis_client = None

        # Wait for model weights to be downloaded and cached

           
        # Determine model path - use cached weights if available
        model_cache_path = "/app/model_cache/ouro-1.4b"
        if os.path.exists(model_cache_path):
            model_path = model_cache_path
            logger.info("Using cached model weights from: {path}", path=model_path)
        else:
            model_path = self._model_name
            logger.info("Downloading model weights from: {path}", path=model_path)

        # Load model configuration and tokenizer from cached path
        # config = AutoConfig.from_pretrained(model_path, token=hf_token)
        self._tokenizer = AutoTokenizer.from_pretrained(model_path, token=hf_token)

        # Initialize vLLM engine with optimized settings for high concurrency
        logger.info("Initializing Ouro-1.4B model with vLLM")
        self._model = LLM(
            model=model_path,  # Use cached path if available
            block_size=256,
            compilation_config=0,
            # enforce_eager=True,
            trust_remote_code=True,
            max_model_len=32768,
            swap_space=0,
            max_num_seqs=256,  # Optimized for high concurrency
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,  # Slightly reduced for stability
            disable_mm_preprocessor_cache=True,
            # dtype='bfloat16'
        )
        logger.info("Model initialization complete")

    def _get_cache_key(self, messages: List[Dict[str, str]], params: Dict[str, Any]) -> str:
        """Generate a cache key for tokenized inputs."""
        # Create a hash of messages and parameters
        content = json.dumps({
            "messages": messages,
            "temperature": params.get("temperature"),
            "top_p": params.get("top_p"),
            "max_tokens": params.get("max_tokens")
        }, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()

    def _cache_get(self, key: str) -> Optional[str]:
        """Get value from Redis cache."""
        if not self._redis_client:
            return None
        try:
            return self._redis_client.get(f"cache:{key}")
        except:
            return None

    def _cache_set(self, key: str, value: str):
        """Set value in Redis cache."""
        if not self._redis_client:
            return
        try:
            self._redis_client.setex(f"cache:{key}", self._cache_ttl, value)
        except:
            pass

    def predict(self, model_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run model inference with multi-turn chat support.

        Expected input format:
        {
            "messages": [
                {"role": "user", "content": "Hello, how are you?"},
                {"role": "assistant", "content": "I'm doing well, thank you!"},
                {"role": "user", "content": "What's the weather like?"}
            ],
            "max_tokens": 512,     # optional, default: 512
            "temperature": 1.0,    # optional, default: 1.0
            "top_p": 0.7,          # optional, default: 0.7
            "session_id": "abc"    # optional, for caching
        }

        Returns:
        {
            "generated_text": "Assistant response here",
            "model": "ByteDance/Ouro-1.4B-Thinking",
            "session_id": "abc",   # echoed back if provided
            "cached": false        # whether response was from cache
        }
        """
        try:
            # Validate input
            messages = model_input.get("messages", [])
            if not messages:
                return {"error": "messages field is required and cannot be empty"}

            # Bind a per-request logger with ids for easy tracing
            request_id = model_input.get("request_id") or str(uuid.uuid4())
            session_id = model_input.get("session_id")
            req_logger = logger.bind(request_id=request_id, session_id=session_id)
            req_logger.info("request_received", messages_len=len(messages))

            # Validate message format
            for msg in messages:
                if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
                    return {"error": "Each message must have 'role' and 'content' fields"}
                if msg["role"] not in ["user", "assistant", "system"]:
                    return {"error": f"Invalid role '{msg['role']}'. Must be 'user', 'assistant', or 'system'"}

            # Limit context length to prevent excessive memory usage
            if len(messages) > self._max_context_length * 2:  # *2 because conversations alternate
                messages = messages[-(self._max_context_length * 2):]

            # Prepare sampling parameters
            max_tokens = 32768
            temperature = 0.9
            top_p = 0.9

            params = {
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens
            }

            # Check cache for identical requests (optional session-based caching)
            cached_response = None
            if session_id and self._redis_client:
                cache_key = self._get_cache_key(messages, params)
                req_logger.debug("checking_cache", cache_key=cache_key)
                cached_data = self._cache_get(cache_key)
                if cached_data:
                    try:
                        cached_response = json.loads(cached_data)
                    except Exception as e:
                        req_logger.warning("cache_deserialize_failed", error=str(e))

            if cached_response:
                cached_response["cached"] = True
                cached_response["session_id"] = session_id
                req_logger.info("cache_hit", cache_key=cache_key)
                return cached_response

            # Format chat prompt using tokenizer's chat template
            req_logger.debug("formatting_prompt")
            inputs = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # Create sampling parameters
            sampling_params = SamplingParams(
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                stop=["<|endoftext|>", "<|im_end|>"]
            )

            # Generate response
            req_logger.info("generation_start", max_tokens=max_tokens, temperature=temperature, top_p=top_p)
            start_ts = time.time()
            outputs = self._model.generate([inputs], sampling_params)
            duration = time.time() - start_ts

            if not outputs or not outputs[0].outputs:
                req_logger.error("no_output_generated", duration=duration)
                return {"error": "No output generated"}

            generated_text = outputs[0].outputs[0].text
            req_logger.info("generation_complete", duration=duration)

            # Prepare response
            response = {
                "generated_text": generated_text,
                "model": self._model_name,
                "cached": False
            }

            if session_id:
                response["session_id"] = session_id
                # Cache the response for future identical requests
                if self._redis_client:
                    cache_key = self._get_cache_key(messages, params)
                    cache_data = json.dumps({
                        "generated_text": generated_text,
                        "model": self._model_name,
                        "cached": False
                    })
                    try:
                        self._cache_set(cache_key, cache_data)
                        req_logger.debug("cache_set", cache_key=cache_key)
                    except Exception as e:
                        req_logger.warning("cache_set_failed", error=str(e))

            req_logger.info("request_complete", response_size=len(generated_text))

            return response

        except Exception as e:
            return {"error": f"Internal error: {str(e)}"}
