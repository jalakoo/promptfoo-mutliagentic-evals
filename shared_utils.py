#!/usr/bin/env python3
"""
Shared utilities for promptfoo multiagentic evals
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

def get_model_name_from_config(options: Dict[str, Any], context: Dict[str, Any], default: str = "openai/gpt-4o-mini") -> str:
    """
    Extract model name from various configuration sources with fallback priority:
    1. Provider config (options["config"]["model_name"])
    2. Test vars (context["vars"]["model_name"])
    3. Direct context (context["model_name"])
    4. Default fallback
    """
    # Method 1: Provider config (traditional promptfoo approach)
    if "config" in options and "model_name" in options["config"]:
        model_name = options["config"]["model_name"]
        logger.debug(f"Using model from provider config: {model_name}")
        return model_name
    
    # Method 2: Test vars (alternative approach)
    if "vars" in context and "model_name" in context["vars"]:
        model_name = context["vars"]["model_name"]
        logger.debug(f"Using model from test vars: {model_name}")
        return model_name
    
    # Method 3: Direct context vars
    if "model_name" in context:
        model_name = context["model_name"]
        logger.debug(f"Using model from direct context: {model_name}")
        return model_name
    
    # Method 4: Default fallback
    logger.warning(f"No model_name found, using default: {default}")
    return default
