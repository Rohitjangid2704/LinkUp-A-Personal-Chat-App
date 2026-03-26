import dspy
import os

# Model providers
PROVIDERS = {
    "openai": "OpenAI"
}

max_tokens = int(os.getenv("MAX_TOKENS", 6000))
default_temperature = min(1.0, max(0.0, float(os.getenv("TEMPERATURE", "1.0"))))

# Lightweight LMs used for small internal tasks (planning, classification, etc.)
small_lm = dspy.LM('openai/gpt-4o-mini', max_tokens=300, api_key=os.getenv("OPENAI_API_KEY"), cache=False)
mid_lm = dspy.LM('openai/gpt-4o-mini', max_tokens=1800, api_key=os.getenv("OPENAI_API_KEY"), cache=False)

# The new requested models mapped to actual OpenAI models
gpt_4_1_mini = dspy.LM(
    model="openai/gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=default_temperature,
    max_tokens=16_000,
    cache=False
)

gpt_4_1 = dspy.LM(
    model="openai/gpt-4o",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=default_temperature,
    max_tokens=16_000,
    cache=False
)

o4_mini = dspy.LM(
    model="openai/o3-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=1.0, # Required for reasoning models
    max_tokens=16_000,
    cache=False
)

o3 = dspy.LM(
    model="openai/o3-mini", # Setting o3 to o3-mini as well to ensure compatibility
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=1.0, # Required for reasoning models
    max_tokens=20_000,
    cache=False
)

MODEL_OBJECTS = {
    "gpt-4.1-mini": gpt_4_1_mini,
    "gpt-4.1": gpt_4_1,
    "o4-mini": o4_mini,
    "o3": o3
}

def get_model_object(model_name: str):
    """Get model object by name"""
    return MODEL_OBJECTS.get(model_name, gpt_4_1)

# Tiers based on cost per 1K tokens
MODEL_TIERS = {
    "tier1": {
        "name": "Basic",
        "credits": 1,
        "models": ["gpt-4.1-mini"]
    },
    "tier2": {
        "name": "Standard",
        "credits": 3,
        "models": ["gpt-4.1"]
    },
    "tier3": {
        "name": "Premium",
        "credits": 5,
        "models": ["o4-mini", "o3"]
    }
}

MODEL_METADATA = {
    "gpt-4.1-mini": {"display_name": "GPT-4.1 Mini", "context_window": 128000},
    "gpt-4.1": {"display_name": "GPT-4.1", "context_window": 128000},
    "o4-mini": {"display_name": "o4 Mini", "context_window": 100000},
    "o3": {"display_name": "o3", "context_window": 100000}
}

MODEL_COSTS = {
    "openai": {
        "gpt-4.1-mini": {"input": 0.00015, "output": 0.0006},
        "gpt-4.1": {"input": 0.0025, "output": 0.01},
        "o4-mini": {"input": 0.0011, "output": 0.0044},
        "o3": {"input": 0.002, "output": 0.008}
    }
}

# Helper functions
def get_provider_for_model(model_name):
    """Determine the provider based on model name"""
    return "openai" if any(model_name in models for models in MODEL_COSTS.values()) else "Unknown"

def get_model_tier(model_name):
    """Get the tier of a model"""
    for tier_id, tier_info in MODEL_TIERS.items():
        if model_name in tier_info["models"]:
            return tier_id
    return "tier1"

def calculate_cost(model_name, input_tokens, output_tokens):
    """Calculate the cost for using the model based on tokens"""
    if not model_name:
        return 0
    input_tokens_in_thousands = input_tokens / 1000
    output_tokens_in_thousands = output_tokens / 1000
    model_provider = get_provider_for_model(model_name)
    if model_provider == "Unknown" or model_name not in MODEL_COSTS.get(model_provider, {}):
        return 0
    return (input_tokens_in_thousands * MODEL_COSTS[model_provider][model_name]["input"] + 
            output_tokens_in_thousands * MODEL_COSTS[model_provider][model_name]["output"])

def get_credit_cost(model_name):
    """Get the credit cost for a model"""
    tier_id = get_model_tier(model_name)
    return MODEL_TIERS[tier_id]["credits"]

def get_display_name(model_name):
    """Get the display name for a model"""
    return MODEL_METADATA.get(model_name, {}).get("display_name", model_name)

def get_context_window(model_name):
    """Get the context window size for a model"""
    return MODEL_METADATA.get(model_name, {}).get("context_window", 4096)

def get_all_models_for_provider(provider):
    """Get all models for a specific provider"""
    if provider not in MODEL_COSTS:
        return []
    return list(MODEL_COSTS[provider].keys())

def get_models_by_tier(tier_id):
    """Get all models for a specific tier"""
    return MODEL_TIERS.get(tier_id, {}).get("models", [])
