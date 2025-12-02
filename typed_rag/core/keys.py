import os


def get_gemini_keys():
    try:
        api_key1 = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
    except KeyError:
        raise KeyError('GEMINI_API_KEY not set')
    return api_key1

def get_fastest_model():
    model_name = 'gemini-2.0-flash-lite'
    return model_name

def is_huggingface_model(model_name: str) -> bool:
    """Check if model name is in HuggingFace format (contains '/')."""
    return "/" in model_name
