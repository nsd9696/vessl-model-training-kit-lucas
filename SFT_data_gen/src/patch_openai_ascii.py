"""
Patch to fix ASCII encoding issues with Azure OpenAI API.
This must be imported BEFORE any openai imports.
"""

import os
import sys

def patch_openai_ascii():
    """Monkey patch openai client to use ASCII-only User-Agent header"""
    try:
        import openai
        if hasattr(openai, '_client'):
            # Patch the default headers to be ASCII-only
            if hasattr(openai._client, 'default_headers'):
                # Ensure User-Agent is ASCII-only
                if 'User-Agent' in openai._client.default_headers:
                    user_agent = openai._client.default_headers['User-Agent']
                    # Remove any non-ASCII characters
                    ascii_user_agent = user_agent.encode('ascii', 'ignore').decode('ascii')
                    openai._client.default_headers['User-Agent'] = ascii_user_agent
                    print(f"✅ Patched User-Agent header to ASCII-only: {ascii_user_agent}")
                else:
                    # Set a simple ASCII-only User-Agent if none exists
                    openai._client.default_headers['User-Agent'] = 'openai-python/1.0.0'
                    print("✅ Set ASCII-only User-Agent header")
    except Exception as e:
        print(f"⚠️ Warning: Could not patch openai client: {e}")

# Apply patch immediately when imported
patch_openai_ascii() 