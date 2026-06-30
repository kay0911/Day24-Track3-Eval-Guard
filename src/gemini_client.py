import os
import time
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Load environment variables
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
load_dotenv(dotenv_path=dotenv_path)

KEY_1 = os.getenv("GEMINI_API_KEY_1")
KEY_2 = os.getenv("GEMINI_API_KEY_2")

keys = [k for k in [KEY_1, KEY_2] if k]
if not keys:
    from dotenv import load_dotenv
    load_dotenv()
    KEY_1 = os.getenv("GEMINI_API_KEY_1")
    KEY_2 = os.getenv("GEMINI_API_KEY_2")
    keys = [k for k in [KEY_1, KEY_2] if k]
    if not keys:
        # Check standard GEMINI_API_KEY as well
        standard_key = os.getenv("GEMINI_API_KEY")
        if standard_key:
            keys = [standard_key]
        else:
            raise RuntimeError("No Gemini API keys found. Please set GEMINI_API_KEY_1 and GEMINI_API_KEY_2 in .env file.")

current_key_idx = 0
last_request_time = 0.0

def get_client():
    global current_key_idx
    key = keys[current_key_idx]
    # Rotate for the next call
    current_key_idx = (current_key_idx + 1) % len(keys)
    return genai.Client(api_key=key)

def generate_gemini(
    prompt: str, 
    system_instruction: str = None, 
    json_mode: bool = False, 
    model_name: str = "gemini-3.1-flash-lite",
    max_retries: int = 5
) -> str:
    global last_request_time
    
    # Rate limit: max 15 RPM per key, with 2 keys that's max 30 RPM total.
    # Sleep to ensure at least 2.2 seconds between calls.
    now = time.time()
    elapsed = now - last_request_time
    delay = 2.2
    if elapsed < delay:
        time.sleep(delay - elapsed)
        
    for attempt in range(max_retries):
        try:
            client = get_client()
            
            config = {}
            if system_instruction:
                config["system_instruction"] = system_instruction
            if json_mode:
                config["response_mime_type"] = "application/json"
            
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(**config) if config else None
            )
            
            last_request_time = time.time()
            return response.text
        except Exception as e:
            err_msg = str(e)
            print(f"[Gemini Client] Attempt {attempt+1}/{max_retries} failed: {err_msg}")
            # Rotate key index immediately on failure to try the other key
            if "429" in err_msg or "ResourceExhausted" in err_msg:
                # Quota error, sleep longer and try again
                time.sleep(5.0)
            else:
                time.sleep(1.0)
                
    raise RuntimeError(f"Gemini API failed after {max_retries} attempts.")


# --- Custom LangChain Chat Model for Key Rotation & Rate Limiting ---
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from typing import List, Optional, Any

class KeyRotatingGeminiChat(BaseChatModel):
    model_name: str = "gemini-3.1-flash-lite"
    
    @property
    def _llm_type(self) -> str:
        return "key-rotating-gemini-chat"
        
    def _generate(
        self, 
        messages: List[Any], 
        stop: Optional[List[str]] = None, 
        run_manager: Optional[Any] = None, 
        **kwargs: Any
    ) -> ChatResult:
        system_instruction = None
        user_prompt = ""
        for msg in messages:
            if msg.type == "system":
                system_instruction = msg.content
            else:
                user_prompt += msg.content + "\n"
                
        response_text = generate_gemini(
            prompt=user_prompt.strip(),
            system_instruction=system_instruction,
            model_name=self.model_name
        )
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=response_text))])

