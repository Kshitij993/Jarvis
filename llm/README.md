# llm/

LLM chat UI and config for the OpenBridge AI personal server.

---

## Files

| File | Purpose |
|---|---|
| `llm_ui.py` | Tkinter chat UI — configure key, URL, system prompt and chat |
| `.llm_config.json` | Shared config read by both `llm_ui.py` and `jarvis/jarvis.py` |

---

## Setup

1. Open `.llm_config.json` and paste your API key:
   ```json
   {
     "provider": "custom",
     "api_key": "YOUR_API_KEY",
     "api_url": "https://openbridgeai.kshitijks.com/api/v1/chat/completions",
     "system_prompt": "You are Jarvis, a smart and friendly AI assistant. Be concise and helpful."
   }
   ```

2. Run the chat UI:
   ```bash
   python llm/llm_ui.py
   ```

   Or launch Jarvis (uses the same config automatically):
   ```bash
   python jarvis/jarvis.py
   ```

---

## Config fields

| Field | Description |
|---|---|
| `provider` | Always `"custom"` |
| `api_key` | Bearer token for the OpenBridge AI server |
| `api_url` | Endpoint URL (change if the server address changes) |
| `system_prompt` | System instruction sent with every conversation |

The UI **Save** button writes all fields back to `.llm_config.json`.  
Jarvis picks up changes on the next restart.
