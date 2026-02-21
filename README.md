# Todo Chatbot

A lightweight chatbot web app built with Flask.

It supports:
- Voice recognition (speech-to-text in browser)
- Text-to-speech (AI voice replies in browser)
- Natural conversation context memory (session-based, best with OpenAI key)
- Multi-language chat mode (English, Hindi, Hinglish, Spanish)
- Smart reply suggestions (dynamic quick-reply chips)
- Live task dashboard (total/open/done/overdue + top tasks)
- Persistent chat history in browser (restores after refresh)
- Smart todo commands (`add`, `list`, `done`, `undone`, `delete`)
- Advanced task controls (`priority`, `due`, `search`, `clear done`, `clear all`, `stats`)
- Todo persistence on disk (`data/todos.json`) so tasks survive restarts
- Product suggestions with online shopping links (Amazon, Walmart, Best Buy, Target)
- Live web search in chat (DuckDuckGo + Wikipedia fallback) with Google links
- Optional OpenAI-powered replies when `OPENAI_API_KEY` is set

## Run Locally

1. Create and activate a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   python -m pip install -r requirements.txt
   ```
3. Set environment variables:
   ```bash
   cp .env.example .env
   ```
   Add your key in `.env` if you want LLM responses:
   ```bash
   OPENAI_API_KEY=your_key_here
   ```
4. Start the server:
   ```bash
   python3 app.py
   ```
5. Open:
   ```text
   http://127.0.0.1:5000
   ```

## Chat Commands

- `help`
- `add buy groceries /p:high /d:2026-03-10`
- `list`
- `list open`
- `list overdue`
- `done 1`
- `undone 1`
- `delete 1`
- `priority 1 low`
- `due 1 2026-03-20`
- `search groceries`
- `search task groceries`
- `clear done`
- `clear all`
- `stats`
- `suggest laptop under 800`
- `best phone for gaming`
- `web oats nutrition`
- `google best protein powder under 50`
