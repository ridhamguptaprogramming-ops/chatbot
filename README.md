# Todo Chatbot

A lightweight chatbot web app built with Flask.

It supports:
- Todo commands (`add`, `list`, `done`, `delete`)
- Product suggestions with online shopping links (Amazon, Walmart, Best Buy, Target)
- Optional OpenAI-powered replies when `OPENAI_API_KEY` is set

## Run Locally

1. Create and activate a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
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
   python app.py
   ```
5. Open:
   ```text
   http://127.0.0.1:5000
   ```

## Chat Commands

- `help`
- `add buy groceries`
- `list`
- `done 1`
- `delete 1`
- `suggest laptop under 800`
- `best phone for gaming`
