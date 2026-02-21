import os
import re
from typing import Optional
from urllib.parse import quote_plus

import requests


class TodoChatbot:
    def __init__(self) -> None:
        self.todos: list[dict[str, object]] = []
        self.api_key = os.getenv("OPENAI_API_KEY", "").strip()
        self.model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini").strip()
        self.system_prompt = (
            "You are a concise, helpful assistant for a todo app. "
            "Give practical answers and keep responses short. "
            "If user asks for product suggestions, recommend good options."
        )

    def reply(self, message: str) -> str:
        local_result = self._handle_todo_command(message)
        if local_result is not None:
            return local_result

        product_result = self._handle_product_request(message)
        if product_result is not None:
            return product_result

        if self.api_key:
            llm_response = self._ask_openai(message)
            if llm_response:
                return llm_response

        return self._fallback_response(message)

    def _handle_todo_command(self, message: str) -> Optional[str]:
        text = message.strip()
        lower = text.lower()

        if lower in {"help", "commands"}:
            return (
                "Commands:\n"
                "- add <task>\n"
                "- list\n"
                "- done <number>\n"
                "- delete <number>\n\n"
                "Product queries:\n"
                "- suggest laptop under 800\n"
                "- best phone for gaming\n"
                "- buy running shoes"
            )

        add_match = re.match(r"^(add|new|todo)\s+(.+)$", text, flags=re.IGNORECASE)
        if add_match:
            task = add_match.group(2).strip()
            self.todos.append({"text": task, "done": False})
            return f"Added task #{len(self.todos)}: {task}"

        if lower in {"list", "tasks", "todos", "show tasks", "show todos"}:
            if not self.todos:
                return "No tasks yet. Add one with: add <task>"

            lines = []
            for idx, item in enumerate(self.todos, start=1):
                status = "x" if bool(item["done"]) else " "
                lines.append(f"{idx}. [{status}] {item['text']}")
            return "\n".join(lines)

        done_match = re.match(r"^(done|complete|finish)\s+(\d+)$", lower)
        if done_match:
            idx = int(done_match.group(2))
            if idx < 1 or idx > len(self.todos):
                return "Invalid task number."
            self.todos[idx - 1]["done"] = True
            return f"Marked task #{idx} as done."

        delete_match = re.match(r"^(delete|remove)\s+(\d+)$", lower)
        if delete_match:
            idx = int(delete_match.group(2))
            if idx < 1 or idx > len(self.todos):
                return "Invalid task number."
            removed = self.todos.pop(idx - 1)
            return f"Deleted task #{idx}: {removed['text']}"

        return None

    def _handle_product_request(self, message: str) -> Optional[str]:
        product_query = self._extract_product_query(message)
        if not product_query:
            return None

        links = self._build_shopping_links(product_query)
        link_text = "\n".join(f"- {store}: {url}" for store, url in links.items())

        if self.api_key:
            recommendation_prompt = (
                "Suggest 3 concrete products for this shopping request. "
                "Keep it concise and practical.\n"
                f"Request: {product_query}"
            )
            recommendation = self._ask_openai(recommendation_prompt)
            if recommendation:
                return (
                    f"Top picks for '{product_query}':\n"
                    f"{recommendation}\n\n"
                    "Buy links:\n"
                    f"{link_text}"
                )

        return (
            f"Online links for '{product_query}':\n"
            f"{link_text}\n\n"
            "Share your budget and preferred brand, and I will narrow it down."
        )

    def _extract_product_query(self, message: str) -> Optional[str]:
        text = message.strip()
        lower = text.lower()

        patterns = [
            r"^(?:buy|get|purchase|order)\s+(.+)$",
            r"^(?:recommend|suggest)\s+(?:me\s+)?(?:a|an|the\s+)?(.+)$",
            r"^(?:best)\s+(.+)$",
            r"^i\s+(?:want|need)\s+to\s+buy\s+(.+)$",
            r"^what\s+is\s+the\s+best\s+(.+)$",
            r"^which\s+(.+?)\s+should\s+i\s+buy\??$",
        ]

        for pattern in patterns:
            match = re.match(pattern, lower, flags=re.IGNORECASE)
            if match:
                candidate = match.group(1).strip(" .?!")
                if candidate:
                    return candidate

        # Support natural queries like: "Can you suggest a laptop under 800?"
        if any(
            keyword in lower
            for keyword in ("recommend", "suggest", "best", "buy", "purchase")
        ):
            candidate = re.sub(
                r"^(can you|please|hey|hi|hello|could you)\s+",
                "",
                text,
                flags=re.IGNORECASE,
            ).strip(" .?!")
            if candidate:
                return candidate

        return None

    def _build_shopping_links(self, query: str) -> dict[str, str]:
        encoded = quote_plus(query)
        return {
            "Amazon": f"https://www.amazon.com/s?k={encoded}",
            "Walmart": f"https://www.walmart.com/search?q={encoded}",
            "Best Buy": f"https://www.bestbuy.com/site/searchpage.jsp?st={encoded}",
            "Target": f"https://www.target.com/s?searchTerm={encoded}",
        }

    def _ask_openai(self, message: str) -> Optional[str]:
        url = "https://api.openai.com/v1/responses"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "input": [
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": self.system_prompt}],
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": message}],
                },
            ],
            "max_output_tokens": 300,
        }

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException:
            return None

        text = (data.get("output_text") or "").strip()
        if text:
            return text

        output = data.get("output", [])
        for item in output:
            for chunk in item.get("content", []):
                if chunk.get("type") == "output_text":
                    candidate = (chunk.get("text") or "").strip()
                    if candidate:
                        return candidate
        return None

    def _fallback_response(self, message: str) -> str:
        lower = message.lower().strip()

        if any(greeting in lower for greeting in ("hi", "hello", "hey")):
            return "Hi! I can manage todos. Try: add buy milk"
        if "name" in lower:
            return "I am your Todo Chatbot."
        if "how are you" in lower:
            return "Doing well. Want to add a task?"

        return (
            "I can help with todos and product links. Use:\n"
            "- add <task>\n"
            "- list\n"
            "- done <number>\n"
            "- delete <number>\n"
            "- suggest <product>"
        )
