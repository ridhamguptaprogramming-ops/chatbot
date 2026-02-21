import json
import os
import re
from datetime import date
from pathlib import Path
from typing import Optional
from urllib.parse import quote_plus

import requests


class TodoChatbot:
    PRIORITY_LEVELS = ("low", "medium", "high")
    SUPPORTED_LANGUAGES = {
        "english": "English",
        "en": "English",
        "hindi": "Hindi",
        "hi": "Hindi",
        "hinglish": "Hinglish",
        "spanish": "Spanish",
        "es": "Spanish",
    }

    def __init__(self) -> None:
        self.storage_path = Path(os.getenv("TODO_STORE_PATH", "data/todos.json"))
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.todos: list[dict[str, object]] = self._load_todos()
        self.api_key = os.getenv("OPENAI_API_KEY", "").strip()
        self.model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini").strip()
        self.max_memory_messages = int(os.getenv("CHAT_MEMORY_MESSAGES", "12"))
        self.session_memory: dict[str, list[dict[str, str]]] = {}
        self.system_prompt = (
            "You are a concise, helpful assistant for a todo app. "
            "Give practical answers and keep responses short. "
            "If user asks for product suggestions, recommend good options."
        )

    def reply(
        self,
        message: str,
        session_id: str = "default",
        language: str = "English",
    ) -> str:
        session_id = (session_id or "default").strip() or "default"
        language = self._normalize_language(language)

        local_result = self._handle_todo_command(message)
        if local_result is not None:
            self._remember_turn(session_id, message, local_result)
            return local_result

        product_result = self._handle_product_request(message)
        if product_result is not None:
            self._remember_turn(session_id, message, product_result)
            return product_result

        if self.api_key:
            llm_response = self._ask_openai(
                message,
                history=self._get_session_history(session_id),
                language=language,
            )
            if llm_response:
                self._remember_turn(session_id, message, llm_response)
                return llm_response

        fallback = self._fallback_response(message)
        self._remember_turn(session_id, message, fallback)
        return fallback

    def _handle_todo_command(self, message: str) -> Optional[str]:
        text = message.strip()
        lower = text.lower()

        if lower in {"help", "commands"}:
            return self._help_text()

        add_match = re.match(r"^(add|new|todo)\s+(.+)$", text, flags=re.IGNORECASE)
        if add_match:
            return self._add_todo(add_match.group(2).strip())

        list_match = re.match(
            r"^(?:list|tasks|todos|show tasks|show todos)"
            r"(?:\s+(all|open|done|high|medium|low|today|overdue))?$",
            lower,
        )
        if list_match:
            return self._render_todo_list(list_match.group(1) or "all")

        done_match = re.match(r"^(done|complete|finish)\s+(\d+)$", lower)
        if done_match:
            idx = int(done_match.group(2))
            item = self._get_todo_by_index(idx)
            if item is None:
                return "Invalid task number."
            item["done"] = True
            self._save_todos()
            return f"Marked task #{idx} as done."

        undone_match = re.match(r"^(undone|undo|reopen)\s+(\d+)$", lower)
        if undone_match:
            idx = int(undone_match.group(2))
            item = self._get_todo_by_index(idx)
            if item is None:
                return "Invalid task number."
            item["done"] = False
            self._save_todos()
            return f"Marked task #{idx} as not done."

        delete_match = re.match(r"^(delete|remove)\s+(\d+)$", lower)
        if delete_match:
            idx = int(delete_match.group(2))
            if idx < 1 or idx > len(self.todos):
                return "Invalid task number."
            removed = self.todos.pop(idx - 1)
            self._save_todos()
            return f"Deleted task #{idx}: {removed['text']}"

        priority_match = re.match(
            r"^(?:priority|set priority)\s+(\d+)\s+(low|medium|high)$", lower
        )
        if priority_match:
            idx = int(priority_match.group(1))
            item = self._get_todo_by_index(idx)
            if item is None:
                return "Invalid task number."
            item["priority"] = priority_match.group(2)
            self._save_todos()
            return f"Updated priority of task #{idx} to {priority_match.group(2)}."

        due_match = re.match(r"^(?:due|set due)\s+(\d+)\s+(\d{4}-\d{2}-\d{2}|none)$", lower)
        if due_match:
            idx = int(due_match.group(1))
            item = self._get_todo_by_index(idx)
            if item is None:
                return "Invalid task number."

            raw_due = due_match.group(2)
            due_value = None
            if raw_due != "none":
                due_value = self._normalize_due_date(raw_due)
                if due_value is None:
                    return "Invalid due date format. Use YYYY-MM-DD."

            item["due_date"] = due_value
            self._save_todos()
            if due_value is None:
                return f"Removed due date from task #{idx}."
            return f"Updated due date of task #{idx} to {due_value}."

        search_match = re.match(r"^(?:search|find)\s+(.+)$", text, flags=re.IGNORECASE)
        if search_match:
            query = search_match.group(1).strip().lower()
            if not query:
                return "Please provide search text."
            matches = [
                (idx + 1, item)
                for idx, item in enumerate(self.todos)
                if query in str(item.get("text", "")).lower()
            ]
            if not matches:
                return "No matching tasks found."
            return "\n".join(self._format_todo_line(idx, item) for idx, item in matches)

        if lower in {"clear done", "clear completed", "remove done", "remove completed"}:
            before = len(self.todos)
            self.todos = [item for item in self.todos if not bool(item["done"])]
            removed = before - len(self.todos)
            self._save_todos()
            return f"Removed {removed} completed task(s)."

        if lower in {"clear all", "reset", "delete all"}:
            count = len(self.todos)
            self.todos = []
            self._save_todos()
            return f"Cleared all tasks ({count})."

        if lower in {"stats", "summary"}:
            return self._stats_text()

        return None

    def _add_todo(self, raw_task: str) -> str:
        priority = "medium"
        due_date: Optional[str] = None

        priority_match = re.search(r"(?:^|\s)/p:(low|medium|high)\b", raw_task, re.IGNORECASE)
        if priority_match:
            priority = priority_match.group(1).lower()

        due_match = re.search(r"(?:^|\s)/d:(\d{4}-\d{2}-\d{2}|none)\b", raw_task, re.IGNORECASE)
        if due_match:
            raw_due = due_match.group(1).lower()
            if raw_due != "none":
                normalized_due = self._normalize_due_date(raw_due)
                if normalized_due is None:
                    return "Invalid due date format. Use /d:YYYY-MM-DD"
                due_date = normalized_due

        cleaned_task = re.sub(
            r"(?:^|\s)/p:(?:low|medium|high)\b",
            " ",
            raw_task,
            flags=re.IGNORECASE,
        )
        cleaned_task = re.sub(
            r"(?:^|\s)/d:(?:\d{4}-\d{2}-\d{2}|none)\b",
            " ",
            cleaned_task,
            flags=re.IGNORECASE,
        )
        cleaned_task = re.sub(r"\s+", " ", cleaned_task).strip()

        if not cleaned_task:
            return (
                "Task text is required. Example:\n"
                "add buy groceries /p:high /d:2026-03-10"
            )

        todo = {
            "text": cleaned_task,
            "done": False,
            "priority": priority,
            "due_date": due_date,
        }
        self.todos.append(todo)
        self._save_todos()

        extras = [f"priority={priority}"]
        if due_date:
            extras.append(f"due={due_date}")
        return f"Added task #{len(self.todos)}: {cleaned_task} ({', '.join(extras)})"

    def _render_todo_list(self, mode: str) -> str:
        if not self.todos:
            return "No tasks yet. Add one with: add <task>"

        mode = mode.lower()
        today = date.today().isoformat()

        def predicate(item: dict[str, object]) -> bool:
            done = bool(item.get("done", False))
            priority = self._normalize_priority(item.get("priority", "medium"))
            due_value = self._normalize_due_date(item.get("due_date"))

            if mode == "all":
                return True
            if mode == "open":
                return not done
            if mode == "done":
                return done
            if mode in self.PRIORITY_LEVELS:
                return priority == mode
            if mode == "today":
                return due_value == today
            if mode == "overdue":
                return (not done) and bool(due_value) and due_value < today
            return True

        lines = [
            self._format_todo_line(idx + 1, item)
            for idx, item in enumerate(self.todos)
            if predicate(item)
        ]
        if not lines:
            return f"No tasks found for filter '{mode}'."
        return "\n".join(lines)

    def _format_todo_line(self, index: int, item: dict[str, object]) -> str:
        status = "x" if bool(item.get("done", False)) else " "
        priority = self._normalize_priority(item.get("priority", "medium")).upper()
        due_value = self._normalize_due_date(item.get("due_date"))
        due_suffix = f" (due: {due_value})" if due_value else ""
        text = str(item.get("text", "")).strip()
        return f"{index}. [{status}] [{priority}] {text}{due_suffix}"

    def _stats_text(self) -> str:
        total = len(self.todos)
        done = sum(1 for item in self.todos if bool(item.get("done", False)))
        open_items = total - done
        high = sum(
            1
            for item in self.todos
            if self._normalize_priority(item.get("priority", "medium")) == "high"
        )
        today = date.today().isoformat()
        overdue = sum(
            1
            for item in self.todos
            if (not bool(item.get("done", False)))
            and bool(self._normalize_due_date(item.get("due_date")))
            and str(self._normalize_due_date(item.get("due_date"))) < today
        )
        return (
            "Task stats:\n"
            f"- Total: {total}\n"
            f"- Open: {open_items}\n"
            f"- Done: {done}\n"
            f"- High priority: {high}\n"
            f"- Overdue (open): {overdue}"
        )

    def _help_text(self) -> str:
        return (
            "Commands:\n"
            "- add <task> [/p:low|medium|high] [/d:YYYY-MM-DD]\n"
            "- list [all|open|done|high|medium|low|today|overdue]\n"
            "- done <number>\n"
            "- undone <number>\n"
            "- delete <number>\n"
            "- priority <number> <low|medium|high>\n"
            "- due <number> <YYYY-MM-DD|none>\n"
            "- search <text>\n"
            "- clear done\n"
            "- clear all\n"
            "- stats\n\n"
            "Product queries:\n"
            "- suggest laptop under 800\n"
            "- best phone for gaming\n"
            "- buy running shoes"
        )

    def _get_todo_by_index(self, index: int) -> Optional[dict[str, object]]:
        if index < 1 or index > len(self.todos):
            return None
        return self.todos[index - 1]

    def _normalize_priority(self, raw_priority: object) -> str:
        value = str(raw_priority or "").strip().lower()
        if value in self.PRIORITY_LEVELS:
            return value
        return "medium"

    def _normalize_due_date(self, raw_due_date: object) -> Optional[str]:
        value = str(raw_due_date or "").strip()
        if not value:
            return None
        try:
            return date.fromisoformat(value).isoformat()
        except ValueError:
            return None

    def _normalize_language(self, raw_language: str) -> str:
        key = raw_language.strip().lower()
        return self.SUPPORTED_LANGUAGES.get(key, raw_language.strip() or "English")

    def _get_session_history(self, session_id: str) -> list[dict[str, str]]:
        history = self.session_memory.get(session_id, [])
        if len(history) <= self.max_memory_messages:
            return list(history)
        return history[-self.max_memory_messages:]

    def _remember_turn(self, session_id: str, user_text: str, assistant_text: str) -> None:
        memory = self.session_memory.setdefault(session_id, [])
        memory.append({"role": "user", "text": user_text})
        memory.append({"role": "assistant", "text": assistant_text})
        if len(memory) > self.max_memory_messages:
            del memory[:-self.max_memory_messages]

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
            match = re.match(pattern, text, flags=re.IGNORECASE)
            if match:
                candidate = match.group(1).strip(" .?!")
                if candidate:
                    return candidate

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

    def _load_todos(self) -> list[dict[str, object]]:
        if not self.storage_path.exists():
            return []

        try:
            raw_data = json.loads(self.storage_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []

        if not isinstance(raw_data, list):
            return []

        cleaned: list[dict[str, object]] = []
        for item in raw_data:
            if not isinstance(item, dict):
                continue

            text = str(item.get("text", "")).strip()
            if not text:
                continue

            cleaned.append(
                {
                    "text": text,
                    "done": bool(item.get("done", False)),
                    "priority": self._normalize_priority(item.get("priority", "medium")),
                    "due_date": self._normalize_due_date(item.get("due_date")),
                }
            )
        return cleaned

    def _save_todos(self) -> None:
        try:
            self.storage_path.write_text(
                json.dumps(self.todos, ensure_ascii=True, indent=2),
                encoding="utf-8",
            )
        except OSError:
            return

    def _ask_openai(
        self,
        message: str,
        history: Optional[list[dict[str, str]]] = None,
        language: str = "English",
    ) -> Optional[str]:
        url = "https://api.openai.com/v1/responses"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        input_items = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            f"{self.system_prompt} "
                            f"Reply in {language}. "
                            "If user asks follow-up questions, use previous context."
                        ),
                    }
                ],
            }
        ]

        for item in history or []:
            role = "assistant" if item.get("role") == "assistant" else "user"
            text = (item.get("text") or "").strip()
            if not text:
                continue
            input_items.append(
                {
                    "role": role,
                    "content": [{"type": "input_text", "text": text}],
                }
            )

        input_items.append(
            {
                "role": "user",
                "content": [{"type": "input_text", "text": message}],
            }
        )

        payload = {
            "model": self.model,
            "input": input_items,
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
            return "Hi! I can manage todos. Try: add buy milk /p:high"
        if "name" in lower:
            return "I am your Todo Chatbot."
        if "how are you" in lower:
            return "Doing well. Want to add a task?"

        return (
            "I can help with todos and product links. Use:\n"
            "- add <task> [/p:high] [/d:YYYY-MM-DD]\n"
            "- list [open|done|today|overdue]\n"
            "- done <number>\n"
            "- undone <number>\n"
            "- priority <number> <low|medium|high>\n"
            "- due <number> <YYYY-MM-DD|none>\n"
            "- search <text>\n"
            "- stats\n"
            "- suggest <product>"
        )
