import json
import re

from stockagent.src.llm import chat, observation
from stockagent.src.prompts import SYSTEM_PROMPT
from stockagent.src.tools import get_stock_price


def extract_json(text: str):
    match = re.search(r"\{.*\}", text, re.DOTALL)

    if not match:
        return None

    try:
        return json.loads(match.group())
    except Exception:
        return None


class StockMarketAgent:

    def run(self, question):

        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": question,
            },
        ]

        response = observation(messages)

        print("\n=== LLM Response ===")
        print(response)

        tool_call = extract_json(response)

        if not tool_call:
            return response

        if tool_call["action"] == "get_stock_price":

            ticker = tool_call["action_input"]["ticker"]

            result = get_stock_price(ticker)
            messages = [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": question,
                },
                {
                    "role": "assistant",
                    "content": f"Tool Result: {result}",
                }
            ]
            response = chat(messages)

            return f"Final Response: {result}"

        return "Unknown tool"