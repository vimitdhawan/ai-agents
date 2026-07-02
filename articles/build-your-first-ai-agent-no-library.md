# AI Agents Are Just LLMs + Tools + a Loop — Here's the Code to Prove It

By the end of this article, you'll have a working AI agent — one that can answer stock market questions by actually using a tool, not just guessing. You'll have written it yourself, from scratch, without a single agent framework. And you'll understand exactly how every line works.

No LangChain. No CrewAI. No AutoGen. Just Python and an LLM.

Let's go.

---

## So What Exactly Is an AI Agent?

An AI Agent is a system that can think, plan, and take actions to achieve a goal. That sounds fancy, but it helps to strip away the jargon.

Think about the last time you asked a human assistant for something. Imagine you say: "Get me a cup of coffee."

What does that assistant actually do?

1. **Understands the request** — they know you want coffee, not tea
2. **Plans the steps** — go to the kitchen, find the coffee machine, figure out the controls
3. **Uses available tools** — the coffee machine, maybe a mug from the cabinet
4. **Delivers the result** — hands you a hot cup of coffee

An AI Agent works the same way. It receives a goal, decides what to do, uses tools when needed, and completes the task.

Compare that to a normal chatbot. A chatbot gets a message, generates a response, and stops. It can't go look something up. It can't call a function. It can't take a second step based on what it just learned.

An agent can.

**The two parts of every agent**

Every AI agent has two essential components:

1. **The Brain** — Usually a Large Language Model (LLM) like GPT, Llama, or Gemini. Its job is to understand instructions, reason about problems, create a plan, and decide on the next action.

2. **The Tools** — Functions that let the agent interact with the outside world. Web search, database queries, code execution, sending emails — these are all tools. Without tools, an LLM can only generate text. Tools give it hands.

**How an agent works**

Most agents follow a simple cycle:

```
Think → Act → Observe → Repeat
```

- **Think**: Understand the goal and decide what to do next
- **Act**: Use a tool or perform an action
- **Observe**: Check the result
- **Repeat**: Until the task is done

That's the entire game. Everything else is details.

In this article, we're going to build an agent that answers stock market questions. Ask it "What's the stock price of AAPL?" and it will think about whether it needs a tool, call a stock price lookup, observe the result, and give you a real answer — not a guess from training data.

Let's start with the brain.

---

## The Brain: Giving Your Agent Something to Think With

Before your agent can do anything, it needs a brain. In modern agents, that brain is a Large Language Model.

### What is an LLM?

An LLM is an AI model trained on massive amounts of text. It learns patterns in language — grammar, reasoning, relationships between concepts — and uses that knowledge to understand and generate human language.

It can answer questions, write code, summarize information, and follow instructions. Popular ones include GPT-4, Llama, Gemini, DeepSeek, and Mistral.

### How does it actually work?

Here's the surprising part: at its core, an LLM has one job — predict the next token.

A token is a small piece of text. Usually a word or part of a word. Given the phrase:

> "The capital of France is"

...the model predicts the next token: **"Paris."**

It repeats this one token at a time until the response is complete. That's it. Everything else — reasoning, creativity, following instructions — emerges from this simple process at massive scale.

### Why are they so powerful?

Modern LLMs are built using something called the Transformer architecture. The key innovation is **Attention** — the model can focus on the most important parts of the input when making predictions.

In the sentence "The capital of France is," the words "capital" and "France" are more relevant than "the" or "is" when predicting "Paris." Attention lets the model weigh that importance.

This ability to understand context is what makes LLMs feel intelligent. They can track what's been said in a conversation, understand references, and maintain coherence across long responses.

### Messages and Chat Templates

When you chat with something like ChatGPT, it *feels* like a continuous conversation. But the model doesn't actually remember anything. Every time you send a message, the entire conversation history is sent again as a single prompt.

A conversation has three types of messages:

- **System message** — Defines how the AI should behave. Example: "You are a helpful financial assistant."
- **User message** — The user's request or question. Example: "What's the stock price of AAPL?"
- **Assistant message** — The AI's response.

Different LLMs expect conversations in different formats. A **chat template** converts structured messages into a single prompt the model can understand. Each model uses its own formatting rules and special tokens — get the template wrong and the model produces garbage.

### Base Models vs Instruct Models

This matters for agents:

- **Base models** are trained to predict the next token. They're not specifically trained to follow instructions. Ask a base model a question and it might continue a story rather than answer.
- **Instruct models** are further fine-tuned to follow user requests and participate in conversations. Examples: `Llama 3 Instruct`, `GPT-4`, `Kimi-K2.5`.

We'll use an instruct model. They understand tasks better and follow instructions more reliably — exactly what you need for an agent.

### Setting Up the LLM Client

Here's where we start writing code. The brain of our agent is an LLM accessed via the HuggingFace Inference API.

Create a file called `src/llm.py`:

```python
import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

client = InferenceClient(
    model="moonshotai/Kimi-K2.5",
    token=os.getenv("HF_TOKEN"),
)
```

This sets up a client that connects to the Kimi-K2.5 model on HuggingFace. You'll need a `HF_TOKEN` environment variable — get one from huggingface.co and add it to a `.env` file:

```
HF_TOKEN=your_token_here
```

Now let's add two functions. This is where it gets interesting.

```python
def observation(messages):
    response = client.chat.completions.create(
        messages=messages,
        stop=["Observation:"],
        max_tokens=2048,
        extra_body={'thinking': {'type': 'disabled'}},
        stream=False,
    )
    return response.choices[0].message.content

def chat(messages):
    response = client.chat.completions.create(
        messages=messages,
        max_tokens=1024,
        extra_body={'thinking': {'type': 'disabled'}},
        stream=False,
    )
    return response.choices[0].message.content
```

**Why two functions?**

`observation()` is called first. It sends the conversation to the LLM and tells it to stop before it generates an "Observation:" token — more on why shortly. The LLM sees the conversation, reasons about it, and returns a structured response telling the agent what to do.

`chat()` is called after the agent has observed the tool result. It takes the full conversation (including what the tool returned) and asks the LLM to generate a natural-language response for the user.

You can swap `Kimi-K2.5` for any instruct model on HuggingFace — `meta-llama/Llama-3.1-8B-Instruct`, `Qwen/Qwen2.5-7B-Instruct`, whatever you prefer. The agent logic stays the same.

---

## The Hands: Teaching Your Agent to Use Tools

LLMs are powerful, but they have real limitations:

- They can't access real-time information on their own
- They make embarrassing mistakes in math
- They can't directly interact with external systems

If you ask an LLM for today's AAPL stock price, it might guess or hallucinate a number. With a stock price tool, it can retrieve the actual data.

A **tool** is simply a function the agent can call to perform a specific task.

### How do tools work?

An LLM can't directly execute code or call APIs. Instead, it decides *when* a tool is needed and generates a structured request describing what to do. The agent framework then:

1. Reads the instruction
2. Executes the tool
3. Collects the result
4. Sends the result back to the LLM
5. The LLM generates a final response

From the user's perspective, it feels like the AI used the tool directly. But the agent does the actual work behind the scenes.

### What makes a good tool?

A tool needs four things:

- A **clear purpose** — the LLM should know exactly what the tool does
- **Well-defined inputs** — what arguments does it need?
- **Predictable outputs** — what does it return?
- A **simple description** — the LLM only sees the description, not the implementation

This is critical: the LLM doesn't see your code. It only sees the text description you write in the prompt. The better the description, the better the model knows when and how to use the tool.

### Writing the System Prompt

The system prompt is the instruction manual for your agent. It tells the LLM what tools are available and how to use them. Create `src/prompts.py`:

```python
SYSTEM_PROMPT = """Answer the following questions as best you can. You have access to the following tools:

get_stock_price: Get the current stock price for a given ticker symbol

The way you use the tools is by specifying a json blob.
Specifically, this json should have an `action` key (with the name of the tool to use) and an `action_input` key (with the input to the tool going here).

The only values that should be in the "action" field are:
get_stock_price: Get the current stock price for a given ticker symbol, args: {"ticker": {"type": "string"}}
example use:

{{
  "action": "get_stock_price",
  "action_input": {{"ticker": "AAPL"}}
}}

ALWAYS use the following format:

Question: the input question you must answer
Thought: you should always think about one action to take. Only one action at a time in this format:
Action:

$JSON_BLOB (inside markdown cell)

Observation: the result of the action. This Observation is unique, complete, and the source of truth.
... (this Thought/Action/Observation can repeat N times, you should take several steps when needed. The $JSON_BLOB must be formatted as markdown and only use a SINGLE action at a time.)

You must always end your output with the following format:

Thought: I now know the final answer
Final Answer: the final answer to the original input question

Now begin! Reminder to ALWAYS use the exact characters `Final Answer:` when you provide a definitive answer."""
```

Notice what's happening here. We're using a format called **ReAct** — Reasoning + Acting. The LLM is instructed to think step by step, take one action at a time, and observe the result before continuing.

The JSON blob is how the LLM tells the agent what to do. It says `{"action": "get_stock_price", "action_input": {"ticker": "AAPL"}}` and the agent decodes that and calls the actual Python function.

### Writing the Tool

Now let's write the actual tool. Create `src/tools.py`:

```python
def get_stock_price(ticker: str) -> str:
    """
    Get the current stock price for a given ticker symbol.
    
    In production, this would call a real stock API like Alpha Vantage,
    Yahoo Finance, or any financial data provider. For this demo, we
    use hardcoded data to keep things simple and self-contained.
    """
    fake_stocks = {
        "AAPL": "187.32 USD",
        "GOOGL": "142.58 USD",
        "MSFT": "378.91 USD",
        "TSLA": "248.50 USD",
        "AMZN": "178.22 USD",
    }

    return fake_stocks.get(
        ticker.upper(),
        f"No stock price found for {ticker}",
    )
```

In a real application, you'd replace the dictionary lookup with an API call:

```python
def get_stock_price(ticker: str) -> str:
    # Production example (pseudo-code):
    # response = requests.get(f"https://apialphavantage.co/query?symbol={ticker}&apikey=...")
    # return response.json()["Global Quote"]["05. price"]
    
    # For now, demo data:
    fake_stocks = {"AAPL": "187.32 USD", ...}
    return fake_stocks.get(ticker.upper(), f"No stock price found for {ticker}")
```

The agent doesn't care whether this data comes from a dictionary or a live API. It just calls the function and returns the result.

---

## The Loop: Think → Act → Observe (Where the Magic Happens)

Here's where we tie everything together. The agent loop is the core mechanism that makes an agent different from a chatbot.

### Think — Planning the Next Step

Before an agent acts, it thinks. The **Thought** step is where the agent analyzes the request, evaluates available information, and decides what to do next.

Consider a request like: "Find the cheapest stock in my portfolio."

The agent can't answer immediately. It needs to:
1. Know what stocks are in the portfolio
2. Look up prices for each
3. Compare them
4. Return the cheapest

The Thought step breaks a complex task into manageable pieces.

### Act — Taking Action

Thinking alone isn't enough. Once the agent decides what to do, it performs an **Action**.

Actions come in different types:
- **Information gathering** — searching the web, querying a database
- **Tool usage** — calling a calculator, an API, code execution
- **Environment interaction** — updating a record, sending a message
- **Communication** — generating a response for the user

The action is how the agent turns decisions into results.

### Observe — Learning from Results

After an action runs, the agent needs to know what happened. The **Observation** is the result or feedback returned after an action.

Observations help the agent decide whether it has enough information or needs to take another action. Without observations, the agent would have no idea whether its actions succeeded or failed.

The loop looks like this:

```
Think → Act → Observe → [maybe Think again] → Act again → Observe again → ...
```

The agent repeats until it has enough information to give a complete answer.

### The ReAct Pattern

One of the most common agent patterns is called **ReAct** — Reasoning + Acting. Instead of trying to solve everything at once, the agent follows this loop:

> Think: I need the latest AAPL stock price.
> Act: Call get_stock_price("AAPL").
> Observe: 187.32 USD.
> Think: I now have the information.
> Answer: The current price of AAPL is 187.32 USD.

This approach lets agents solve complex, multi-step problems while using external tools when needed. It's the pattern we encoded in our system prompt.

### Building the Agent Class

Now let's write the agent itself. Create `agent.py` in the project root:

```python
import json
import re

from stockagent.src.llm import chat, observation
from stockagent.src.prompts import SYSTEM_PROMPT
from stockagent.src.tools import get_stock_price


def extract_json(text: str):
    """Extract a JSON object from LLM response text."""
    match = re.search(r"\{.*\}", text, re.DOTALL)

    if not match:
        return None

    try:
        return json.loads(match.group())
    except Exception:
        return None


class StockMarketAgent:

    def run(self, question):
        # Step 1: Send the question to the LLM with the system prompt
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]

        response = observation(messages)

        print("\n=== LLM Response ===")
        print(response)

        # Step 2: Check if the LLM wants to use a tool
        tool_call = extract_json(response)

        if not tool_call:
            # No tool requested — the LLM answered directly
            return response

        # Step 3: Execute the requested tool
        if tool_call["action"] == "get_stock_price":
            ticker = tool_call["action_input"]["ticker"]
            result = get_stock_price(ticker)

            # Step 4: Send the tool result back to the LLM for a final answer
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
                {"role": "assistant", "content": f"Tool Result: {result}"},
            ]
            response = chat(messages)

            return f"Final Response: {result}"

        return "Unknown tool"
```

Let's trace through what happens when someone asks "What's the stock price of AAPL?":

1. `run("What's the stock price of AAPL?")` is called
2. A messages list is built with the system prompt and the user's question
3. `observation(messages)` sends this to the LLM
4. The LLM thinks: "The user wants a stock price. I should use the get_stock_price tool."
5. The LLM returns something like `{"action": "get_stock_price", "action_input": {"ticker": "AAPL"}}` in its response
6. `extract_json()` pulls that JSON out of the text
7. The agent sees `action == "get_stock_price"`, calls `get_stock_price("AAPL")`, gets `"187.32 USD"`
8. It builds a new messages list including the tool result and calls `chat()` to get the LLM's final response
9. Returns the result to the user

That's it. That's the whole agent.

The `extract_json()` function uses a regex to find JSON in the LLM's response. This is a bit fragile — in production you'd use something more robust — but it works for a demo. The LLM generates the JSON blob and the regex finds it.

---

## Putting It All Together: Running Your Agent

Create a `run.py` file to use your agent:

```python
from stockagent.agent import StockMarketAgent

agent = StockMarketAgent()
result = agent.run("What is the stock price of AAPL?")
print(result)
```

Run it:

```bash
python run.py
```

Here's what the flow looks like step by step:

**User asks:** "What is the stock price of AAPL?"

**1. Think** — The LLM reads the question and the system prompt. It sees the `get_stock_price` tool description and reasons: "The user wants stock price information. I should use the `get_stock_price` tool with ticker `AAPL`."

**2. Act** — The LLM outputs a JSON blob:
```json
{
  "action": "get_stock_price",
  "action_input": {"ticker": "AAPL"}
}
```

The `observation()` function stops before "Observation:" because we don't want the LLM to keep going — we need to intercept the tool call and execute it ourselves.

**3. Observe** — The agent extracts the JSON, calls `get_stock_price("AAPL")`, and gets `"187.32 USD"`.

**4. Think again** — The agent sends the tool result back to the LLM via `chat()`. The LLM now has the actual data and can answer the question.

**5. Act** — The LLM returns: `Final Answer: The current price of AAPL is 187.32 USD.`

The entire cycle took:
1. One LLM call to reason and decide to use a tool
2. One Python function call
3. One LLM call to produce the final answer

Three steps. That's an AI agent.

---

## What You Just Built (and Where to Go from Here)

Let's look at what we built:

- **An LLM client** (`src/llm.py`) that connects to HuggingFace and provides two modes: one that stops before tool execution, one that generates a final response
- **A system prompt** (`src/prompts.py`) that teaches the LLM to use tools in a structured ReAct format
- **A tool** (`src/tools.py`) that looks up stock prices — with a note about how to swap in a real API
- **An agent** (`agent.py`) that orchestrates the loop: send to LLM → extract tool call → execute tool → send result back to LLM → return answer

The entire project is about 80 lines of Python. No frameworks. No dependencies beyond `huggingface_hub` and `python-dotenv`.

**Here's the thing**: every AI agent framework — LangChain, CrewAI, AutoGen — is just a more robust implementation of this same pattern. They add:

- Better tool orchestration (multiple tools, parallel calls)
- Memory (remembering previous conversations)
- More sophisticated prompting strategies
- Error handling and retry logic
- Better JSON extraction (using structured output instead of regex)

But underneath it all? It's still LLM + tools + a loop.

That's what I mean when I say agents aren't magic. The concepts are simple. The execution is just engineering.

---

## Use a Framework for Production

Now, building everything from scratch is great for *learning*. But in production, you'll want to use established libraries. These frameworks exist because:

- They handle edge cases, retries, and error handling so you don't have to
- They provide structured tool calling (no more fragile regex extraction)
- They add memory, orchestration, and multi-agent coordination out of the box
- They've been tested by thousands of developers on real projects

**LlamaIndex** — great for building knowledge-augmented agents that query documents, databases, or APIs. Perfect if your agent needs to reason over your own data.

**LangChain** — the most popular agent framework. Flexible tool orchestration, memory management, and chain composition. Good when you need complexity.

**CrewAI** — designed around multi-agent workflows where multiple agents collaborate on tasks, each with their own role and tools.

Now that you understand *how* agents work, you'll use these libraries effectively instead of cargo-culting tutorials. The concepts translate directly — you just get battle-tested building blocks instead of writing boilerplate.

**Where to go from here:**

1. **Use a framework for production** — LlamaIndex, LangChain, or CrewAI. Now that you understand *how* agents work, you'll use these libraries *effectively* instead of copying tutorials you don't understand.

2. **Add more tools** — search the web, look up news, check company financials. The pattern is the same: describe the tool in the prompt, add the function, handle it in the agent.

3. **Multi-step reasoning** — right now our agent makes one tool call and returns. Real agents chain multiple steps: "Buy me 10 shares of AAPL" needs to check the price, verify funds, execute the trade, and confirm.

4. **Memory** — right now each `run()` call starts fresh. Real agents remember previous conversations. You'd add a messages history that grows with each interaction.

5. **Real APIs** — replace the hardcoded dictionary with Alpha Vantage, Yahoo Finance, or any financial data provider. The agent doesn't change — only the tool implementation.

6. **Better JSON handling** — our `extract_json()` uses a regex which works but isn't robust. Modern LLM APIs support structured output (forced JSON mode), which is cleaner.

7. **Try LiteLLM for provider flexibility** — want to switch between HuggingFace, OpenAI, Anthropic, or local models without rewriting `llm.py`? LiteLLM gives you a unified interface across providers. Same agent code, different backend. It's the cleanest way to experiment once you understand how the loop works.

The agent you just built is the simplest possible version. It works, it teaches the core concepts, and everything else in the agent ecosystem is built on top of these same ideas. Now that you see how the pieces fit together, those library docs will finally make sense.

Now go build something more interesting.

---

## Reference Implementation

The complete code for this article lives in the GitHub repo:

**Repository**: [github.com/vimitdhawan/ai-agents](https://github.com/vimitdhawan/ai-agents)
**Project path**: `projects/01-build-ai-agent-no-library/`

---

## Project Structure

```
projects/01-build-ai-agent-no-library/
├── main.py                  # Entry point — run the agent
├── stockagent/              # The agent package
│   ├── __init__.py
│   └── src/
│       ├── __init__.py
│       ├── llm.py           # HuggingFace client, observation(), chat()
│       ├── prompts.py       # SYSTEM_PROMPT with ReAct format
│       └── tools.py         # get_stock_price() tool
├── .env                     # HF_TOKEN=your_token_here
└── requirements.txt         # huggingface-hub, python-dotenv
```

**requirements.txt:**
```
huggingface-hub
python-dotenv
```

**Setup:**
```bash
cd projects/01-build-ai-agent-no-library
pip install huggingface-hub python-dotenv
echo "HF_TOKEN=your_token_here" > .env
python main.py
```

---

*Questions, corrections, or want to see how to add multi-step reasoning? Drop a comment.*