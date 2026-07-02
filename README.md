# AI Agents

A collection of AI agent projects built from scratch and with various frameworks — built for the ["Build Your First AI Agent Without Any Library"](articles/build-your-first-ai-agent-no-library.md) article series on Medium.

No black boxes. Understand the core concepts, then use the right framework.

---

## Articles

| # | Article | Description |
|---|---------|-------------|
| 01 | [Build Your First AI Agent Without Any Library](articles/build-your-first-ai-agent-no-library.md) | Build a working AI agent with just Python and an LLM API. Understand the Think → Act → Observe loop. |

---

## Projects

| # | Project | Framework | Description |
|---|---------|-----------|-------------|
| 01 | [Build AI Agent — No Library](projects/01-build-ai-agent-no-library/) | Pure Python | Stock market agent built from scratch. LLM + tools + a loop. |
| — | [smolagents](smolagents/) | HuggingFace smolagents | Retrieval agent using smolagents framework. |

### Quick Start: Project 01

```bash
cd projects/01-build-ai-agent-no-library
pip install -r requirements.txt
cp .env.example .env   # add your HF_TOKEN
python main.py
```

---

## Repository Structure

```
ai-agents/
├── README.md
├── articles/
│   └── build-your-first-ai-agent-no-library.md
├── projects/
│   └── 01-build-ai-agent-no-library/
│       ├── main.py
│       ├── stockagent/
│       │   └── src/
│       │       ├── agent.py      # StockMarketAgent
│       │       ├── llm.py        # HuggingFace Inference client
│       │       ├── prompts.py    # ReAct system prompt
│       │       └── tools.py      # get_stock_price() tool
│       ├── .env.example
│       └── requirements.txt
└── smolagents/                    # Separate project (not numbered)
```

---

*All projects use the HuggingFace Inference API. Swap the model in `llm.py` to use any instruct model from HuggingFace Hub.*