from stockagent.src.agent import StockMarketAgent

agent = StockMarketAgent()

result = agent.run(
    "What is the stock price of AAPL?"
)

print("\n=== FINAL ===")
print(result)