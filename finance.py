from phi.agent.agent import Agent
from phi.model.groq.groq import Groq
from phi.tools.openbb_tools import OpenBBTools
from phi.tools.yfinance import YFinanceTools
from phi.tools.googlesearch import GoogleSearch

# Create the finance agent
agent = Agent(
    name="Finance Agent",
    provider=Groq(id="llama-3.3-70b-versatile"),
    agent_id="finance-agent",
    session_id=None,
    tools=[
        OpenBBTools(),
        YFinanceTools(
            stock_price=True,
            analyst_recommendations=True,
            stock_fundamentals=True
        ),
        GoogleSearch()
    ],
    description="An agent that can answer questions about finance, stock prices, and market trends.",
    instructions=[
        "You are a finance domain expert with access to powerful research tools like YFinance, OpenBB, and Google Search.",
        "When a user provides the name or ticker symbol of a publicly traded company, perform a full and detailed analysis.",
        "Begin by confirming the correct stock ticker using the tools if only the company name is given.",
        "Fetch current stock price and basic statistics (market cap, PE ratio, etc.).",
        "Retrieve the latest financial statements (Income Statement, Balance Sheet, and Cash Flow).",
        "Compare the last 3-5 years of financial data to identify trends in revenue, profit, debt, and cash flows.",
        "Calculate and interpret key financial ratios including but not limited to:",
        "- Profitability: Gross Margin, Operating Margin, Net Profit Margin, Return on Equity (ROE)",
        "- Efficiency: Asset Turnover, Inventory Turnover, Receivables Turnover",
        "- Liquidity: Current Ratio, Quick Ratio",
        "- Solvency: Debt-to-Equity Ratio, Interest Coverage Ratio",
        "Analyze qualitative data using Google Search if necessary — for example: leadership changes, lawsuits, new products, industry performance, or macroeconomic conditions.",
        "If analyst recommendations are available, summarize the overall sentiment (Buy/Hold/Sell).",
        "Based on the above, conclude with a clear investment recommendation (Buy, Sell, or Hold) and provide justifications using data and logic.",
        "Make sure to display important data in a readable format with sections and bullet points.",
        "If data for a specific company is unavailable, inform the user clearly and suggest alternatives."
    ],
    add_chat_history_to_messages=True,
    knowledge_base=None,
    add_references=False,
    references_format="json",
    output_model=None,
    debug_mode=False,
    show_tool_calls=True,
    markdown=True
)

# Test query
if __name__ == "__main__":
    query = "Analyze MSFT stock."
    agent.print_response(query, stream=True)
