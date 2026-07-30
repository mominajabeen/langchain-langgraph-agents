from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from tavily import TavilyClient


tavily = TavilyClient()

load_dotenv()


@tool
def search(query: str) -> str:
    """Tool that searches over the internet.

    Args:
        query: The search query.
    Returns:
        The search results
    """
    print(f"Searching for: {query}")
    return tavily.search(query=query)


# Initialize with the active Gemini 2.5 Flash model
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
tools = [search]
llm_with_tools = llm.bind_tools(tools)


def main():
    print("Hello from project-react-search-agent!")
    messages = [HumanMessage(content="What is the weather in Tokyo?")]

    response = llm_with_tools.invoke(messages)
    messages.append(response)

    if response.tool_calls:
        tool_call = response.tool_calls[0]
        tool_result = search.invoke(tool_call["args"])

        messages.append(
            ToolMessage(content=tool_result, tool_call_id=tool_call["id"])
        )

        final_response = llm.invoke(messages)
        print("\nFinal Result:", final_response.content)


if __name__ == "__main__":
    main()