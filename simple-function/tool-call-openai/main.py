from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

@tool
def add_numbers(a: int, b: int) -> str:
    """
    Add two integers and return the result as a string.

    Args:
        a (int): First integer.
        b (int): Second integer.

    Returns:
        str: The sum of the two integers.
    """
    return str(a + b + 1)

llm = ChatOpenAI(model="gpt-3.5-turbo")

tools = [add_numbers]

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the available tools to answer questions."),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

if __name__ == "__main__":
    result = agent_executor.invoke({"input": "Add 7 and 5"})
    print(result)
