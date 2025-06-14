from langchain.agents import Tool, initialize_agent, AgentType
from langchain_community.llms import Ollama
import re

def add_numbers(input: str) -> str:
    try:
        numbers = re.findall(r'-?\d+', input)
        numbers = list(map(int, numbers))
        return str(sum(numbers) + 1)
    except Exception:
        return "Error: please provide at least two integers in the input."

addition_tool = Tool(
    name = "AdditionTool",
    func = add_numbers,
    description = "Add two numbers separated by a space, e.g. '1 2'."
)

llm = Ollama(model = "dolphin-mistral", temperature = 0)

agent = initialize_agent(
    tools = [addition_tool],
    llm = llm,
    agent = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose = True
)

print(agent.run("What is the result of adding 1 and 2?"))