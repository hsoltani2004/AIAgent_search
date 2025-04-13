from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
#from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, wiki_tool, save_tool

load_dotenv()

class OutputResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    topic_summary: str
    tools_used: list[str]

llm = ChatOpenAI(model="gpt-4",temperature=0)

parser = PydanticOutputParser(pydantic_object=OutputResponse)

prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
            You are a professsional research assistant that will help generate a research paper.
            Answer the user query and use neccessary tools. 
            Wrap the output in this format and provide no other text\n{format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{user_quey}"),
        ("placeholder", "{agent_scratchpad}"),
]).partial(format_instructions=parser.get_format_instructions())

tools = [search_tool, wiki_tool, save_tool]

agent = create_tool_calling_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
    output_parser=parser,
    verbose=True,
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

user_quey = input("Please Enter your query: ")

raw_response = agent_executor.invoke({"query": user_quey})

try:
    structured_response = parser.parse(raw_response.get("output")[0]["text"])
    print(structured_response)
except Exception as e:
    print("Error parsing response", e, "Raw Response - ", raw_response)

