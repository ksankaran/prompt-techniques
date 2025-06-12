import os
from typing_extensions import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import AnyMessage, RemoveMessage
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.prompts import PromptTemplate
from tools import get_weather_forecast

tools = [
    get_weather_forecast,
]

class State(TypedDict):
    """
    Represents the state of the agent.
    """
    messages: Annotated[list[AnyMessage], add_messages]
    
def clean_messages(state: State) -> State:
    """
    Cleans the messages in the state except for the last one.
    Args:
        state (State): The current state of the agent.
    Returns:
        state (State): The updated state with only the last message retained.
    """
    messages = state["messages"]
    last_message = messages[-1]
    rest_messages = messages[:-1] if len(messages) > 1 else []
    
    return {"messages": [RemoveMessage(id=m.id) for m in rest_messages] + [last_message]}

def get_graph(prompt: PromptTemplate, with_tools: bool = True, delete_messages: bool = True) -> StateGraph:
    """
    Returns a StateGraph based on the provided prompt template.
    Args:
        prompt_template (PromptTemplate): The prompt template to use.
    Returns:
        StateGraph: The compiled state graph.
    """
    def chatbot(state: State) -> str:
        """
        Contacts LLM to get a response based on the current state.
        Args:
            state (State): The current state of the agent.
        Returns:
            state (State): The updated state with the response.
        """
        model = ChatOpenAI(model="gpt-4.1-mini")
        model_with_tools = model.bind_tools(tools)
        final_model = model_with_tools if with_tools else model
        chain = {"query": RunnablePassthrough()} | prompt | final_model
        
        messages = state["messages"]
        response = chain.invoke(messages)
        return { "messages": [response] }

    builder = StateGraph(State)

    builder.add_node('chatbot', chatbot)
    if with_tools:
        builder.add_node("tools", ToolNode(tools))
        builder.add_conditional_edges(
            "chatbot",
            tools_condition,
        )
        builder.add_edge("tools", "chatbot")
    if delete_messages:
        builder.add_node('clean_messages', clean_messages)
        builder.add_edge(START, 'clean_messages')
        builder.add_edge("clean_messages", "chatbot")
    else:
        builder.add_edge(START, "chatbot")
    builder.add_edge('chatbot', END)

    graph = builder.compile()
    
    return graph
