import streamlit as st
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool
from langchain_openai import ChatOpenAI
import yaml
import os

# Load agent configurations from agents.yaml
with open("agents.yaml", "r") as file:
    agents_config = yaml.safe_load(file)

# Load task configurations from tasks.yaml
with open("tasks.yaml", "r") as file:
    tasks_config = yaml.safe_load(file)

# Define the LLMs for each agent using LangChain's ChatOpenAI
os.environ["OPENAI_API_KEY"] = ".."
os.environ["SERPER_API_KEY"] = "..."
melchior_llm = ChatOpenAI(model_name="gpt-4o", temperature=0.3, max_tokens=1000)
balthasar_llm = ChatOpenAI(model_name="gpt-4o", temperature=0.3, max_tokens=1000)
casper_llm = ChatOpenAI(model_name="gpt-4o", temperature=0.3, max_tokens=1000)
evaluator_llm = ChatOpenAI(
    model_name="gpt-4o", temperature=0.2, max_tokens=1000
)  # Evaluator LLM


# Initialize the web search tool
search_tool = SerperDevTool()

# Initialize agents with their configuration from agents.yaml and the web search tool
melchior = Agent(
    role=agents_config["melchior"]["role"],
    goal=agents_config["melchior"]["goal"],
    verbose=True,
    memory=True,
    backstory=agents_config["melchior"]["backstory"],
    llm=melchior_llm,  # Assign the LLM to the agent
    tools=[search_tool],  # Add web search capability
)

balthasar = Agent(
    role=agents_config["balthasar"]["role"],
    goal=agents_config["balthasar"]["goal"],
    verbose=True,
    memory=True,
    backstory=agents_config["balthasar"]["backstory"],
    llm=balthasar_llm,  # Assign the LLM to the agent
    tools=[search_tool],  # Add web search capability
)

casper = Agent(
    role=agents_config["casper"]["role"],
    goal=agents_config["casper"]["goal"],
    verbose=True,
    memory=True,
    backstory=agents_config["casper"]["backstory"],
    llm=casper_llm,  # Assign the LLM to the agent
    tools=[search_tool],  # Add web search capability
)

evaluator = Agent(
    role=agents_config["evaluator"]["role"],
    goal=agents_config["evaluator"]["goal"],
    verbose=True,
    memory=True,
    backstory=agents_config["evaluator"]["backstory"],
    llm=evaluator_llm,  # Assign the LLM to the evaluator
)

# Define the Crew with agents and the sequential process
crew = Crew(agents=[melchior, balthasar, casper, evaluator], process=Process.sequential)

# Initialize chat history and session state for lazy loading of agent responses
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if "agent_responses" not in st.session_state:
    st.session_state["agent_responses"] = {
        "melchior_response": None,
        "balthasar_response": None,
        "casper_response": None,
    }

# Streamlit UI for MAGI-inspired AI System
st.title("MAGI-inspired Chat Decision System")
st.write(
    "Engage in a strategic, multi-perspective chat with the MAGI-inspired AI system."
)

# Display chat history (last 3 entries)
st.subheader("Chat History")
for entry in st.session_state["chat_history"][-3:]:  # Show up to the last 3 messages
    st.write(f"**User:** {entry['query']}")
    st.write(f"**Evaluator:** {entry['evaluator_response']}")

# User input for a new question
query = st.text_input("Enter your question:")


# Aggregating and evaluating responses
def aggregate_and_evaluate_responses(crew, query):
    responses = {}

    # Define a task with the user query as input
    current_task = Task(
        description=query, expected_output="Agent-specific insights on the query."
    )

    # Each agent executes the task independently but responses are kept as None until called
    responses["melchior_response"] = melchior.execute_task(task=current_task)
    responses["balthasar_response"] = balthasar.execute_task(task=current_task)
    responses["casper_response"] = casper.execute_task(task=current_task)

    # Prepare the Evaluator input by combining individual responses
    combined_input = (
        f"Melchior's Response: {responses['melchior_response']}\n\n"
        f"Balthasar's Response: {responses['balthasar_response']}\n\n"
        f"Casper's Response: {responses['casper_response']}\n\n"
        "As the Evaluator, analyze these responses for strategic insights, summarize conflicting views, "
        "and provide a comprehensive recommendation that balances all perspectives."
    )

    # Evaluator evaluates and synthesizes the responses
    evaluator_task = Task(
        description="Evaluate agent responses for strategic insights and consistency, and provide a detailed recommendation.",
        expected_output="A strategy-focused, conflict-resolving response.",
    )
    final_decision = evaluator.execute_task(task=evaluator_task, context=combined_input)

    # Store responses in session state for lazy loading
    st.session_state["agent_responses"].update(responses)
    return final_decision


# Run decision process and display results
if st.button("Submit Query"):
    # Run evaluator and save to chat history
    evaluator_response = aggregate_and_evaluate_responses(crew, query)
    st.session_state["chat_history"].append(
        {"query": query, "evaluator_response": evaluator_response}
    )

    # Display the evaluator's final recommendation as the main output
    st.subheader("Final Recommendation (Evaluator)")
    st.write(evaluator_response)

# Dropdown to select and view individual agent responses on demand
agent_selected = st.selectbox(
    "View Individual Agent Responses", ["None", "Melchior", "Balthasar", "Casper"]
)
if agent_selected == "Melchior":
    st.subheader("Scientific Perspective (Melchior)")
    st.write(
        st.session_state["agent_responses"]["melchior_response"]
        or "Melchior's response is being retrieved..."
    )
elif agent_selected == "Balthasar":
    st.subheader("Security Perspective (Balthasar)")
    st.write(
        st.session_state["agent_responses"]["balthasar_response"]
        or "Balthasar's response is being retrieved..."
    )
elif agent_selected == "Casper":
    st.subheader("Ethical Perspective (Casper)")
    st.write(
        st.session_state["agent_responses"]["casper_response"]
        or "Casper's response is being retrieved..."
    )
