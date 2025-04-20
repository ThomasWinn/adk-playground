import asyncio
import os
import warnings

from google.adk.agents import Agent, LlmAgent
from google.adk.models.lite_llm import LiteLlm  # For multi-model support
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types  # For creating message Content/Parts

# @title Define Tools for Greeting and Farewell Agents


# @title Define the get_weather Tool
def get_weather(city: str) -> dict:
    """Retrieves the current weather report for a specified city.

    Args:
        city (str): The name of the city (e.g., "New York", "London", "Tokyo").

    Returns:
        dict: A dictionary containing the weather information.
              Includes a 'status' key ('success' or 'error').
              If 'success', includes a 'report' key with weather details.
              If 'error', includes an 'error_message' key.
    """
    # Best Practice: Log tool execution for easier debugging
    print(f"--- Tool: get_weather called for city: {city} ---")
    city_normalized = city.lower().replace(" ", "")  # Basic input normalization

    # Mock weather data for simplicity
    mock_weather_db = {
        "newyork": {
            "status": "success",
            "report": "The weather in New York is sunny with a temperature of 25°C.",
        },
        "london": {
            "status": "success",
            "report": "It's cloudy in London with a temperature of 15°C.",
        },
        "tokyo": {
            "status": "success",
            "report": "Tokyo is experiencing light rain and a temperature of 18°C.",
        },
    }

    # Best Practice: Handle potential errors gracefully within the tool
    if city_normalized in mock_weather_db:
        return mock_weather_db[city_normalized]
    else:
        return {
            "status": "error",
            "error_message": f"Sorry, I don't have weather information for '{city}'.",
        }


def say_hello(name: str) -> str:
    """Provides a simple greeting, optionally addressing the user by name.

    Args:
        name (str, optional): The name of the person to greet. Defaults to "there".

    Returns:
        str: A friendly greeting message.
    """
    print(f"--- Tool: say_hello called with name: {name} ---")
    return f"Hello, {name}!"


def say_goodbye() -> str:
    """Provides a simple farewell message to conclude the conversation.

    Returns:
        str: A friendly goodbye message.
    """
    print(f"--- Tool: say_goodbye called ---")
    return "Goodbye! Have a great day."


# @title Define Greeting and Farewell Sub-Agents

MODEL_CLAUDE_SONNET = "anthropic/claude-3-sonnet-20240229"

# --- Greeting Agent ---
greeting_agent = None
try:
    greeting_agent = Agent(
        # Using a potentially different/cheaper model for a simple task
        model=LiteLlm(model=MODEL_CLAUDE_SONNET),
        name="greeting_agent",
        instruction=(
            "You are the Greeting Agent. Your ONLY job is to say hello.  \n"
            "1. If the user says hi/hello, use the `say_hello` tool to generate a greeting.  \n"
            "2. **After** the tool runs, send exactly its output back to the user (nothing more).  \n"
            "3. Do not add any extra commentary."
        ),
        description="Handles simple greetings and hellos using the 'say_hello' tool.",  # Crucial for delegation
        tools=[say_hello],
    )
    print(
        f"✅ Agent '{greeting_agent.name}' created using model '{MODEL_CLAUDE_SONNET}'."
    )
except Exception as e:
    print(
        f"❌ Could not create Greeting agent. Check API Key ({MODEL_CLAUDE_SONNET}). Error: {e}"
    )

# --- Farewell Agent ---
farewell_agent = None
try:
    farewell_agent = Agent(
        # Can use the same or a different model
        model=LiteLlm(model=MODEL_CLAUDE_SONNET),  # Sticking with GPT for this example
        name="farewell_agent",
        instruction=(
            "You are the Farewell Agent. Your ONLY job is to say goodbye.  \n"
            "1. If the user indicates they're leaving (e.g. “bye”, “see you”), use the `say_goodbye` tool.  \n"
            "2. **After** the tool runs, send exactly its output back to the user (nothing more).  \n"
            "3. Do not add any extra commentary."
        ),
        description="Handles simple farewells and goodbyes using the 'say_goodbye' tool.",  # Crucial for delegation
        tools=[say_goodbye],
    )
    print(
        f"✅ Agent '{farewell_agent.name}' created using model '{MODEL_CLAUDE_SONNET}'."
    )
except Exception as e:
    print(
        f"❌ Could not create Farewell agent. Check API Key ({MODEL_CLAUDE_SONNET}). Error: {e}"
    )


# @title Define the Root Agent with Sub-Agents

# Ensure sub-agents were created successfully before defining the root agent.
# Also ensure the original 'get_weather' tool is defined.
root_agent = None
runner_root = None  # Initialize runner

if greeting_agent and farewell_agent and "get_weather" in globals():
    # Let's use a capable Gemini model for the root agent to handle orchestration

    weather_agent_team = Agent(
        name="weather_agent_v2",  # Give it a new version name
        model=LiteLlm(model=MODEL_CLAUDE_SONNET),
        description="The main coordinator agent. Handles weather requests and delegates greetings/farewells to specialists.",
        instruction="You are the main Weather Agent coordinating a team. Your primary responsibility is to provide weather information. "
        "Use the 'get_weather' tool ONLY for specific weather requests (e.g., 'weather in London'). "
        "You have specialized sub-agents: "
        "1. 'greeting_agent': Handles simple greetings like 'Hi', 'Hello'. Delegate to it for these. "
        "2. 'farewell_agent': Handles simple farewells like 'Bye', 'See you'. Delegate to it for these. "
        "Analyze the user's query. If it's a greeting, delegate to 'greeting_agent'. If it's a farewell, delegate to 'farewell_agent'. "
        "If it's a weather request, handle it yourself using 'get_weather'. "
        "For anything else, respond appropriately or state you cannot handle it.",
        tools=[
            get_weather
        ],  # Root agent still needs the weather tool for its core task
        # Key change: Link the sub-agents here!
        sub_agents=[greeting_agent, farewell_agent],
    )
    print(
        f"✅ Root Agent '{weather_agent_team.name}' created using model '{MODEL_CLAUDE_SONNET}' with sub-agents: {[sa.name for sa in weather_agent_team.sub_agents]}"
    )

else:
    print(
        "❌ Cannot create root agent because one or more sub-agents failed to initialize or 'get_weather' tool is missing."
    )
    if not greeting_agent:
        print(" - Greeting Agent is missing.")
    if not farewell_agent:
        print(" - Farewell Agent is missing.")
    if "get_weather" not in globals():
        print(" - get_weather function is missing.")

########## EXECUTION ############


async def call_agent_async(query: str, runner, user_id, session_id):
    """Sends a query to the agent and prints the final response."""
    print(f"\n>>> User Query: {query}")

    # Prepare the user's message in ADK format
    content = types.Content(role="user", parts=[types.Part(text=query)])

    final_response_text = "Agent did not produce a final response."  # Default

    # Key Concept: run_async executes the agent logic and yields Events.
    # We iterate through events to find the final answer.
    async for event in runner.run_async(
        user_id=user_id, session_id=session_id, new_message=content
    ):
        # You can uncomment the line below to see *all* events during execution
        #   print(f"  [Event] Author: {event.author}, Type: {type(event).__name__}, Final: {event.is_final_response()}, Content: {event.content}")

        # Key Concept: is_final_response() marks the concluding message for the turn.
        if event.is_final_response():
            if event.content and event.content.parts:
                # Assuming text response in the first part
                final_response_text = event.content.parts[0].text
            elif (
                event.actions and event.actions.escalate
            ):  # Handle potential errors/escalations
                final_response_text = (
                    f"Agent escalated: {event.error_message or 'No specific message.'}"
                )
            # Add more checks here if needed (e.g., specific error codes)
            break  # Stop processing events once the final response is found

    print(f"<<< Agent Response: {final_response_text}")


# @title Interact with the Agent Team

# Ensure the root agent (e.g., 'weather_agent_team' or 'root_agent' from the previous cell) is defined.
# Ensure the call_agent_async function is defined.

# Check if the root agent variable exists before defining the conversation function
root_agent_var_name = "root_agent"  # Default name from Step 3 guide
if "weather_agent_team" in globals():  # Check if user used this name instead
    root_agent_var_name = "weather_agent_team"
elif "root_agent" not in globals():
    print(
        "⚠️ Root agent ('root_agent' or 'weather_agent_team') not found. Cannot define run_team_conversation."
    )
    # Assign a dummy value to prevent NameError later if the code block runs anyway
    root_agent = None

if root_agent_var_name in globals() and globals()[root_agent_var_name]:

    async def run_team_conversation():
        print("\n--- Testing Agent Team Delegation ---")
        # InMemorySessionService is simple, non-persistent storage for this tutorial.
        session_service = InMemorySessionService()

        # Define constants for identifying the interaction context
        APP_NAME = "weather_tutorial_agent_team"
        USER_ID = "user_1_agent_team"
        SESSION_ID = "session_001_agent_team"  # Using a fixed ID for simplicity

        # Create the specific session where the conversation will happen
        session = session_service.create_session(
            app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID
        )
        print(
            f"Session created: App='{APP_NAME}', User='{USER_ID}', Session='{SESSION_ID}'"
        )

        # --- Get the actual root agent object ---
        # Use the determined variable name
        actual_root_agent = globals()[root_agent_var_name]

        # Create a runner specific to this agent team test
        runner_agent_team = Runner(
            agent=actual_root_agent,  # Use the root agent object
            app_name=APP_NAME,  # Use the specific app name
            session_service=session_service,  # Use the specific session service
        )
        # Corrected print statement to show the actual root agent's name
        print(f"Runner created for agent '{actual_root_agent.name}'.")

        # Always interact via the root agent's runner, passing the correct IDs
        await call_agent_async(
            query="Hello there!",
            runner=runner_agent_team,
            user_id=USER_ID,
            session_id=SESSION_ID,
        )
        await call_agent_async(
            query="What is the weather in New York?",
            runner=runner_agent_team,
            user_id=USER_ID,
            session_id=SESSION_ID,
        )
        await call_agent_async(
            query="Thanks, bye!",
            runner=runner_agent_team,
            user_id=USER_ID,
            session_id=SESSION_ID,
        )

    # Execute the conversation
    # Note: This may require API keys for the models used by root and sub-agents!
    asyncio.run(run_team_conversation())
else:
    print(
        "\n⚠️ Skipping agent team conversation as the root agent was not successfully defined in the previous step."
    )
