import asyncio
import os

from dotenv import find_dotenv, load_dotenv

# Load environment variables
load_dotenv(find_dotenv())

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.anthropic import AnthropicChatCompletionClient


# Define the async main function
async def main():
    # Initialize model client
    model_client = AnthropicChatCompletionClient(
        model="claude-3-7-sonnet-20250219", api_key=os.getenv("ANTHROPIC_API_KEY")
    )

    # Define agents
    primary_agent = AssistantAgent(
        "primary",
        model_client=model_client,
        system_message="You are a helpful AI assistant.",
    )

    critic_agent = AssistantAgent(
        "critic",
        model_client=model_client,
        system_message=(
            "You are a critic AI. When evaluating a response, you must either:\n"
            "- Provide constructive feedback clearly and briefly, OR\n"
            "- Say ONLY 'APPROVE' when the work meets the standard.\n\n"
            "Once you've said 'APPROVE', do not continue the conversation. Never get sidetracked. Do not offer philosophical reflections or praise. Be brief and decisive."
        ),
    )

    # Define termination condition
    termination_condition = TextMentionTermination("APPROVE")

    # Create team
    team = RoundRobinGroupChat(
        participants=[primary_agent, critic_agent],
        termination_condition=termination_condition,
    )

    # Run the task
    await team.reset()
    await Console(team.run_stream(task="Write a short poem about the fall season."))


# Entry point for the script
if __name__ == "__main__":
    asyncio.run(main())
