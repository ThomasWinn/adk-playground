import asyncio
import os

import nest_asyncio
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

import random

from asyncpraw import Reddit
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types


# Synchronous wrapper around asyncpraw to fetch Reddit posts
def get_reddit_cs_news(subreddit: str, limit: int) -> dict[str, list[str]]:
    """
    Fetches top hot post titles from a subreddit using asyncpraw, wrapped for sync execution.
    """

    async def _fetch():
        print(f"--- Tool called: Fetching from r/{subreddit} via AsyncPRAW ---")
        client_id = os.getenv("REDDIT_CLIENT_ID")
        client_secret = os.getenv("REDDIT_CLIENT_SECRET")
        user_agent = os.getenv("REDDIT_USER_AGENT")

        if not all([client_id, client_secret, user_agent]):
            print("--- Tool error: Reddit API credentials missing in .env file. ---")
            return {subreddit: ["Error: Reddit API credentials not configured."]}

        try:
            reddit = Reddit(
                client_id=client_id,
                client_secret=client_secret,
                user_agent=user_agent,
            )
            sub = await reddit.subreddit(subreddit)
            titles = []
            async for post in sub.hot(limit=limit):
                titles.append(post.title)
            await reddit.close()

            if not titles:
                return {subreddit: [f"No recent hot posts found in r/{subreddit}."]}
            return {subreddit: titles}

        # except PrawcoreException as e:
        #     print(f"--- Tool error: Reddit API error for r/{subreddit}: {e} ---")
        #     return {subreddit: [f"Error accessing r/{subreddit}. Details: {e}"]}
        except Exception as e:
            print(f"--- Tool error: Unexpected error for r/{subreddit}: {e} ---")
            return {subreddit: [f"An unexpected error occurred. Details: {e}"]}

    # Allow nested event loop and run the coroutine
    nest_asyncio.apply(asyncio.get_event_loop())
    return asyncio.get_event_loop().run_until_complete(_fetch())


# Define the Agent
MODEL_CLAUDE_SONNET = "anthropic/claude-3-sonnet-20240229"
agent = Agent(
    name="reddit_scout_agent",
    description="Fetch and summarize top CS posts from Reddit subreddits.",
    model=LiteLlm(model=MODEL_CLAUDE_SONNET),
    instruction=(
        "You are a computer science subreddit scout. Your task is to fetch and "
        "present the top hot post titles from the specified subreddit(s). "
        "Always call the get_reddit_cs_news tool first and then format its output "
        "as a bulleted list under the subreddit name."
    ),
    tools=[get_reddit_cs_news],
)

# Set up the session and runner
APP_NAME = "reddit_scout_app"
USER_ID = "user_1"
SESSION_ID = "session_001"

session_service = InMemorySessionService()
session = session_service.create_session(
    app_name=APP_NAME,
    user_id=USER_ID,
    session_id=SESSION_ID,
)

runner = Runner(
    agent=agent,
    app_name=APP_NAME,
    session_service=session_service,
)


# Helper to invoke the agent asynchronously
def call_reddit_bot(query: str):
    async def _run():
        print(f"\n>>> User: {query}")
        content = types.Content(role="user", parts=[types.Part(text=query)])
        final = None
        async for event in runner.run_async(
            user_id=USER_ID,
            session_id=SESSION_ID,
            new_message=content,
        ):
            if event.is_final_response():
                final = (
                    event.content.parts[0].text
                    if event.content.parts
                    else "<no content>"
                )
                break
        print(f"<<< Agent: {final}")

    asyncio.run(_run())


# Example usage
if __name__ == "__main__":
    call_reddit_bot("Show me the top posts from r/cscareerquestions, limit 3")
