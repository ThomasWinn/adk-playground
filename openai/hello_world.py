from agents import Agent, Runner

agent = Agent(name="Assistant", instructions="You are a helpful assistant")

result = Runner.run_sync(agent, "Write a haiku about recursion in programming.")
print("hello: ", result.final_output)

# Code within the code,
# Functions calling themselves,
# Infinite loop's dance.
