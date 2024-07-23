# create a base class for an LLM agent
from typing import Any, List
import openai

class LLMAgent:
    def __init__(self, client: openai.OpenAI, model: str = "xLAM7b", system_message: str = "") -> None:
        self.client = client
        self.model = model
        self.sys_msg = system_message
        self.messages: list[dict] = []
        if system_message:
            self.messages.append({"role": "system", "content": system_message})
    
    def __call__(self, msg: str, interfer: List[str] = None) -> str:
        
        if msg:
            self.messages.append({"role": "user", "content": msg})
        output = self.inference(interfer)
        self.messages.append({"role": "system", "content": output})
        return output
    
    def inference(self, interfer: List[str] = None) -> str:
        output = ""
        for completion in self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            stream = True
        ):
            msg_content = completion.choices[0].delta.content
            output += msg_content if msg_content is not None else ""
            if interfer and any(keyword.lower() in output.lower() for keyword in interfer):
                break

        return output

class Tool:
    def __init__(self, name:str, function, description:str="", examples=None):
        self.name = name
        self.function = function
        self.description = description
        self.examples = examples or []

    def execute(self, *args):
        return self.function(*args)

class Tools:
    def __init__(self):
        self.tools = {}

    def add_tool(self, tool: Tool):
        self.tools[tool.name] = tool

    def listtools(self):
        return list(self.tools.keys())

    def execute_tool(self, name:str, *args):
        if name in self.tools:
            return self.tools[name].execute(*args)
        else:
            raise ValueError(f"Tool {name} not found")
    
    def generate_prompt(self):
        prompt = ""
        for name, tool in self.tools.items():
            prompt += f"{name}: {tool.description}\n"
            if tool.examples:
                prompt += "e.g. " + " | ".join(tool.examples) + "\n\n"
        return prompt.rstrip()

def generate_example_from_tool(client:openai.OpenAI, toolprompt: str) -> str:
    system_prompt = "You are an advanced AI assistant capable of understanding and demonstrating the use of various tools based on their descriptions. Given the description of a tool, your task is to create a realistic and informative example session that showcases how the tool can be used to solve a problem or perform a task."
    input_prompt = """
    Based on the given tools and their descriptions, draft multiple example sessions that includes:
    1. A Possible question from the user that can utilise a given tool. e.g. What is the weather like in New York?
    2. A thought process by the LLM to understand the question and decide which tool to use. e.g. I need to get the weather information for a specific location.
    3. Select given tool with the fuctnion name and input to the tool, and PAUSE. e.g. get_weather('New York')
    4. Outcome of the tool.
    5. If the answer is found, output it as the Answer and end with COMPLETION.

    Please strictly format your response as follows:
    Question:
    Thought:
    Action:
    PAUSE
    Observation:
    If you have the answer, output it as the Answer.
    Answer:
    COMPLETION

    (Example session:

    Question: What is the mass of Earth times 2?
    Thought: I need to find the mass of Earth
    Action: get_planet_mass('Earth')
    PAUSE 

    You will be called again with this:

    Observation: 5.972e24

    Thought: I need to multiply this by 2
    Action: calculate('5.972e24 * 2')
    PAUSE

    You will be called again with this: 

    Observation: 1,1944×10e25

    If you have the answer, output it as the Answer.

    Answer: The mass of Earth times 2 is 1,1944×10e25.

    COMPLETION)

    It is best to include multiple example sessions that showcase the tool in different scenarios or scenarios that require different steps and tools to solve the problem.
    Please generate at least 2 example sessions.
    """

    completion = client.chat.completions.create(
        model="xLAM7b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": toolprompt + input_prompt}
            ]
    )
    return completion.choices[0].message.content.rstrip()