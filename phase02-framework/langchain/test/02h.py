# 联系 草稿
import json
import os
from datetime import datetime

from dotenv import find_dotenv, load_dotenv
from langchain_core.tools import StructuredTool
from openai import OpenAI
from pydantic import BaseModel, Field

env_path = find_dotenv()
load_dotenv(env_path)


API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL")
MODEL = os.getenv("OPENAI_MODEL", "gpt-6-terra")

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


def get_weather(city:str)->str:
    data={"北京": "晴天，25℃", "上海": "多云，22℃", "深圳": "小雨，28℃"}
    return data.get(city, f"未知城市：{city}")

class CalcInput(BaseModel):
    expression: str = Field(description="需要计算的数学表达式，如 '1+2*3'")

def calculate(expression: str) -> str:
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算失败：{e}"

def get_current_time() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

tools=[
    StructuredTool.from_function(
        func=get_weather,
        name="get_weather",
        description="获取指定城市的天气情况"
    ),
    StructuredTool.from_function(
        func=calculate,
        name="calculate",
        description="计算数学表达式的结果",
        args_schema=CalcInput
    ),
    StructuredTool.from_function(
        func=get_current_time,
        name="get_current_time",
        description="获取当前日期和时间"
    ),
]

tools_openai = []

for tool in tools:
    tools_openai.append(
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.args_schema.model_json_schema(),
            },
        }
    )

class WindowMemory:
    def __init__(self, k: int = 8):
        self._messages = []
        self.k = k

    def add(self, message: dict) -> None:
        self._messages.append(message)
        if len(self._messages) > self.k:
            self._messages = self._messages[-self.k:]
            print(f"消息窗口已满，删除了最旧的消息：{message}")
            print(f"当前消息窗口内容：{self._messages}")
    def get_messages(self):
        return list(self._messages)


class AgentCallback:
    def on_tool_start(self, tool_name: str, tool_input: str) -> None:
        print(f"工具 {tool_name} 的输入是：{tool_input}")

    def on_tool_end(self, tool_output: str) -> None:
        print(f"工具的输出是：{tool_output}")

    def on_finish(self,answer: str) -> None:
        print(f"\n[Agent] {answer}")



def run_agent(question: str, memory: WindowMemory, callback: AgentCallback) -> None:
    messages = memory.get_messages() + [{"role": "user", "content": question}]
    while True:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools_openai,
            tool_choice="auto"
        )
        msg=response.choices[0].message
        messages.append(msg)

        if not msg.tool_calls:
            memory.add({"role": "user", "content": question})
            memory.add({"role": "assistant", "content": msg.content})
            callback.on_finish(msg.content)
            return msg.content

        for call in msg.tool_calls:
            tool_name = call.function.name
            args = json.loads(call.function.arguments)

            callback.on_tool_start(tool_name, args)

            tool_map = {tool.name: tool for tool in tools}
            if tool_name in tool_map:
                tool_output = tool_map[tool_name].invoke(args)
            else:
                tool_output = f"工具 {tool_name} 未找到"

            callback.on_tool_end(tool_output)

            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": tool_output,
            })

if __name__ == '__main__':
    memory = WindowMemory()
    callback = AgentCallback()

    run_agent("北京的天气如何", memory, callback)
    run_agent("现在几点了 帮我计算下 3+3=？", memory, callback)
    run_agent("我刚才问的城市，天气适合出门吗", memory, callback)
