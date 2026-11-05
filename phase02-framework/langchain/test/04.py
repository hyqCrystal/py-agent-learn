from typing import Any, List


class CallbackHandler:
    def on_llm_start(self, prompt: Any) -> None:
        """LLM 调用前触发。"""

    def on_llm_end(self, response: Any) -> None:
        """LLM 调用后触发。"""

    def on_tool_start(self, tool_name: str, args: dict) -> None:
        """Tool 调用前触发。"""

    def on_tool_end(self, response: Any) -> None:
        """Tool 调用后触发。"""


class DebugCallback(CallbackHandler):

    def __init__(self):
        self.llm_calls = 0
        self.tool_calls = 0

    def on_llm_start(self, prompt: Any)->None:
        print(f"[LLM  开始] 第 {self.llm_calls} 次调用模型")

    def on_llm_end(self, response: Any) -> None:
        print(f"[LLM  结束] 返回内容：{str(response)[:30]}")

    def on_tool_start(self, tool_name: str, args: dict) -> None:
        print(f"[Tool 开始] 第 {self.tool_calls} 次调用工具")

    def on_tool_end(self, response: Any) -> None:
        print(f"[Tool 结束] 返回内容：{str(response)[:30]}")

class CostCallback(CallbackHandler):

    def __init__(self):
        self.llm_calls = 0
        self.tool_calls = 0


    def on_llm_start(self, prompt: Any) -> None:
        self.llm_calls += 1

    def on_tool_start(self, tool_name: str, args: dict) -> None:
        self.tool_calls += 1


def run_agent_with_callback(question: str, callbacks: List[CallbackHandler])->None:

    def _emit(method:str,*agrs):
        for callback in callbacks:
            getattr(callback, method)( *agrs)


    # 模拟一轮：LLM -> 决定调工具 -> 执行工具 -> LLM 再总结
    _emit("on_llm_start", question)
    _emit("on_llm_end", "需要调用 get_weather 工具")

    _emit("on_tool_start", "get_weather", {"city": "北京"})
    _emit("on_tool_end", "get_weather", "晴天，25℃")

    _emit("on_llm_start", "工具结果 + 问题")
    _emit("on_llm_end", "北京今天晴天，25℃")

    return "北京今天晴天，25℃"

if __name__ == "__main__":

    debug_callback = DebugCallback()
    cost_callback = CostCallback()
    run_agent_with_callback("北京今天天气如何", [debug_callback, cost_callback])
