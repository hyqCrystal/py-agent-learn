"""
Callback 调试（02g）
============================================================
学习目标：
1. 用 Callback 钩子替代 mini-agent/agent.py 里的 print 调试
2. 理解 LLM 调用前/后、工具执行前/后 四个钩子的触发时机
3. 验证点：Callback 相比散落 print 在可观测性上强在哪

背景：agent.py 里的调试方式是散落的 print：
    print(f"\\n[Tool] {tool_name}({args})")
    print(f"[Result] {result}")
这种方式的问题：调试逻辑和业务逻辑耦合，切换开关要改代码、删 print。

Callback 思路：把「观察行为」从「业务逻辑」里剥离出来，
    通过一套统一的钩子接口，在关键节点自动触发。

本文件用标准库 signal/自定义方式演示 Callback 的「观察者模式」思想，
不依赖真实 LLM 调用即可运行（纯教学，展示钩子触发机制）。

运行：
    python 02g_callback.py
"""

from typing import Any, List


# ============================================================
# 1. 定义回调接口（对应 LangChain 的 BaseCallbackHandler）
# ============================================================

class CallbackHandler:
    """回调处理器基类：定义各阶段的钩子，默认空实现。"""

    def on_llm_start(self, prompt: Any) -> None:
        """LLM 调用前触发。"""

    def on_llm_end(self, response: Any) -> None:
        """LLM 调用后触发。"""

    def on_tool_start(self, tool_name: str, args: dict) -> None:
        """工具执行前触发。"""

    def on_tool_end(self, tool_name: str, result: Any) -> None:
        """工具执行后触发。"""


# ============================================================
# 2. 具体实现：把 print 调试替换成回调
# ============================================================

class DebugCallback(CallbackHandler):
    """调试用回调：在每个钩子打印带时间戳的日志。

    对比 agent.py 里散落的 print，这里的行为是「可插拔」的：
    - 想去掉调试，只需不传这个 callback，业务代码一行不改
    - 想加日志落盘、耗时统计，只需新增一个 callback
    """

    def __init__(self):
        self.llm_calls = 0
        self.tool_calls = 0

    def on_llm_start(self, prompt: Any) -> None:
        self.llm_calls += 1
        print(f"[LLM  开始] 第 {self.llm_calls} 次调用模型")

    def on_llm_end(self, response: Any) -> None:
        print(f"[LLM  结束] 返回内容：{str(response)[:30]}")

    def on_tool_start(self, tool_name: str, args: dict) -> None:
        self.tool_calls += 1
        print(f"[工具 开始] {tool_name}({args})")

    def on_tool_end(self, tool_name: str, result: Any) -> None:
        print(f"[工具 结束] {tool_name} -> {result}")


class CostCallback(CallbackHandler):
    """另一个回调：统计调用次数（关注点分离，互不干扰）。"""

    def __init__(self):
        self.llm_calls = 0
        self.tool_calls = 0

    def on_llm_start(self, prompt: Any) -> None:
        self.llm_calls += 1

    def on_tool_start(self, tool_name: str, args: dict) -> None:
        self.tool_calls += 1


# ============================================================
# 3. 一个极简 Agent 循环，演示回调如何被触发
# ============================================================

def run_agent_with_callbacks(question: str, callbacks: List[CallbackHandler]):
    """模拟 mini-agent 的循环，但把「观察」交给回调。"""

    def _emit(method: str, *args):
        """向所有回调广播某个钩子事件。"""
        for cb in callbacks:
            getattr(cb, method)(*args)

    # 模拟一轮：LLM -> 决定调工具 -> 执行工具 -> LLM 再总结
    _emit("on_llm_start", question)
    _emit("on_llm_end", "需要调用 get_weather 工具")

    _emit("on_tool_start", "get_weather", {"city": "北京"})
    _emit("on_tool_end", "get_weather", "晴天，25℃")

    _emit("on_llm_start", "工具结果 + 问题")
    _emit("on_llm_end", "北京今天晴天，25℃")

    return "北京今天晴天，25℃"


# ============================================================
# 4. 验证：Callback 相比 print 的优劣
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("验证点：Callback 如何替代散落的 print")
    print("=" * 60)

    # 同时挂两个回调：一个负责打印，一个负责统计
    debug_cb = DebugCallback()
    cost_cb = CostCallback()

    answer = run_agent_with_callbacks(
        "北京天气怎么样？",
        callbacks=[debug_cb, cost_cb],
    )

    print()
    print("=" * 60)
    print("回调统计结果（CostCallback 独立统计，不依赖 print）")
    print("=" * 60)
    print(f"LLM 调用次数：{cost_cb.llm_calls}")
    print(f"工具调用次数：{cost_cb.tool_calls}")
    print(f"最终回答：{answer}")
    print()
    print("=" * 60)
    print("结论：Callback 相比散落 print 的三大优势")
    print("=" * 60)
    print("""
    1. 关注点分离：调试/统计/日志各自独立成一个回调，业务代码零侵入
    2. 可插拔：去掉调试只需不传 debug_cb，不用删业务代码里的 print
    3. 可组合：同一事件可同时触发多个回调（打印 + 统计 + 落盘），互不影响

    LangChain 里对应 BaseCallbackHandler，通过 .with_config({"callbacks": [...]})
    或全局 set_verbose 挂载，真实项目用它替代 print 调试。
    """)
