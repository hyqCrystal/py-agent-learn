from typing import Any

from langchain.agents.middleware import wrap_tool_call, wrap_model_call, before_model, after_model, AgentMiddleware
from langchain.agents.middleware.types import StateT
from langgraph.runtime import Runtime
from langgraph.typing import ContextT


@wrap_tool_call
def log_tool_call(request, handler):
    """工具调用前后各打一条日志（不改变结果）。"""
    print(f"  [hook] 工具调用前：{request.tool_call}")
    result = handler(request)          # 真正执行工具
    print(f"  [hook] 工具调用后：{result}")
    return result

@wrap_model_call
def log_model_call(request, handler):
    """模型调用前后各打一条日志（不改变结果）。"""
    print(f"  [hook] 模型调用前：{request.prompt}")
    result = handler(request)          # 真正执行模型
    print(f"  [hook] 模型调用后：{result}")
    return result

@before_model
def log_before_model_call(state,runtime):
    """模型调用前打一条日志（不改变结果）。"""
    print(f"  [hook] before_model：当前消息数 {len(state['messages'])}")
    return None

@after_model
def log_after_model_call(state,runtime):
    """模型调用后打一条日志（不改变结果）。"""
    print(f"  [hook] after_model：当前消息数 {len(state['messages'])}")
    return None

class TimingHook(AgentMiddleware):

    def before_model(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:
        import time
        self._start = time.time()   # 记录开始时间
        print("  [hook class] before_model：开始计时")
        return None

    def after_model(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:
        import time
        end = time.time()           # 记录结束时间
        print(f"  [hook class] after_model：结束计时 {end - self._start:.2f} 秒")
        return None


if __name__ == '__main__':
    # 用 MockRequest 模拟一次工具调用请求，观察包裹式 hook 如何拦截
    class MockToolRequest:
        tool_call = "get_weather(城市=北京)"


    class MockHandler:
        def __call__(self, request):
            return "晴天，25℃"


    result = log_tool_call.wrap_tool_call(MockToolRequest(), MockHandler())

    # 直接调用装饰器生成的 hook，模拟框架在模型调用前后触发
    state = {"messages": ["user: 北京天气怎么样？"]}
    log_before_model_call.before_model(state, runtime=None)
    log_after_model_call.after_model(state, runtime=None)

    timing = TimingHook()
    timing.before_model(state, runtime=None)
    timing.after_model(state, runtime=None)