"""
中间件 Middleware（02c）
============================================================
学习目标：
1. 理解 LangChain 1.x 的「中间件」是什么：拦截请求链路、改写行为的横切机制
2. 区分 Middleware 与 Callback（02g）的本质差异
3. 认识三类内置中间件：摘要 / 限流 / 重试
4. 手写一个最小中间件，看清 before / after / wrap 三个切入点的触发时机

先记住一句话：
    Callback    -> 「观察」：在旁边记录日志、统计耗时，不改变执行结果
    Middleware  -> 「拦截」：可以改写输入、拦截输出、决定是否重试，改变执行结果

对照 mini-agent/agent.py：那些散落的逻辑（算 token、限制轮数、出错重试）
原本都写在循环体里，中间件把它们抽成独立的、可插拔的横切组件。

运行：
    uv run python 02c_middleware.py
"""

from typing import Any

from langchain.agents.middleware import (
    AgentMiddleware,
    ModelCallLimitMiddleware,
    ModelRetryMiddleware,
    SummarizationMiddleware,
    ToolCallLimitMiddleware,
    ToolRetryMiddleware,
)


# ============================================================
# 1. 横切关注点一：限流（Limit）—— 给「调用次数」设上限
# ============================================================

# ModelCallLimitMiddleware：限制整个会话里 LLM 被调用的次数
#   thread_limit -> 一个会话（thread）内的上限
#   run_limit    -> 一次 run 内的上限
#   exit_behavior-> 达到上限后是 'end'（正常结束）还是 'error'（抛错）
model_limit = ModelCallLimitMiddleware(
    thread_limit=10,   # 单会话最多调用 10 次模型
    exit_behavior="error",  # 超限后抛错，而不是静默结束
)

# ToolCallLimitMiddleware：限制某个工具的调用次数
tool_limit = ToolCallLimitMiddleware(
    tool_name="get_weather",  # 只针对这个工具
    thread_limit=3,           # 单会话最多调用 3 次
    exit_behavior="continue", # 超限后跳过该工具，继续跑（不抛错）
)


# ============================================================
# 2. 横切关注点二：重试（Retry）—— 失败自动重试
# ============================================================

# ModelRetryMiddleware：模型调用失败（如网络抖动、限流）时自动重试
#   max_retries   -> 最多重试几次
#   backoff_factor-> 指数退避系数（延迟 1s -> 2s -> 4s）
#   initial_delay -> 首次重试前的等待秒数
model_retry = ModelRetryMiddleware(
    max_retries=2,
    backoff_factor=2.0,
    initial_delay=1.0,
)

# ToolRetryMiddleware：某个工具执行失败时自动重试（可按工具名过滤）
tool_retry = ToolRetryMiddleware(
    max_retries=2,
    tools=["calculate"],  # 只重试 calculate 这个工具
)


# ============================================================
# 3. 横切关注点三：摘要（Summarization）—— 上下文快爆时自动压缩
# ============================================================

# SummarizationMiddleware：当上下文 token 达到阈值时，把历史压缩成摘要
#   注意：它需要传入一个 model 来生成摘要（真实环境要连 LLM）。
#   这里只展示「如何声明」，不实际触发（否则会调用模型）。
#   声明后即可挂到 agent 上，token 超限时自动把旧消息替换成摘要。
# summarizer = SummarizationMiddleware(
#     model="gpt-4o-mini",              # 用哪个模型做摘要
#     trigger=("tokens", 8000),         # 上下文达到 8000 token 时触发
#     keep=("messages", 20),            # 摘要后保留最近 20 条消息
# )


# ============================================================
# 4. 手写最小中间件：看清 before / after / wrap 三个切入点
# ============================================================

class LoggingMiddleware(AgentMiddleware):
    """自定义中间件：演示横切钩子的三个切入点。

    对应 AgentMiddleware 提供的能力（覆盖即可生效）：
    - before_model      : 模型调用前触发，可改写 request
    - after_model       : 模型调用后触发，可改写 response
    - before_agent      : agent 启动前触发
    - after_agent       : agent 结束后触发
    - wrap_model_call   : 包裹模型调用（装饰器，可 try/except 包住整次调用）
    - wrap_tool_call    : 包裹工具调用（装饰器，可拦截/改写工具的入参出参）

    下面演示最常用的 wrap_model_call + before_model，其余钩子同理。
    """

    # 切入点①：before —— 在模型调用前打印，可改写 request
    def before_model(self, state, runtime):
        print(f"[middleware before_model] 即将调用模型，当前状态 keys={list(state.keys())}")
        # 返回 None 表示不修改；也可以返回一个新的 state 来改写输入
        return None

    # 切入点②：wrap —— 把整次模型调用包起来，能做 before/after/try-except
    def wrap_model_call(self, request, handler):
        print(f"[middleware wrap_model_call] 包裹模型调用，request={str(request)[:40]}...")
        # handler(request) 是「原始调用」，包一层即可前后插入逻辑
        response = handler(request)
        print(f"[middleware wrap_model_call] 模型调用完成，response={str(response)[:40]}...")
        return response


# ============================================================
# 5. 验证：中间件 vs Callback 的本质差异
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("验证点①：内置中间件的声明（不触发真实模型调用）")
    print("=" * 60)
    print(f"model_limit.name  = {model_limit.name}")
    print(f"tool_limit.name   = {tool_limit.name}")
    print(f"model_retry.name  = {model_retry.name}")
    print(f"tool_retry.name   = {tool_retry.name}")
    print()
    print("这三个中间件分别对应：限流 / 限流 / 重试 / 重试")
    print("（SummarizationMiddleware 需要真实 model，此处仅注释演示）")
    print()

    # 验证点②：手写中间件的钩子机制
    print("=" * 60)
    print("验证点②：手写中间件的 before / wrap 钩子如何工作")
    print("=" * 60)
    mw = LoggingMiddleware()

    # 手动触发 before_model（真实场景由 agent 框架自动调用）
    mw.before_model({"messages": []}, runtime=None)

    # 手动触发 wrap_model_call，模拟一次模型调用被包裹
    def fake_handler(request):
        return "模拟的模型响应"

    result = mw.wrap_model_call("模拟请求", fake_handler)
    print(f"wrap_model_call 最终返回：{result}")
    print()

    print("=" * 60)
    print("结论：Middleware 与 Callback（02g）的分工")
    print("=" * 60)
    print("""
    ┌──────────────┬────────────────────────┬──────────────────────────┐
    │              │  Callback（02g）        │  Middleware（02c）        │
    ├──────────────┼────────────────────────┼──────────────────────────┤
    │ 定位          │ 观察（旁路记录）        │ 拦截（改写执行链路）      │
    │ 能否改输入     │ 否                     │ 能（before/wrap 改写）    │
    │ 能否改输出     │ 否                     │ 能（after/wrap 改写）     │
    │ 能否重试       │ 否                     │ 能（RetryMiddleware）     │
    │ 能否限流       │ 否                     │ 能（LimitMiddleware）     │
    │ 典型用途       │ 日志、统计、调试 print  │ 摘要、限流、重试、脱敏    │
    │ 心智模型       │ 「监听器」             │ 「拦截器」               │
    └──────────────┴────────────────────────┴──────────────────────────┘

    一句话：Callback 是「看」，Middleware 是「管」。
    想记录发生了什么 -> Callback；想改变接下来会发生什么 -> Middleware。
    """)
