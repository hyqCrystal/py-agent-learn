"""
hook 函数（02d）
============================================================
学习目标：
1. 理解 LangChain 1.x 的「hook」是什么：生命周期钩子，在 agent 运行的关键节点插入自定义逻辑
2. 掌握三种 hook 形态：装饰器（@wrap_tool_call / @wrap_model_call）、类（继承 AgentMiddleware）、wrap 系列
3. 区分 hook 与 Middleware（02c）：hook 是「轻量、单点」的拦截，Middleware 是「完整、可组合」的组件

先记住一句话：
    hook         -> 单点拦截：只想在某一个时刻（调用前/后）做一件事
    Middleware   -> 完整组件：一组 hook + 状态 + 配置，可以组合、复用

实际上 02c 里的中间件，底层就是「hook 的集合」。
本文件专门讲最轻量的 hook 写法——用一个装饰器，几行代码就完成一次拦截。

运行：
    uv run python 02d_hook.py
"""

from typing import Any

from langchain.agents.middleware import (
    after_agent,
    after_model,
    before_agent,
    before_model,
    wrap_model_call,
    wrap_tool_call,
)


# ============================================================
# 1. 形态一：装饰器 —— 用 @wrap_tool_call 包裹工具调用
# ============================================================
# 一个 hook 函数签名固定为 (request, handler)：
#   - request ：这次调用的入参
#   - handler ：原始调用，handler(request) 才会真正执行
# 在 handler 前后插入逻辑，就实现了「拦截」。

@wrap_tool_call
def log_tool_call(request: Any, handler):
    """工具调用前后各打一条日志（不改变结果）。"""
    print(f"  [hook] 工具调用前：{request}")
    result = handler(request)   # 真正执行工具
    print(f"  [hook] 工具调用后：{result}")
    return result


@wrap_tool_call
def reject_expensive_tool(request: Any, handler):
    """工具调用前拦截：如果某个工具太贵，直接拒绝调用（改变结果！）。"""
    # request 里能拿到工具名，这里用 request 的类型信息演示拦截
    if getattr(request, "tool_name", None) == "expensive_api":
        print(f"  [hook] 拦截了 {request.tool_name}，直接返回拒绝信息")
        return "已拦截：该工具调用成本过高"
    return handler(request)


# 也可以带参数使用：@wrap_tool_call(name="xxx") 给 hook 命名
@wrap_tool_call(name="my_model_logger")
def log_model_call(request: Any, handler):
    print(f"  [hook] 模型调用前：{str(request)[:40]}...")
    result = handler(request)
    print(f"  [hook] 模型调用后：{str(result)[:40]}...")
    return result


# ============================================================
# 2. 形态二：类 —— 继承 AgentMiddleware 实现多个 hook
# ============================================================
# 当一个横切关注点需要多个 hook 协同（比如既要 before 又要 after），
# 用类更清晰。这是 02c 里 LoggingMiddleware 的完整版。

from langchain.agents.middleware import AgentMiddleware


class TimingHook(AgentMiddleware):
    """类形态：统计模型调用耗时（before + after 配合）。"""

    def before_model(self, state, runtime):
        import time
        self._start = time.time()   # 记录开始时间
        print(f"  [hook class] before_model：开始计时")
        return None

    def after_model(self, state, runtime):
        import time
        elapsed = time.time() - getattr(self, "_start", time.time())
        print(f"  [hook class] after_model：耗时 {elapsed:.3f}s")
        return None


# ============================================================
# 3. 形态三：wrap 系列 —— 直接作为函数调用
# ============================================================
# wrap_tool_call / wrap_model_call 既能当装饰器（形态一），
# 也能当普通函数手动触发（方便单独测试 hook 逻辑）。


# ============================================================
# 4. 验证：三种 hook 形态的触发
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("验证点①：装饰器形态 @wrap_tool_call")
    print("=" * 60)

    # 模拟一次工具调用被 hook 包裹
    def fake_tool_handler(request):
        return f"工具执行结果（处理 {request}）"

    # 直接调用 hook 的 wrap_tool_call 方法，模拟框架内部触发
    result = log_tool_call.wrap_tool_call("北京天气请求", fake_tool_handler)
    print(f"  最终返回：{result}")
    print()

    print("=" * 60)
    print("验证点②：装饰器形态 —— 拦截（改变结果）")
    print("=" * 60)
    # 模拟一个「昂贵工具」的调用请求
    class FakeRequest:
        tool_name = "expensive_api"
    result = reject_expensive_tool.wrap_tool_call(FakeRequest(), fake_tool_handler)
    print(f"  最终返回：{result}")
    print()

    print("=" * 60)
    print("验证点③：类形态 —— before + after 协同")
    print("=" * 60)
    timing = TimingHook()
    timing.before_model({"messages": []}, runtime=None)
    timing.after_model({"messages": []}, runtime=None)
    print()

    print("=" * 60)
    print("结论：hook 的三种形态 & 与 Middleware 的关系")
    print("=" * 60)
    print("""
    ┌──────────────┬────────────────────────────────────────────────┐
    │ 形态          │ 适用场景                                        │
    ├──────────────┼────────────────────────────────────────────────┤
    │ 装饰器 @wrap  │ 单点拦截、几行搞定（最常用）                     │
    │ 类 AgentMw    │ 多 hook 协同、带状态、要复用                     │
    │ wrap 函数     │ 单独测试 / 动态触发 hook 逻辑                    │
    └──────────────┴────────────────────────────────────────────────┘

    三者的关系：
        hook 是「最小拦截单元」，Middleware 是「hook 的容器」。
        @wrap_tool_call 装饰一个函数 -> 自动生成一个中间件 -> 挂到 agent 上生效。

    挂载方式（真实使用）：
        agent = create_agent(model, tools, middleware=[log_tool_call, ...])
        把 hook/中间件放进 middleware 列表，框架就会在对应节点自动触发。

    与 02c 的分工回顾：
        02c 讲「内置中间件的横切能力」（摘要/限流/重试）
        02d 讲「自己写拦截逻辑的最小单元」（hook）
    """)
