"""
hook 函数（02d）
============================================================
学习目标：
1. 理解 LangChain 1.x 的「hook」是什么：生命周期钩子，在 agent 运行的关键节点插入自定义逻辑
2. 掌握三种真实 hook 形态：
   - @wrap_model_call / @wrap_tool_call（包裹式，签名 (request, handler)）
   - @before_model / @after_model（节点式，签名 (state, runtime)）
   - 继承 AgentMiddleware（类式，重写 before_model / after_model）
3. 区分 hook 与 Middleware（02c）：hook 是「轻量、单点」的拦截，Middleware 是「完整、可组合」的组件

先记住一句话：
    hook         -> 单点拦截：只想在某一个时刻（调用前/后）做一件事
    Middleware   -> 完整组件：一组 hook + 状态 + 配置，可以组合、复用

实际上 02c 里的内置中间件，底层就是「hook 的集合」。
本文件专门讲最轻量的 hook 写法——用一个装饰器，几行代码就完成一次拦截。

运行：
    python 02d_hook.py
"""

from langchain.agents.middleware import (
    AgentMiddleware,
    after_model,
    before_model,
    wrap_model_call,
    wrap_tool_call,
)


# ============================================================
# 1. 形态一：包裹式 hook —— @wrap_model_call / @wrap_tool_call
# ============================================================
# 被装饰函数签名固定为 (request, handler)：
#   - request ：这次调用的入参（含 state + runtime 上下文）
#   - handler ：原始调用，handler(request) 才会真正执行模型 / 工具
# 在 handler 前后插入逻辑，就实现了「拦截」。

@wrap_tool_call
def log_tool_call(request, handler):
    """工具调用前后各打一条日志（不改变结果）。"""
    print(f"  [hook] 工具调用前：{request.tool_call}")
    result = handler(request)          # 真正执行工具
    print(f"  [hook] 工具调用后：{result}")
    return result


@wrap_model_call
def log_model_call(request, handler):
    """模型调用前后各打一条日志。"""
    print("  [hook] 模型调用前")
    result = handler(request)          # 真正执行模型
    print("  [hook] 模型调用后")
    return result


# 也可以带参数使用：@wrap_tool_call(name="xxx") 给 hook 命名
@wrap_tool_call(name="retry_tool")
def retry_on_error(request, handler):
    """工具调用失败自动重试（改变行为！）。"""
    for attempt in range(2):
        try:
            return handler(request)
        except Exception:
            if attempt == 1:
                raise
    return None


# ============================================================
# 2. 形态二：节点式 hook —— @before_model / @after_model
# ============================================================
# 被装饰函数签名固定为 (state, runtime)，返回 None 表示「不修改状态」。

@before_model
def log_before_model(state, runtime):
    """模型调用前触发：打印当前消息数（不修改 state）。"""
    print(f"  [hook] before_model：当前消息数 {len(state['messages'])}")
    return None


@after_model
def log_after_model(state, runtime):
    """模型调用后触发：打印返回消息数。"""
    print(f"  [hook] after_model：返回消息数 {len(state['messages'])}")
    return None


# ============================================================
# 3. 形态三：类式 hook —— 继承 AgentMiddleware
# ============================================================
# 当一个横切关注点需要多个 hook 协同（既要 before 又要 after），
# 用类更清晰，还能持有实例属性（累积状态）。

class TimingHook(AgentMiddleware):
    """类形态：统计模型调用耗时（before_model + after_model 配合）。"""

    def before_model(self, state, runtime):
        import time
        self._start = time.time()   # 记录开始时间
        print("  [hook class] before_model：开始计时")
        return None

    def after_model(self, state, runtime):
        import time
        elapsed = time.time() - getattr(self, "_start", time.time())
        print(f"  [hook class] after_model：耗时 {elapsed:.3f}s")
        return None


# ============================================================
# 4. 验证：三种 hook 形态的触发
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("验证点①：包裹式 @wrap_tool_call（签名 request, handler）")
    print("=" * 60)

    # 用 MockRequest 模拟一次工具调用请求，观察包裹式 hook 如何拦截
    class MockToolRequest:
        tool_call = "get_weather(城市=北京)"

    class MockHandler:
        def __call__(self, request):
            return "晴天，25℃"

    result = log_tool_call.wrap_tool_call(MockToolRequest(), MockHandler())
    print(f"  最终返回：{result}")
    print()

    print("=" * 60)
    print("验证点②：节点式 @before_model / @after_model（签名 state, runtime）")
    print("=" * 60)
    # 直接调用装饰器生成的 hook，模拟框架在模型调用前后触发
    state = {"messages": ["user: 北京天气怎么样？"]}
    log_before_model.before_model(state, runtime=None)
    log_after_model.after_model(state, runtime=None)
    print()

    print("=" * 60)
    print("验证点③：类式 AgentMiddleware（before + after 协同、带状态）")
    print("=" * 60)
    timing = TimingHook()
    timing.before_model(state, runtime=None)
    timing.after_model(state, runtime=None)
    print()

    print("=" * 60)
    print("结论：hook 的三种形态 & 与 Middleware 的关系")
    print("=" * 60)
    print("""
    ┌────────────────┬──────────────────────────────────────────────┐
    │ 形态            │ 签名 / 适用场景                                │
    ├────────────────┼──────────────────────────────────────────────┤
    │ @wrap_*        │ (request, handler) 包裹模型/工具调用，可重试   │
    │ @before/@after │ (state, runtime) 单点日志/监控，不改状态        │
    │ 类 AgentMw      │ 重写 before/after，多 hook 协同、累积状态       │
    └────────────────┴──────────────────────────────────────────────┘

    三者的关系：
        hook 是「最小拦截单元」，Middleware 是「hook 的容器」。
        02c 里的内置中间件，底层就是把这些 hook 组装起来的。

    挂载方式（真实使用，与 02c 一致）：
        agent = create_agent(model, tools, middleware=[
            log_tool_call, log_before_model, TimingHook(),
        ])
        把 hook/中间件放进 middleware 列表，框架就会在对应节点自动触发。

    与 02c 的分工回顾：
        02c 讲「内置中间件的横切能力」（摘要/限流/重试）
        02d 讲「自己写拦截逻辑的最小单元」（hook）
    """)
