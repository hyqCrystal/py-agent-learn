"""
LangChain Tools 基础（Day 1a）：@tool 装饰器
============================================================
学习目标：
1. 用 @tool 将普通 Python 函数一键转换为 Tool
2. 理解 Tool 的 name / description / args 三个核心属性
3. 用 invoke() 手动调用工具
4. 验证点①：docstring / 类型注解 / 默认值分别映射到 schema 的哪个字段

对照 mini-agent/agent.py 里那段手写 JSON Schema：
    tools = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询指定城市的天气",
            "parameters": { ... }   # 手写的一整段 JSON Schema
        }
    }]
本文件展示：@tool 如何自动生成这段 schema，省去手写。

运行：
    python 01_tool_decorator.py
"""

from langchain_core.tools import tool


# ============================================================
# 1. 用 @tool 转换函数
# ============================================================

@tool
def get_weather(city: str) -> str:
    """查询指定城市的天气情况。

    注意：这段 docstring 的第一行会自动成为 Tool 的 description，
    后续行会进入 description 的剩余部分（供 LLM 判断何时调用）。
    """
    weather_data = {
        "北京": "晴天，25℃",
        "上海": "多云，22℃",
        "深圳": "小雨，28℃",
    }
    return weather_data.get(city, "未知城市")


# 带类型注解 + 默认值的示例，观察 schema 如何自动推断
@tool
def get_weather_forecast(city: str, days: int = 1) -> str:
    """查询指定城市未来几天的天气预报。"""
    return f"正在查询 {city} 未来 {days} 天的天气预报"


# ============================================================
# 2. 验证：@tool 生成的 schema 与手写版是否等价
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("工具基本信息（对应手写 schema 的 name / description）")
    print("=" * 60)
    print("工具名称 name      ：", get_weather.name)
    print("工具描述 description：", get_weather.description)
    print("参数 Schema args   ：", get_weather.args)
    print()

    # 验证点①：docstring 映射到 description；类型注解映射到 parameters
    # 对比手写版：
    #   手写 name        = "get_weather"            <- get_weather.name
    #   手写 description = "查询指定城市的天气"        <- docstring 首行
    #   手写 parameters  = {"city": {"type": "string"}} <- 类型注解 city: str
    print("=" * 60)
    print("验证点①：类型注解 + 默认值如何映射到 schema")
    print("=" * 60)
    print("get_weather_forecast 参数 schema：")
    print(get_weather_forecast.args)
    # 观察：city 无默认值 -> required；days 有默认值 -> 非 required 且有 default
    print()

    # 验证：手动调用工具
    print("=" * 60)
    print("手动调用工具 invoke()")
    print("=" * 60)
    result = get_weather.invoke({"city": "北京"})
    print("执行结果：", result)
    print()

    # 验证点②（对照）：@tool 传错类型参数会发生什么？
    # 实测结论（langchain 1.x）：@tool 底层同样用 Pydantic 做参数解析，
    # 因此基础类型错误（如 str 传给 int）也会被拦截。
    print("=" * 60)
    print("验证点②：@tool 传错类型参数会发生什么？")
    print("=" * 60)
    try:
        # days 传了字符串 "abc"，Pydantic 无法解析为 int，被拦截
        r = get_weather_forecast.invoke({"city": "北京", "days": "abc"})
        print("结果（未报错）：", r)
    except Exception as e:
        print("抛出异常：", e)
        print("结论：@tool 也能拦截基础类型错误（底层同为 Pydantic 解析）")
    print()
    print("=> @tool 与 StructuredTool 的真正差异不在「有没有校验」，")
    print("   而在「校验规则能否自定义」，请看 02_structured_tool.py")
