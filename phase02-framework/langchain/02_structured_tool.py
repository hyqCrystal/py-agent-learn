"""
StructuredTool + Pydantic（Day 1b）
============================================================
学习目标：
1. 用 Pydantic BaseModel 定义工具参数 Schema
2. 用 StructuredTool.from_function() 创建工具
3. 理解参数校验机制：错误类型参数在哪一侧被拦截
4. 验证点③：@tool 与 StructuredTool 的取舍场景

核心区别（先记住这一句）：
    @tool        -> 参数 schema 靠「类型注解」自动推断，无运行时校验
    StructuredTool -> 参数 schema 靠「Pydantic」显式声明，调用前先校验

运行：
    python 02_structured_tool.py
"""

from pydantic import BaseModel, Field, ValidationError
from langchain_core.tools import StructuredTool


# ============================================================
# 1. 用 Pydantic 显式定义参数 Schema
# ============================================================

class WeatherInput(BaseModel):
    """查询天气所需的参数。"""

    city: str = Field(description="需要查询天气的城市名称")
    days: int = Field(default=1, ge=1, le=7, description="查询未来几天的天气（1-7）")


# ============================================================
# 2. 定义普通 Python 函数
# ============================================================

def get_weather(city: str, days: int = 1) -> str:
    """查询指定城市未来几天的天气。"""
    return f"正在查询 {city} 未来 {days} 天的天气"


# ============================================================
# 3. 用 StructuredTool 创建工具
# ============================================================

weather_tool = StructuredTool.from_function(
    func=get_weather,
    name="get_weather",
    description="查询指定城市的天气",
    args_schema=WeatherInput,
)


# ============================================================
# 4. 测试与验证
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("工具信息")
    print("=" * 60)
    print("工具名称 name       ：", weather_tool.name)
    print("工具描述 description ：", weather_tool.description)
    print("参数 Schema (JSON)  ：")
    print(weather_tool.args_schema.model_json_schema())
    print()

    # 验证点②：正常调用
    print("=" * 60)
    print("正常调用")
    print("=" * 60)
    result = weather_tool.invoke({"city": "北京", "days": 3})
    print("执行结果：", result)
    print()

    # 验证点②：错误类型参数在哪一侧被拦截？
    print("=" * 60)
    print("验证点②：错误参数被拦截在哪一侧？")
    print("=" * 60)
    try:
        result = weather_tool.invoke({"city": "北京", "days": "abc"})
        print("结果：", result)
    except ValidationError as e:
        # 这里能捕获到，说明是 Pydantic 在「工具调用前」就拦截了，
        # 而不是等到 get_weather 函数体内部才报错。
        print("被 Pydantic 拦截（调用函数前）！")
        print("校验错误：", e.errors())
    print()

    # 验证点②补充：超出范围也会被拦截（ge/le 约束）
    print("=" * 60)
    print("验证点②补充：范围约束 ge/le 拦截")
    print("=" * 60)
    try:
        result = weather_tool.invoke({"city": "北京", "days": 99})
        print("结果：", result)
    except ValidationError as e:
        print("被拦截，days=99 超出 1-7 范围")
        print("校验错误：", e.errors())
    print()

    # 验证点③：@tool 与 StructuredTool 的取舍场景总结
    print("=" * 60)
    print("验证点③：@tool  vs  StructuredTool 取舍")
    print("=" * 60)
    print("""
    ┌───────────────┬────────────────────────┬──────────────────────────┐
    │               │  @tool                 │  StructuredTool          │
    ├───────────────┼────────────────────────┼──────────────────────────┤
    │ Schema 来源    │ 类型注解自动推断        │ Pydantic 显式声明         │
    │ 基础类型校验    │ 有（底层同为 Pydantic） │ 有                       │
    │ 复杂约束        │ 无法声明（无 ge/le/枚举）│ 支持（Field 全能力）      │
    │ description 控制│ 自动取 docstring        │ 可自由指定               │
    │ 代码量          │ 少，一个装饰器          │ 多，需额外写 BaseModel    │
    │ 适用场景        │ 简单工具、快速原型      │ 参数复杂/需强约束/对外暴露 │
    └───────────────┴────────────────────────┴──────────────────────────┘

    核心区别一句话：
        @tool 的 schema 由「类型注解」自动生成，约束能力受限于注解；
        StructuredTool 的 schema 由「Pydantic」显式声明，可加 ge/le/枚举/校验器等全量约束。
    """)
