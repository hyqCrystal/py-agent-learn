"""
LangChain Tools 基础学习

学习目标：
1. 使用 @tool 将 Python 函数转换为 Tool
2. 理解 Tool 的 name 和 description
3. 使用 invoke() 调用工具
"""

from langchain.tools import tool


@tool
def get_weather(city: str) -> str:
    """查询指定城市的天气情况。"""
    weather_data = {
        "北京": "晴天，25℃",
        "上海": "多云，22℃",
        "深圳": "小雨，28℃",
    }
    return weather_data.get(city, "未知城市")



if __name__ == "__main__":
    # 查看工具基本信息
    print("工具名称：", get_weather.name)
    print("工具描述：", get_weather.description)

    # 查看工具参数 Schema
    print("工具参数：", get_weather.args)

    # 手动调用工具
    result = get_weather.invoke({"city": "北京"})

    print("执行结果：", result)