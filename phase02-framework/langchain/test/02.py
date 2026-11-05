from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


class WeatherInput(BaseModel):
    city: str = Field(description="需要查询天气的城市名称")
    days: int =Field(default=1,ge=1,le=7,description="查询未来的天气 1-7")

def get_weather(city:str, days:int) -> str:
    """查询指定城市未来几天的天气。"""
    return f"正在查询 {city} 未来 {days} 天的天气"

weather_tool=StructuredTool.from_function(
    func=get_weather,
    name="get_weather",
    description="查询指定城市未来几天的天气。",
    args_schema=WeatherInput
)

if __name__ == '__main__':
    str=weather_tool.invoke({"city": "北京", "days": 3})
    print(str)
