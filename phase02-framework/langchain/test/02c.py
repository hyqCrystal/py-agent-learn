import os

from dotenv import find_dotenv, load_dotenv
from langchain.agents.middleware import SummarizationMiddleware, ModelCallLimitMiddleware, ToolRetryMiddleware
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

env_path=find_dotenv()
load_dotenv(env_path)

model=ChatOpenAI(
    model=os.getenv("OPENAI_MODEL"),
    api_key=os.getenv("openai_api_key"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)

@tool
def get_weather(city:str)->str:
    """查询指定城市的实时天气。"""
    data = {"北京": "晴天，25℃", "上海": "多云，22℃", "深圳": "小雨，28℃"}
    return data.get(city, f"未知城市：{city}")

@tool
def calculate(expression:str)->str:
    """计算一个数学表达式（仅支持 + - * / 和括号）。"""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算失败：{e}"

def main() -> None:
    summarizer = SummarizationMiddleware(
        model=model,  # 用哪个模型做摘要（复用上方 .env 读到的 model）
        trigger=("tokens", 8000),  # 上下文达到 8000 token 时触发
        keep=("messages", 20),  # 摘要后保留最近 20 条消息
    )

    limit = ModelCallLimitMiddleware(
        thread_limit=3,  # 单会话最多调用 3 次模型
        exit_behavior="error",  # 超限后抛错（另一种是 'end' 正常结束）
    )


    retry = ToolRetryMiddleware(
        max_retries=2,                  # 工具执行失败最多重试 2 次
        backoff_factor=2.0,             # 指数退避：1s -> 2s
        initial_delay=1.0,              # 首次重试前等待 1 秒
    )



if __name__ == '__main__':
    main()