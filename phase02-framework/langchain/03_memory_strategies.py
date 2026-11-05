"""
手写 Memory 三策略（Day 2a）
============================================================
学习目标：
1. 不调框架，手写实现三种记忆策略：
   - Buffer   ：全量保留所有历史消息
   - Window   ：滑动窗口，只保留最近 N 条
   - Summary  ：用摘要压缩历史，保留核心信息
2. 验证点：三种策略在「上下文爆炸」面前的表现差异

背景：mini-agent/agent.py 里的 messages 列表就是最原始的 Buffer——
    每轮都把完整历史塞回给 LLM，对话变长后 token 会无限膨胀。
本文件抽象出统一的 Memory 接口，对比三种策略如何控制上下文长度。

运行：
    python 03_memory_strategies.py
"""

from typing import List, Dict


# ============================================================
# 1. 统一接口：所有策略都实现 save_history / get_messages
# ============================================================

class BaseMemory:
    """记忆策略基类，定义统一接口。"""

    def save_history(self, messages: List[Dict]) -> None:
        raise NotImplementedError

    def get_messages(self) -> List[Dict]:
        raise NotImplementedError

    @property
    def name(self) -> str:
        return self.__class__.__name__


# ============================================================
# 2. Buffer：全量保留
# ============================================================

class BufferMemory(BaseMemory):
    """全量保留所有历史消息，不做任何裁剪。"""

    def __init__(self):
        self._messages: List[Dict] = []

    def save_history(self, messages: List[Dict]) -> None:
        self._messages.extend(messages)

    def get_messages(self) -> List[Dict]:
        return self._messages


# ============================================================
# 3. Window：滑动窗口，只保留最近 K 条
# ============================================================

class WindowMemory(BaseMemory):
    """滑动窗口：只保留最近 K 条消息，旧消息被丢弃。"""

    def __init__(self, k: int = 4):
        self._messages: List[Dict] = []
        self.k = k

    def save_history(self, messages: List[Dict]) -> None:
        self._messages.extend(messages)
        # 只保留最近 k 条
        self._messages = self._messages[-self.k:]

    def get_messages(self) -> List[Dict]:
        return self._messages


# ============================================================
# 4. Summary：摘要压缩
# ============================================================

class SummaryMemory(BaseMemory):
    """摘要压缩：始终只保留「摘要 + 最近 K 条」。

    实现思路：
    - 用极简规则模拟「LLM 摘要」（真实场景会调 LLM 生成摘要）
    - 历史超过窗口后，把旧消息压成一行摘要
    - 最终返回 [摘要, 最近若干条]，既省 token 又保留近期细节
    """

    def __init__(self, k: int = 4):
        self._messages: List[Dict] = []
        self._summary = ""  # 累积摘要
        self.k = k

    @staticmethod
    def _summarize(messages: List[Dict]) -> str:
        """模拟摘要：真实实现应调用 LLM，这里用规则拼接示意。"""
        # 提取每条消息的 role 和内容前 15 字
        parts = []
        for m in messages:
            content = str(m.get("content", ""))
            parts.append(f"{m['role']}: {content[:15]}")
        return " | ".join(parts)

    def save_history(self, messages: List[Dict]) -> None:
        self._messages.extend(messages)
        # 超出窗口的消息压入摘要
        if len(self._messages) > self.k:
            overflow = self._messages[:-self.k]
            self._summary += "摘要:" + self._summarize(overflow) + " → "
            self._messages = self._messages[-self.k:]

    def get_messages(self) -> List[Dict]:
        result: List[Dict] = []
        if self._summary:
            result.append({"role": "system", "content": f"历史摘要：{self._summary}"})
        result.extend(self._messages)
        return result


# ============================================================
# 5. 验证：三种策略在「上下文爆炸」面前的表现
# ============================================================

if __name__ == "__main__":
    # 模拟一轮完整对话（user/assistant 交替）
    turns = [
        {"role": "user", "content": "帮我查一下北京的天气"},
        {"role": "assistant", "content": "北京晴天，25℃"},
        {"role": "user", "content": "那上海呢"},
        {"role": "assistant", "content": "上海多云，22℃"},
        {"role": "user", "content": "深圳呢"},
        {"role": "assistant", "content": "深圳小雨，28℃"},
        {"role": "user", "content": "明天适合去哪个城市玩"},
        {"role": "assistant", "content": "建议去北京，天气最好"},
        {"role": "user", "content": "帮我订张去北京的机票"},
    ]

    # 三种策略各跑一遍，观察最终 get_messages 返回的条数
    memories = [BufferMemory(), WindowMemory(k=4), SummaryMemory(k=4)]

    for mem in memories:
        mem.save_history(turns)

    print("=" * 60)
    print("验证点：三种策略在「上下文爆炸」面前的表现差异")
    print("=" * 60)
    print(f"{'策略':<14} {'最终消息数':<10} 说明")
    print("-" * 60)
    for mem in memories:
        msgs = mem.get_messages()
        note = {
            "BufferMemory": "全量保留，随对话无限增长 → 上下文爆炸",
            "WindowMemory": "恒定 K 条，丢弃早期信息 → 可能丢失关键前提",
            "SummaryMemory": "摘要 + 最近 K 条 → 兼顾长度与信息保留",
        }[mem.name]
        print(f"{mem.name:<14} {len(msgs):<10} {note}")

    print()
    print("=" * 60)
    print("各策略实际返回内容")
    print("=" * 60)
    for mem in memories:
        print(f"\n【{mem.name}】")
        for m in mem.get_messages():
            content = str(m["content"])
            if len(content) > 40:
                content = content[:40] + "..."
            print(f"  [{m['role']}] {content}")

    print()
    print("=" * 60)
    print("结论")
    print("=" * 60)
    print("""
    - Buffer  ：最简单，token 随对话线性增长，长对话必然爆上下文
    - Window  ：恒定长度，但会「失忆」，早期约束/用户偏好会被遗忘
    - Summary ：用少量 token 保留全局信息 + 近期细节，是折中方案

    真实工程里，LangChain 的 ConversationBufferMemory /
    ConversationBufferWindowMemory / ConversationSummaryMemory
    就是这三种策略的封装，理解了手写版再看框架 API 会非常快。
    """)
