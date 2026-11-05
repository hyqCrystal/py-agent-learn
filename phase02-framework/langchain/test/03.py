from typing import Dict, List
from unittest import result


class BaseMemory:

    def save_history(self,messages: List[Dict])->None:
        """
        Save the history of messages to the memory.
        """
        raise NotImplementedError
    def get_messages(self)->List[Dict]:
        """
        Get the history of messages from the memory.
        """
        raise NotImplementedError

    @property
    def name(self):
        return self.__class__.__name__

class BufferMemory(BaseMemory):

    def __init__(self):
        self._messages: List[Dict] = []

    def save_history(self,messages: List[Dict])->None:
        self._messages.extend(messages)

    def get_messages(self)->List[Dict]:
        return self._messages

class WindowMemory(BaseMemory):

    def __init__(self, k: int = 4):
        self._messages: List[Dict] = []
        self.k = k

    def save_history(self,messages: List[Dict])->None:
        self._messages.extend(messages)
        self._messages=self._messages[-self.k:]

    def get_messages(self)->List[Dict]:
        return self._messages

class SummaryMemory(BaseMemory):
    """
    SummaryMemory is a type of memory that summarizes the history of messages.
    """
    def __init__(self,k:int=4):
        self._messages: List[Dict] = []
        self._sumarry=""
        self.k=k

    @staticmethod
    def _summarize_history(messages: List[Dict])->str:
        parts = []
        for m in messages:
            content = str(m.get("content", ""))
            parts.append(f"{m['role']}: {content[:15]}")
        return " | ".join(parts)

    def save_history(self,messages: List[Dict])->None:
        self._messages.extend(messages)
        if(len(self._messages)>self.k):
            overflow = self._messages[:-self.k]
            self._sumarry +="摘要："+self._summarize_history(overflow)+"->"
            self._messages=self._messages[-self.k:]

    def get_messages(self)->List[Dict]:
        result: List[Dict] = []
        if self._sumarry:
            result.append({"role": "system", "content": f"历史摘要 {self._sumarry}"})
        result.extend(self._messages)
        return result

if __name__ == '__main__':
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

    memories = [BufferMemory(), WindowMemory(k=4), SummaryMemory(k=4)]


    for mem in memories:
        mem.save_history(turns)

    for mem in memories:
        print(f"\n【{mem.name}】")
        for m in mem.get_messages():
            content = str(m["content"])
            if len(content) > 40:
                content = content[:40] + "..."
            print(f"  [{m['role']}] {content}")
