"""
长期记忆持久化（02f）
============================================================
学习目标：
1. 理解「短期记忆」与「长期记忆」的本质差异
2. 掌握 LangGraph 的两个持久化组件：
   - checkpointer（会话状态）：保存「当前这次对话」的进度，断点可恢复
   - Store（跨会话事实）：保存「跨对话」的长期事实，如用户偏好、身份信息
3. 对比 02e 手写 Memory：02e 是「进程内、不落盘」，02f 是「可落盘、可恢复」

先记住一句话（与 02e 的分工）：
    02e 短期记忆 -> 单次对话内的 Buffer/Window/Summary，进程一结束就丢
    02f 长期记忆 -> 用 checkpointer 保存会话状态 + 用 Store 保存跨会话事实

记忆的两条轴线（务必分清）：
    ┌──────────────┬────────────────────────────┬──────────────────────────┐
    │              │  会话状态（checkpointer）   │  跨会话事实（Store）       │
    ├──────────────┼────────────────────────────┼──────────────────────────┤
    │ 存什么        │ 当前对话进行到哪、消息历史  │ 用户是谁、偏好、长期约定  │
    │ 作用范围      │ 单个会话（thread）         │ 所有会话共享              │
    │ 生命周期      │ 会话结束即归档              │ 永久保存                  │
    │ 典型场景      │ 断点续传、人机交互暂停      │ 「记住我叫张三」「记住偏好」 │
    │ 类比          │ 一张便签纸                 │ 一个档案柜                │
    └──────────────┴────────────────────────────┴──────────────────────────┘

运行：
    uv run python 02f_long_term_memory.py
"""

from langgraph.checkpoint.base import Checkpoint
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore


# ============================================================
# 1. checkpointer：会话状态持久化（断点可恢复）
# ============================================================
# InMemorySaver 是「内存版」checkpointer（教学用，不落磁盘）。
# 真实项目会换成 SqliteSaver / PostgresSaver，把状态写进数据库。
# 这里的 InMemory 只是「跨代码位置可恢复」，但已能演示持久化思想。

def demo_checkpointer():
    """演示：会话状态如何保存、如何恢复。"""
    print("=" * 60)
    print("验证点①：checkpointer 保存 / 恢复会话状态")
    print("=" * 60)

    checkpointer = InMemorySaver()

    # 会话的唯一标识是 thread_id，一个 thread = 一次独立对话
    def make_config(thread_id: str, checkpoint_id: str):
        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": "",        # 命名空间（同一 thread 内可再分组）
                "checkpoint_id": checkpoint_id,
            }
        }

    # 会话 A：写入一个 checkpoint（模拟 agent 跑完一步后的状态）
    config_a = make_config("session-A", "ckpt-A1")
    cp_a = Checkpoint(
        v=1,
        id="ckpt-A1",
        ts="2026-01-01T00:00:00",
        channel_values={"messages": ["用户：帮我查天气"]},
        channel_versions={"messages": "1"},
        versions_seen={},
        updated_channels={"messages"},
    )
    checkpointer.put(
        config_a,
        checkpoint=cp_a,
        metadata={"source": "demo", "step": 1},
        new_versions={"messages": "1"},
    )

    # 会话 B：另一个独立会话，写入不同的状态
    config_b = make_config("session-B", "ckpt-B1")
    cp_b = Checkpoint(
        v=1,
        id="ckpt-B1",
        ts="2026-01-01T00:00:05",
        channel_values={"messages": ["用户：帮我订机票"]},
        channel_versions={"messages": "1"},
        versions_seen={},
        updated_channels={"messages"},
    )
    checkpointer.put(
        config_b,
        checkpoint=cp_b,
        metadata={"source": "demo", "step": 1},
        new_versions={"messages": "1"},
    )

    # 模拟「进程重启」：新建一个 checkpointer 实例，指向同一个「数据库」
    # （真实场景是重新连同一个 Postgres，这里用同一个实例演示）
    restored_a = checkpointer.get_tuple(config_a)
    restored_b = checkpointer.get_tuple(config_b)
    print("进程重启后，按 thread_id 恢复各自会话的状态：")
    print(f"  会话 A（session-A）消息 = {restored_a.checkpoint['channel_values']['messages']}")
    print(f"  会话 B（session-B）消息 = {restored_b.checkpoint['channel_values']['messages']}")
    print()
    print("=> 关键①：checkpointer 把「每个会话进行到哪了」都存了下来，")
    print("   进程重启后按 thread_id 即可恢复，而不是从零开始。")
    print("=> 关键②：不同 thread_id 的状态彼此隔离，互不干扰。")
    print()


# ============================================================
# 2. Store：跨会话事实（长期记忆，所有会话共享）
# ============================================================
# Store 用 namespace + key 组织数据：
#   namespace -> 分类（如 ("users", "张三") 表示「张三这个用户」）
#   key       -> 事实名（如 "preference"）
#   value     -> 事实内容（一个 dict）
# 相比 checkpointer 的「单会话」，Store 的数据是「全局」的，任何会话都能读到。

def demo_store():
    """演示：跨会话事实如何写入、如何检索。"""
    print("=" * 60)
    print("验证点②：Store 保存 / 检索跨会话事实")
    print("=" * 60)

    store = InMemoryStore()

    # 写入：记住用户的偏好（namespace 定位到「用户 张三」）
    store.put(
        ("users", "zhangsan"),
        "preference",
        {"语言": "中文", "城市": "北京", "关注": "AI 领域"},
    )
    store.put(
        ("users", "zhangsan"),
        "identity",
        {"姓名": "张三", "角色": "团队负责人"},
    )
    store.put(
        ("users", "lisi"),
        "preference",
        {"语言": "中文", "城市": "上海"},
    )

    # 检索：按 namespace + key 精确取
    pref = store.get(("users", "zhangsan"), "preference")
    print(f"张三的偏好：{pref.value}")
    print()

    # 检索：按 namespace 前缀搜索（列出张三的所有事实）
    items = store.search(("users", "zhangsan"))
    print("张三名下的所有事实：")
    for item in items:
        print(f"  key={item.key!r:14} value={item.value}")
    print()

    # 检索：只按 "users" 前缀，列出所有用户（跨会话共享的体现）
    all_users = store.search(("users",))
    print("所有用户的 key（体现「跨会话」——不同用户的记忆互不干扰但都存着）：")
    for item in all_users:
        print(f"  namespace={item.namespace}  key={item.key}")
    print()
    print("=> 关键：Store 存的是「用户是谁、偏好什么」这类长期事实，")
    print("   换一个新会话，agent 依然能读到「张三偏好中文、关注 AI」。")
    print()


# ============================================================
# 3. 对比总结
# ============================================================

if __name__ == "__main__":
    demo_checkpointer()
    demo_store()

    print("=" * 60)
    print("结论：02e vs 02f —— 短期记忆 vs 长期记忆")
    print("=" * 60)
    print("""
    ┌──────────────┬─────────────────────────┬────────────────────────────┐
    │              │  02e 短期记忆           │  02f 长期记忆               │
    ├──────────────┼─────────────────────────┼────────────────────────────┤
    │ 实现          │ 手写 Buffer/Window/Sum │ checkpointer + Store       │
    │ 是否落盘      │ 否（进程内 list）       │ 可落盘（Sqlite/Postgres）  │
    │ 会话间共享     │ 否                     │ 是                         │
    │ 断点可恢复     │ 否                     │ 是（checkpointer）         │
    │ 存长期事实     │ 否                     │ 是（Store）                │
    │ 教学价值      │ 理解三种裁剪策略        │ 理解「状态」与「事实」分离  │
    └──────────────┴─────────────────────────┴────────────────────────────┘

    工程实践：
        - 单次对话的上下文管理 -> 02e 的 Window/Summary 思路
        - 对话中途暂停/恢复   -> checkpointer
        - 记住用户是谁、偏好  -> Store（配合语义检索可做到「跨会话记忆」）

    本文件用 InMemorySaver / InMemoryStore 演示（不落磁盘、可独立运行）。
    真实项目替换为 SqliteSaver / PostgresSaver 即可把记忆持久化到数据库。
    """)
