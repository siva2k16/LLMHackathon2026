"""The agent loop used for evaluation (same message format as AgentTune's rollout)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Callable

from .env import SafeQueryEnv
from .prompts import parse_tool_calls, system_message, tool_result_turn, user_prompt
from .tasks import Task
from .tools import TOOLS

# policy(messages, task) -> assistant text
Policy = Callable[[list[dict], Task], str]


@dataclass
class Episode:
    task_id: str
    messages: list[dict]
    calls: list[dict] = field(default_factory=list)  # {"name", "arguments", "observation"}
    final_answer: str | None = None
    parse_errors: list[str] = field(default_factory=list)
    n_turns: int = 0
    hit_turn_limit: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def initial_messages(env: SafeQueryEnv, task: Task) -> list[dict]:
    return [
        {"role": "system", "content": system_message(TOOLS)},
        {"role": "user", "content": user_prompt(env.schema_text(), task.question)},
    ]


def run_episode(env: SafeQueryEnv, task: Task, policy: Policy, max_turns: int = 4) -> Episode:
    messages = initial_messages(env, task)
    ep = Episode(task_id=task.id, messages=messages)
    for _ in range(max_turns):
        text = policy(messages, task)
        messages.append({"role": "assistant", "content": text})
        ep.n_turns += 1
        calls, err = parse_tool_calls(text)
        if err:
            ep.parse_errors.append(err)
        if not calls:  # plain text (or malformed call) ends the episode, like AgentTune
            ep.final_answer = text
            break
        results = []
        for c in calls:
            obs = env.call(c["name"], c["arguments"])
            ep.calls.append({**c, "observation": obs})
            results.append(tool_result_turn(c["name"], obs))
        messages.append({"role": "user", "content": "\n\n".join(results)})
    else:
        ep.hit_turn_limit = True
    return ep
