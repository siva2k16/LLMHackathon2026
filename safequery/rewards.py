"""GRPO reward for AgentTune's `create_agentic_trainer`.

In agentic mode AgentTune forwards each rollout's `conversations` (plus every dataset
column, here `task`) to the reward functions. We rebuild the tool calls from the
conversation and score them with the same verified scorer used for evaluation.
"""

from __future__ import annotations

import json

from .env import SafeQueryEnv
from .prompts import parse_tool_calls
from .scoring import naive_reward_from_score, reward_from_score, score_episode
from .tasks import Task


def _as_text(x) -> str:
    if isinstance(x, str):
        return x
    if isinstance(x, list):  # conversational completion: list of messages
        for m in reversed(x):
            if isinstance(m, dict) and m.get("role") == "assistant":
                return str(m.get("content") or "")
        return ""
    if isinstance(x, dict):
        return str(x.get("content") or "")
    return str(x)


def calls_from_conversation(conv: list[dict]) -> tuple[list[dict], str | None, list[str]]:
    """Extract (calls, final_answer, parse_errors) from an AgentTune/OpenAI-style conversation."""
    calls: list[dict] = []
    errors: list[str] = []
    final = None
    for m in conv or []:
        if not isinstance(m, dict) or m.get("role") != "assistant":
            continue
        structured = m.get("tool_calls")
        if structured:
            for tc in structured:
                fn = tc.get("function", tc) if isinstance(tc, dict) else {}
                args = fn.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        errors.append("arguments_not_json")
                        continue
                calls.append({"name": fn.get("name"), "arguments": args if isinstance(args, dict) else {}})
            continue
        content = str(m.get("content") or "")
        parsed, err = parse_tool_calls(content)
        if err:
            errors.append(err)
        if parsed:
            calls.extend(parsed)
        elif content.strip():
            final = content
    return calls, final, errors


def make_grpo_reward(env: SafeQueryEnv, naive: bool = False):
    """Build the reward function. `naive=True` gives the hackable ablation reward."""
    to_reward = naive_reward_from_score if naive else reward_from_score

    def safequery_reward(completions=None, prompts=None, task=None, conversations=None, **kwargs):
        n = len(conversations) if conversations is not None else len(completions)
        tasks = task if isinstance(task, list) else [task] * n
        convs = conversations if conversations is not None else [None] * n
        comps = completions if completions is not None else [None] * n
        rewards = []
        for comp, t, conv in zip(comps, tasks, convs):
            tk = Task.from_json(t)
            if conv:
                calls, final, errs = calls_from_conversation(conv)
            else:
                text = _as_text(comp)
                parsed, err = parse_tool_calls(text)
                calls, final, errs = parsed or [], (None if parsed else text), ([err] if err else [])
            rewards.append(to_reward(score_episode(env, tk, calls, final, errs)))
        return rewards

    safequery_reward.__name__ = "safequery_naive_reward" if naive else "safequery_reward"
    return safequery_reward
