"""Represent full agent traces as conditioning data and verify their final completions."""
import json


def candidate_trajectories(items):
    if not isinstance(items, list) or len(items) < 2:
        raise ValueError("Provide at least two candidate trajectories")
    traces = []
    for index, item in enumerate(items):
        if isinstance(item, str):
            item = {"completion": item}
        if not isinstance(item, dict):
            raise ValueError("Each candidate must contain a completion or a message trajectory")
        agent = item.get("agent", f"Agent {index + 1}")
        if not isinstance(agent, str) or not agent.strip():
            raise ValueError("Each candidate needs a nonempty agent label")
        messages = item.get("messages")
        if messages is None:
            messages = [{"role": "assistant", "content": item.get("completion")}]
        if (not isinstance(messages, list) or not messages or
                any(not isinstance(m, dict) or m.get("role") not in {"system", "user", "assistant", "tool"}
                    or not isinstance(m.get("content"), str) or not m["content"].strip() for m in messages)
                or messages[-1]["role"] != "assistant"):
            raise ValueError("A trajectory requires text messages ending in the agent's answer")
        traces.append({"agent": agent, "messages": [{"role": m["role"], "content": m["content"]} for m in messages]})
    return traces


def joint_rows(prompt, traces, rewards):
    """Supervise winning completions alone and in the context of peer traces."""
    best = max(rewards)
    rows = []
    seen = set()

    def add(cue, completion):
        key = cue, completion
        if key not in seen:
            rows.append({"prompt": cue, "completion": completion})
            seen.add(key)

    for i, (trace, reward) in enumerate(zip(traces, rewards)):
        if reward != best:
            continue
        answer = trace["messages"][-1]["content"]
        add(prompt, answer)
        peers = []
        for j, other in enumerate(traces):
            # Hide the target completion when the same trace occurs more than once.
            messages = other["messages"]
            if i == j or messages[-1]["content"] == answer:
                messages = messages[:-1]
            if messages:
                peers.append({"agent": other["agent"], "messages": messages})
        if peers:
            cue = f"{prompt}\n\nCandidate reasoning records:\n{json.dumps(peers, ensure_ascii=False)}"
            add(cue + "\nUse the reasoning above to solve the task.", answer)
    return rows
