"""
General Knowledge Anchor Dataset.
Used during sleep consolidation to compute Fisher Information Matrix
and prevent catastrophic forgetting of core language, coding, and mathematical reasoning skills.
"""

from typing import Dict, List

CORE_ANCHORS: List[Dict[str, str]] = [
    {
        "prompt": "Write a Python function to reverse a linked list.",
        "completion": "```python\ndef reverse_linked_list(head):\n    prev = None\n    curr = head\n    while curr:\n        next_node = curr.next\n        curr.next = prev\n        prev = curr\n        curr = next_node\n    return prev\n```"
    },
    {
        "prompt": "Explain binary search and implement it in Python.",
        "completion": "```python\ndef binary_search(arr, target):\n    low, high = 0, len(arr) - 1\n    while low <= high:\n        mid = (low + high) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            low = mid + 1\n        else:\n            high = mid - 1\n    return -1\n```"
    },
    {
        "prompt": "What is the quadratic formula for solving ax^2 + bx + c = 0?",
        "completion": "The quadratic formula is given by: x = (-b ± sqrt(b^2 - 4ac)) / (2a), where the term b^2 - 4ac is the discriminant."
    },
    {
        "prompt": "Implement a Python function to check if a string is a valid palindrome.",
        "completion": "```python\ndef is_palindrome(s: str) -> bool:\n    cleaned = ''.join(c.lower() for c in s if c.isalnum())\n    return cleaned == cleaned[::-1]\n```"
    },
    {
        "prompt": "Explain Bayes' Theorem and its mathematical formulation.",
        "completion": "Bayes' Theorem describes the probability of an event based on prior knowledge of conditions related to the event: P(A|B) = (P(B|A) * P(A)) / P(B)."
    },
    {
        "prompt": "Write a Python function to calculate the greatest common divisor (GCD) using the Euclidean algorithm.",
        "completion": "```python\ndef gcd(a: int, b: int) -> int:\n    while b != 0:\n        a, b = b, a % b\n    return abs(a)\n```"
    },
    {
        "prompt": "Implement depth-first search (DFS) for a graph represented as an adjacency list.",
        "completion": "```python\ndef dfs(graph, start, visited=None):\n    if visited is None:\n        visited = set()\n    visited.add(start)\n    for neighbor in graph.get(start, []):\n        if neighbor not in visited:\n            dfs(graph, neighbor, visited)\n    return visited\n```"
    },
    {
        "prompt": "Explain the time complexity of quicksort and merge sort.",
        "completion": "Merge sort has a guaranteed O(N log N) time complexity in worst, average, and best cases. Quicksort has an average time complexity of O(N log N) and a worst-case time complexity of O(N^2) when bad pivots are chosen."
    },
    {
        "prompt": "Write a Python function to compute the intersection of two sorted arrays.",
        "completion": "```python\ndef intersect_sorted_arrays(arr1, arr2):\n    i, j = 0, 0\n    res = []\n    while i < len(arr1) and j < len(arr2):\n        if arr1[i] == arr2[j]:\n            res.append(arr1[i])\n            i += 1\n            j += 1\n        elif arr1[i] < arr2[j]:\n            i += 1\n        else:\n            j += 1\n    return res\n```"
    },
    {
        "prompt": "What is the difference between supervised and unsupervised learning?",
        "completion": "Supervised learning trains models on labeled input-output pairs to predict target values. Unsupervised learning identifies underlying patterns, clusters, or representations in unlabeled data."
    }
]


def get_anchor_dataset() -> List[Dict[str, str]]:
    """Returns the full anchor dataset as prompt-completion dictionaries."""
    return CORE_ANCHORS.copy()


def get_anchor_texts() -> List[str]:
    """Formats anchor dataset into structured training dialogue strings."""
    return [
        f"<|im_start|>user\n{item['prompt']}<|im_end|>\n<|im_start|>assistant\n{item['completion']}<|im_end|>"
        for item in CORE_ANCHORS
    ]
