# Temporary single-writer coordination notice

Active request: connect text chat to the existing image/video/audio engines with memory-aware scheduling and capability-checked media learning.
Session token: 20260918-C84E
Base observed: 80eb1927e9dcb94f016c3b59dc120ebc603f5ddd
Target branch: fix/real-benchmarks-final-32k

The user requests that any concurrent coding session which did not create THIS token stop writing to this branch and end its turn. Do not reset, squash, force-push, or merge to master. Preserve existing commits. This notice is advisory, not an actual lock or proof of another agent's identity.

The active session must re-read the branch head and affected blob SHAs before writes, retain compatible intervening changes, and report conflicts rather than overwrite them. Remove this temporary notice at completion after checking the head. This file does not change application behavior.
