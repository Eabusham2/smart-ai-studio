# Using persistent learning in Studio

Studio can carry information from an earlier conversation into a later one by saving changes to a model's adapter, a small set of trainable parameters. The chat archive keeps the original wording for reading and export. The model still has a finite **Context** window for the current instructions, active conversation and answer.

## Start a project memory

1. Load the text model you want to use. Open **Memory DB** in the toolbar, then **Learning & saved memory…**. Select **Check model support** and click **Run learning task** to see which operations the loaded model and device support.
2. In **Memory**, select **project**, enter a key such as `field-guide`, and click **Apply memory**. Return to the same model and key to resume this project's transcript and learned state. The default **model** scope shares learned state across chats for that model. Choose **session** and a key to keep a particular conversation separate.
3. For a one-off learning experiment, select **Temporary practice** on a supported task. Studio forks the current checkpoint and restores the previous state when practice finishes. **Clear Chat** clears the selected transcript and preserves its learned adapter.

## What happens as a chat grows

On a trainable backend, Studio starts learning from older, completed exchanges when the packed input reaches 80% of Context. It tests whether the updated model can reproduce the selected text exactly and answer a grounded question without seeing the source. Only a saved checkpoint with a successful recall check lets Studio omit that text from later prompts. A parameter update can pass its training checks while the original exchange stays active.

After successful consolidation, Studio works toward 60% of Context. At the hard limit, if the oldest batch fails validation, Studio can keep it in text and try later eligible exchanges until the request fits. System instructions, tool exchanges and unfinished turns still need space. If the remaining input fills the window, Studio reports the limit. Successful checks establish what the model recalled at the time of validation. Later updates can affect recall. [Chat consolidation and retention](parametric-memory.md)

For a manual update, open **Sleep EWC**. **Current chat exchanges** can study a batch before the 80% threshold. **Verified episodes (original Sleep/EWC)** uses the existing queue of verified solutions, including when the current chat is empty. **Learn** and **RSI** also remain available for studying a verified source and learning from a self-generated revision. [Sleep, Learn and RSI](parametric-memory.md#what-happens-to-a-chat)

## Save a document and ask about it

Suppose you want the model to remember a project's naming conventions. In the learning panel for `field-guide`, select **Save a document memory**. Click **Choose source…**, pick the local conventions file, then enter a stable `key` such as `naming` and a descriptive `title` in the request editor. Click **Run learning task**. If the result is `paused`, run the same request on the same model and scope to continue. A completed study with an accepted update registers a specific document checkpoint.

Select **List saved memories** to check the entry. Then select **Answer using saved memory**, enter a question about a convention and run it. Studio selects candidate document checkpoints, asks each for relevant facts, restores the original model and gives the accepted recalled answers to the answering model. Ordinary chat can also use the selected scope's saved documents automatically. A relevant document or detail can be missed. The original file remains the reference for exact wording.

Choose **Read a document** to train the selected scope directly in resumable chunks. **Study relevant evidence** uses your question to select verified quotations. The existing **Learn** attachment preview uses up to 16,000 characters for its lesson. Use document ingestion to process a complete source file or directory. [Document requests, CLI examples and memory catalog](test-time-learning.md)

## Other ways to learn

| Task in the learning panel | What to supply and what Studio does |
| --- | --- |
| **Adapt to a question** | Supply context and a query. On compatible architectures Studio holds the context's attention keys and values fixed while updating query-side parameters. The context must fit. |
| **Practice examples** | Supply independent input/output demonstrations and a different unanswered query. Studio adapts on held-out-example tasks and votes across predictions. |
| **Learn from a checked answer** or **Learn from agreement** | Supply a prompt and a verifier, or choose strict-majority agreement. Informative reward differences guide updates. Agreement alone cannot establish correctness. |
| **Discover a better solution** | Supply a task and verifier. Studio revisits earlier candidates, samples revisions and learns from relative rewards. **Consolidate candidate solutions** can also study accepted individual or peer-conditioned solutions. |

The same requests work through `ProReasoningEngine.run_online_learning()` and `tools/online_learning.py`. The detailed guide includes runnable request JSON and backend-specific options. [Online learning examples](test-time-learning.md) · [Research coverage](ttt-research-coverage.md)

## Refine a response with Best Work

Enable **Pro branches** to use Best Work. By default each stage requests two sequential passes. The second sees the request, active history and first selected draft. To change the count, open **Branches**, enter **Best Work passes**, then click **Apply passes**. One pass selects a response without a refinement. Ordinary chat retains its streaming path.

If a later refinement cannot fit before it emits text, Studio returns the last completed draft in full and identifies the skipped pass. An oversized first pass still reports a capacity error. Only the stage's final selection can dispatch a tool, subject to authorization against the original request. An actual tool result starts a fresh stage. [Best Work passes and tool handoffs](best-work-handoffs.md)

## Which models can learn

Compatible MLX and Transformers models need a supported adapter layout and training dependencies. GGUF and supported Prism routes need a verified local parent, conversion, native adapter loading and candidate scoring. BitNet needs a verified parent and a working rebuild and scoring path. **Strata is inference-only** and serves chat within the loaded native context.

On Apple Silicon, you can check a local Bonsai 2 snapshot from the repository root with the Studio Python environment active. Substitute the path to your snapshot for `/models/bonsai2`.

```bash
python tools/online_learning.py --backend mlx --target bonsai2 --model-path /models/bonsai2 --capabilities
```

Native MLX CPU and Metal CI passed the small packed Bonsai 2 and CRACK layout tests, including frozen weights, checkpoint restoration and rollback. Full 27B checkpoint recall, throughput and peak memory still require separate measurements. [Backend support and full-model evaluation](parametric-memory.md#measure-recall-on-apple-silicon) · [Validation results](validation/test-time-learning.md)
