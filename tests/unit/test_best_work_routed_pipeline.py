"""Best Work through the installed Pro/router/verifier and local text decoders."""
from types import SimpleNamespace
import json
import threading
import pytest

from config.settings import Settings
from core.pro_engine import ProReasoningEngine
from core.engines.gguf_engine import GGUFReasoningBackend
from core.best_work import stream_best_work
from core.temperature_policy import pro_ladder
from core.generation_metrics import RequestMetrics, observe_generations
from tests.unit.test_native_context_learning import tiny_controller


def _gguf_engine(model, entropy):
    backend = GGUFReasoningBackend.__new__(GGUFReasoningBackend)
    backend.model = model
    backend.is_gguf_available = True
    backend.n_ctx = model.n_ctx()
    backend.calculate_token_entropy = entropy
    settings = Settings(use_mock=False, auto_download=False, backend='gguf', max_new_tokens=1)
    engine = ProReasoningEngine(settings=settings)
    engine.mlx_backend = engine.bitnet_backend = engine.controller_backend = None
    engine.gguf_backend = backend
    engine.active_backend = 'gguf'
    engine.awake_consolidator = None
    engine._context_budget_tokens = 32768
    engine.speculative_engine = SimpleNamespace(get_telemetry=lambda: {})

    return engine


@pytest.mark.parametrize('before,after,first_n,next_n', [(0.1, 0.5, 1, 9), (0.5, 0.1, 9, 1), (0.85, 0.5, 17, 9)])
def test_original_router_switches_only_after_actual_tool_output(before, after, first_n, next_n):
    events = []
    invocations = []
    entropies = []
    closed = []
    winner_call = '<tool_call>{"name":"read_file","arguments":{"path":"selected.txt"}}</tool_call>'
    losing_call = '<tool_call>{"name":"read_file","arguments":{"path":"losing.txt"}}</tool_call>'

    class Model:
        def tokenize(self, text, **kwargs):
            return list(text)
        def n_ctx(self):
            return 32768
        def __call__(self, prompt, **kwargs):
            assert kwargs['stream'] is True
            stage = 1 if 'ACTUAL selected file content' in prompt else 0
            index = len([row for row in invocations if row['stage'] == stage])
            invocations.append({'stage':stage, 'temperature':kwargs['temperature'],
                                'max_tokens':kwargs['max_tokens'], 'prompt':prompt})
            def stream():
                try:
                    if stage == 0:
                        yield {'choices':[{'text':'<think>Inspect the file.</think>Chosen plan. '}]}
                        yield {'choices':[{'text':losing_call if first_n > 1 and index == 0 else winner_call}]}
                        pytest.fail('Candidate decoder advanced beyond its complete tool invocation')
                    else:
                        yield {'choices':[{'text':'<think>Use the actual result.</think>Final answer.'}]}
                finally:
                    closed.append((stage,index))
            return stream()

    def entropy(prompt):
        entropies.append(prompt)
        return after if 'ACTUAL selected file content' in prompt else before
    engine = _gguf_engine(Model(), entropy)
    settings = engine.settings

    class Tools:
        def list_available_tools(self):
            return [{'name':'read_file'}]
        def execute_tool(self, name, arguments):
            events.append(('execute',name,arguments))
            assert arguments == {'path':'selected.txt'}
            shown=''.join(row[1] for row in events if row[0]=='display')
            assert 'Chosen plan.' in shown and 'Calling read_file' in shown
            return True, 'ACTUAL selected file content'
    controller = SimpleNamespace(app=SimpleNamespace(engine=engine,tools=Tools()))
    decisions=[]
    progress=[]
    text=[]
    for chunk in stream_best_work(controller,'Read the file and summarize it',cancel_event=threading.Event(),
                                  status_callback=decisions.append, passes=1,
                                  progress_callback=lambda *a, **k: progress.append((a,k))):
        text.append(chunk);events.append(('display',chunk))
    answer=''.join(text)
    assert answer.endswith('Final answer.')
    assert 'losing.txt' not in answer
    assert sum(row[0]=='execute' for row in events) == 1
    assert len(entropies) == 2  # No redundant entropy evaluation in solve().
    assert 'ACTUAL selected file content' in entropies[-1]
    assert len(invocations) == first_n + next_n
    assert len(closed) == len(invocations)
    assert [d['total'] for d in decisions if d['kind']=='route'] == [first_n,next_n]
    for stage,count in [(0,first_n),(1,next_n)]:
        temperatures=[row['temperature'] for row in invocations if row['stage']==stage]
        assert temperatures == pro_ladder(count-1 if count>1 else 1)
    assert all(row['max_tokens'] > 1536 for row in invocations)
    assert settings.max_new_tokens == 1 # Compatibility value restored, not enforced as a cap.
    assert not engine._best_work_stop_at_tools


def test_real_pro_winner_alone_enters_next_pass_with_full_routing_and_usage():
    decoded = []
    entropies = []
    winner = "Winner: inspect north gallery."
    loser = "Loser: inspect south gallery."
    class Model:
        def tokenize(self, text, **kwargs):
            return list(text)
        def n_ctx(self):
            return 32768
        def __call__(self, prompt, **kwargs):
            assert kwargs["stream"] is True
            index = len(decoded)
            decoded.append(kwargs)
            if index < 9:
                assert winner not in prompt and loser not in prompt
                text, tokens = (loser if index == 0 else winner), 3
            else:
                assert winner in prompt and loser not in prompt
                assert "Which gallery?" in prompt and "Use complete sentences." in prompt
                text, tokens = "The north gallery needs inspection.", 5
            yield {"choices": [{"text": text}], "usage": {"completion_tokens": tokens}}

    def entropy(prompt):
        entropies.append(prompt)
        return 0.1 if winner in prompt else 0.5

    engine = _gguf_engine(Model(), entropy)
    ctrl = SimpleNamespace(app=SimpleNamespace(engine=engine))
    decisions = []
    history = [{"role": "system", "content": "Use complete sentences."}]
    request = RequestMetrics()
    with observe_generations(request.update):
        response = "".join(stream_best_work(ctrl, "Which gallery?", history, status_callback=decisions.append))
    assert response == "The north gallery needs inspection."
    assert len(entropies) == 2 and winner in entropies[1] and loser not in entropies[1]
    assert [call["temperature"] for call in decoded] == pro_ladder(8) + pro_ladder(1)
    assert [(event["total"], event["pass_index"]) for event in decisions
            if event["kind"] == "route"] == [(9, 1), (1, 2)]
    assert any(e["kind"] == "selected" and e["pass_index"] == 1 for e in decisions)
    meta = ctrl.last_best_work_metadata
    assert [(p["routed_branches"], p["invocations"], p["tokens"]) for p in meta["pass_summaries"]] == [
        (9, 9, 27), (1, 1, 5)]
    assert meta["raw_branches"][meta["winning_branch"]] == winner
    assert meta["pass_usage"] == {"tokens": 32, "invocations": 10}
    assert request.snapshot()["tokens"] == 32 and request.snapshot()["invocations"] == 10


def test_refinement_at_physical_context_boundary_keeps_whole_request_and_refuses_truncation():
    from core.best_work import BEST_WORK_GUIDE
    question = "Keep the final constraint: report in metres."
    history = [{"role": "system", "content": "Preserve the complete question."}]
    selected = "X" * 380
    decoded = []
    entropies = []
    class Model:
        def tokenize(self, text, **kwargs):
            return list(text)
        def n_ctx(self):
            return 4096
        def __call__(self, prompt, **kwargs):
            decoded.append(prompt)
            assert kwargs["max_tokens"] >= len(selected)
            yield {"choices": [{"text": selected}]}

    def entropy(prompt):
        entropies.append(prompt)
        return 0.1

    engine = _gguf_engine(Model(), entropy)
    packed = engine._format_prompt_with_history(question, [{"role": "system", "content": BEST_WORK_GUIDE}, *history])
    engine._context_budget_tokens = len(packed.encode()) + 400
    ctrl = SimpleNamespace(app=SimpleNamespace(engine=engine))
    visible = []
    with pytest.raises(RuntimeError, match="refusing truncation"):
        for text in stream_best_work(ctrl, question, history):
            visible.append(text)
    assert not visible and len(decoded) == 1
    assert question in entropies[-1] and selected in entropies[-1] and history[0]["content"] in entropies[-1]
    assert [row["completed"] for row in ctrl.last_best_work_metadata["pass_summaries"]] == [True, False]
    assert history == [{"role": "system", "content": "Preserve the complete question."}]
    assert not engine._best_work_stop_at_tools


@pytest.mark.parametrize("tiny_controller", [512], indirect=True)
def test_real_cpu_transformers_decoder_consumes_previous_sampled_answer(tiny_controller, monkeypatch, tmp_path):
    # The tiny model samples one word per pass. Every visible vocabulary entry
    # is terminal so the test observes native EOS independent of sampled text.
    model_path = tiny_controller.model_path
    tiny_controller.unload_model()
    monkeypatch.setattr("core.transformers_placement.transformers_memory_limits",
                        lambda *a, **kw: {"cpu": 256 * 1024**2})
    settings = Settings(use_mock=False, auto_download=False, entropy_low_threshold=4,
                        entropy_high_threshold=5, database_path=str(tmp_path / "chat.db"))
    engine = ProReasoningEngine(settings=settings, device="cpu")
    try:
        assert engine.load_model("Tiny local decoder", model_path=model_path, backend="torch")["status"] == "loaded"
        backend = engine.controller_backend
        assert {p.device.type for p in backend.model.parameters()} == {"cpu"}
        terminal_ids = sorted(set(backend.tokenizer.get_vocab().values()) - {0, 1, 2})
        backend.model.generation_config.eos_token_id = terminal_ids
        backend.model.generation_config.suppress_tokens = sorted(set(range(backend.model.config.vocab_size))
                                                                - set(terminal_ids))
        samples = []
        generate = backend.model.generate
        def record(**kwargs):
            input_ids = kwargs["input_ids"][0].tolist()
            result = generate(**kwargs)
            samples.append((input_ids, result[0, len(input_ids):].tolist()))
            return result
        monkeypatch.setattr(backend.model, "generate", record)
        ctrl = SimpleNamespace(app=SimpleNamespace(engine=engine))
        question = "What is the vault color?"
        text = "".join(stream_best_work(ctrl, question))
        assert len(samples) == 2 and all(len(output) == 1 for _, output in samples)
        first_input, first_answer = samples[0]
        second_input, second_answer = samples[1]
        # In the review prompt, one quoted draft stands between the original
        # request and the assistant generation delimiter. Check its real ID.
        assert len(second_input) > len(first_input)
        draft = backend.tokenizer.decode(first_answer, skip_special_tokens=True)
        from core.best_work import BEST_WORK_GUIDE, _refinement_prompt
        packed = engine._format_prompt_with_history(_refinement_prompt(question, draft),
                                                    [{"role": "system", "content": BEST_WORK_GUIDE}])
        assert second_input == backend.tokenizer.encode(packed)
        assert text == backend.tokenizer.decode(second_answer, skip_special_tokens=True)
        assert ctrl.last_best_work_metadata["pass_usage"] == {"invocations": 2, "tokens": 2}
    finally:
        engine.unload_model()
