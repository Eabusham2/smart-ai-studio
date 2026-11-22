"""Best Work through the installed Pro/router/verifier and local text decoders."""
from types import SimpleNamespace
from functools import wraps
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
from tests.unit.test_best_work_tournament import request_kind, previous_draft


def _gguf_engine(model, entropy):
    backend = GGUFReasoningBackend.__new__(GGUFReasoningBackend)
    backend.model = model
    backend.is_gguf_available = True
    backend.n_ctx = model.n_ctx()
    backend.calculate_token_entropy = entropy
    settings = Settings(use_mock=False, auto_download=False, backend='gguf', max_new_tokens=1, best_work_passes=2)
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
            kind = request_kind(prompt)
            index = len([row for row in invocations if row['stage'] == stage and row['kind'] == 'original'])
            invocations.append({'stage':stage, 'temperature':kwargs['temperature'],
                                'max_tokens':kwargs['max_tokens'], 'prompt':prompt, 'kind':kind})
            def stream():
                try:
                    if kind == 'compare':
                        yield {'choices':[{'text':'{"winner":"B"}'}]}
                    elif stage == 0:
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
    assert len(invocations) == first_n + next_n + (first_n - 1) + (next_n - 1)
    assert len(closed) == len(invocations)
    assert [d['total'] for d in decisions if d['kind']=='route'] == [first_n,next_n]
    for stage,count in [(0,first_n),(1,next_n)]:
        temperatures=[row['temperature'] for row in invocations if row['stage']==stage and row['kind']=='original']
        assert temperatures == pro_ladder(count-1 if count>1 else 1)
    assert all(row['max_tokens'] > 1536 for row in invocations)
    assert settings.max_new_tokens == 1 # Compatibility value restored, not enforced as a cap.
    assert not engine._best_work_stop_at_tools


def test_real_pro_refines_all_branches_before_tournament_with_full_routing_and_usage():
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
            kind = request_kind(prompt)
            index = sum(row['kind'] == 'original' for row in decoded)
            decoded.append({**kwargs, 'kind':kind})
            if kind == 'original':
                assert winner not in prompt and loser not in prompt
                text, tokens = (loser if index == 0 else winner), 3
            elif kind == 'refine':
                draft = previous_draft(prompt)
                assert draft in (winner, loser)
                assert "Which gallery?" in prompt and "Use complete sentences." in prompt
                text, tokens = ("The north gallery needs inspection." if draft == winner
                                else "The south gallery needs inspection."), 5
            else:
                text, tokens = '{"winner":"B"}', 1
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
        response = "".join(stream_best_work(ctrl, "Which gallery?", history, status_callback=decisions.append, passes=2))
    assert response == "The north gallery needs inspection."
    assert len(entropies) == 1
    assert [call["temperature"] for call in decoded if call['kind']=='original'] == pro_ladder(8)
    assert [call["temperature"] for call in decoded if call['kind']=='refine'] == pro_ladder(8)
    assert all(call['temperature'] == .20 for call in decoded if call['kind']=='compare')
    assert [event["total"] for event in decisions if event["kind"] == "route"] == [9]
    assert any(e["kind"] == "selected" and e["winning_branch"] == 8 for e in decisions)
    meta = ctrl.last_best_work_metadata
    assert meta['pass_summaries'][0]['routed_branches'] == 9
    assert [(p["invocations"], p["tokens"]) for p in meta["pass_summaries"]] == [(9, 27)] + [(1, 5)] * 9
    assert meta["raw_branches"][meta["winning_branch"]] == response
    assert meta["pass_usage"] == {"tokens": 89, "invocations": 35}
    assert request.snapshot()["tokens"] == 89 and request.snapshot()["invocations"] == 35


@pytest.mark.parametrize("branches", [1, 9])
def test_full_window_selected_draft_survives_unfit_refinement(branches, monkeypatch):
    from core.best_work import BEST_WORK_GUIDE
    from core.context_memory import ContextOverflow
    question = "Keep the final constraint: report in metres."
    history = [{"role": "system", "content": "Preserve the complete question."}]
    selected = "X" * 380
    lost = "unselected candidate" + "Y" * (len(selected) - len("unselected candidate"))
    decoded = []
    packed_inputs = []
    class Model:
        def tokenize(self, text, **kwargs):
            return list(text)
        def n_ctx(self):
            return 4096
        def __call__(self, prompt, **kwargs):
            decoded.append(kwargs)
            assert kwargs["max_tokens"] == len(selected)
            text = lost if branches > 1 and len(decoded) == 1 else selected
            yield {"choices": [{"text": text}], "usage": {"completion_tokens": len(text)}}

    engine = _gguf_engine(Model(), lambda _: 0.1 if branches == 1 else 0.5)
    format_prompt = engine._format_prompt_with_history
    def pack(prompt, turns):
        packed = format_prompt(prompt, turns)
        packed_inputs.append(packed)
        return packed
    monkeypatch.setattr(engine, "_format_prompt_with_history", pack)
    packed = pack(question, [{"role": "system", "content": BEST_WORK_GUIDE}, *history])
    engine._context_budget_tokens = len(packed.encode()) + len(selected)
    ctrl = SimpleNamespace(app=SimpleNamespace(engine=engine))
    events = []
    if branches == 1:
        assert "".join(stream_best_work(ctrl, question, history, status_callback=events.append, passes=2)) == selected
    else:
        # A full pair must fit before it can be judged; never silently restore
        # the former consensus/first-candidate fallback when comparison cannot fit.
        with pytest.raises(ContextOverflow):
            list(stream_best_work(ctrl, question, history, status_callback=events.append, passes=2))
    assert len(decoded) == branches
    assert [call["temperature"] for call in decoded] == pro_ladder(8 if branches > 1 else 1)
    assert question in packed_inputs[-1] and selected in packed_inputs[-1]
    assert (lost in packed_inputs[-1]) is (branches > 1)
    assert history[0]["content"] in packed_inputs[-1]
    skipped = [event for event in events if event["kind"] == "refinement_skipped"]
    assert len(skipped) == branches
    assert all(event['stage'] == 1 and event['pass_index'] == 2 and event['passes'] == 2
               and not event['final'] and event['reason'] == 'context_limit'
               and event['selected_pass'] == 1 for event in skipped)
    meta = ctrl.last_best_work_metadata
    assert meta["passes_requested"] == 2 and meta["passes_completed"] == 1
    assert meta["total_passes_completed"] == branches
    assert [row["completed"] for row in meta["pass_summaries"]] == [True] + [False] * branches
    assert all(row['skip_reason'] == 'context_limit' for row in meta['pass_summaries'][1:])
    tokens = len(selected) if branches == 1 else len(selected) * 8 + len(lost)
    assert meta["pass_usage"] == {"invocations": branches, "tokens": tokens}
    assert history == [{"role": "system", "content": "Preserve the complete question."}]
    assert not engine._best_work_stop_at_tools


@pytest.fixture
def cpu_decoder(tiny_controller, monkeypatch, tmp_path):
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
        samples = []
        generate = backend.model.generate
        def record(**kwargs):
            input_ids = kwargs["input_ids"][0].tolist()
            result = generate(**kwargs)
            samples.append((input_ids, result[0, len(input_ids):].tolist()))
            return result
        monkeypatch.setattr(backend.model, "generate", record)
        # This random tiny vocabulary cannot emit the structured A/B protocol.
        # Keep both actual neural draft/revision generations and script only the
        # selector; selection orchestration has separate real-wrapper tests.
        generate_branches = backend.generate_branches
        @wraps(generate_branches)
        def select_or_decode(prompt, *args, **kwargs):
            if 'BEST_WORK_COMPARISON' in prompt:
                from core.generation_metrics import TokenMeter
                meter = TokenMeter(backend, 'scripted_selector')
                meter.observe(count=1)
                meter.finish()
                return ['{"winner":"B"}']
            return generate_branches(prompt, *args, **kwargs)
        monkeypatch.setattr(backend, 'generate_branches', select_or_decode)
        yield engine, samples
    finally:
        engine.unload_model()


@pytest.mark.parametrize("tiny_controller", [512], indirect=True)
def test_real_cpu_transformers_decoder_consumes_previous_sampled_answer(cpu_decoder):
    import torch

    # Every visible vocabulary entry is terminal so the model stops naturally
    # after one word per pass, independent of the specific answer it samples.
    engine, samples = cpu_decoder
    backend = engine.controller_backend
    # Include actual adapter tensors, created before inference, so immutability
    # is checked for learned parameters as well as the base model's weights.
    backend._ensure_transformers_lora()
    backend.model.eval()
    model_before = backend.model
    weights_before = {name: tensor.detach().clone()
                      for name, tensor in backend.model.state_dict().items()}
    assert any('lora_' in name for name in weights_before)
    assert any('lora_' not in name for name in weights_before)
    terminal_ids = sorted(set(backend.tokenizer.get_vocab().values()) - {0, 1, 2})
    backend.model.generation_config.eos_token_id = terminal_ids
    backend.model.generation_config.suppress_tokens = sorted(set(range(backend.model.config.vocab_size))
                                                            - set(terminal_ids))
    ctrl = SimpleNamespace(app=SimpleNamespace(engine=engine))
    question = "What is the vault color?"
    text = "".join(stream_best_work(ctrl, question, passes=2))
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
    assert ctrl.last_best_work_metadata["pass_usage"] == {"invocations": 3, "tokens": 3}
    assert backend.model is model_before
    weights_after = backend.model.state_dict()
    assert weights_after.keys() == weights_before.keys()
    assert all(torch.equal(weights_after[name], before) for name, before in weights_before.items())


@pytest.mark.parametrize("tiny_controller", [512], indirect=True)
def test_real_cpu_full_physical_window_returns_entire_selected_sample(cpu_decoder):
    from core.context_memory import ContextOverflow

    engine, samples = cpu_decoder
    backend = engine.controller_backend
    capacity = backend.model.config.n_positions
    engine._context_budget_tokens = capacity
    # Sample real words through the entire physical window, including the last
    # position. Exclude special tokens so early EOS cannot mask this boundary.
    words = {idx for token, idx in backend.tokenizer.get_vocab().items() if token.isalpha()}
    backend.model.generation_config.suppress_tokens = sorted(set(range(backend.model.config.vocab_size)) - words)
    ctrl = SimpleNamespace(app=SimpleNamespace(engine=engine))
    question = "What is the vault color?"
    text = "".join(stream_best_work(ctrl, question, passes=2))
    assert len(samples) == 1
    input_ids, output_ids = samples[0]
    assert len(input_ids) + len(output_ids) == capacity
    assert text == backend.tokenizer.decode(output_ids, skip_special_tokens=True)
    meta = ctrl.last_best_work_metadata
    assert meta["pass_usage"] == {"invocations": 1, "tokens": len(output_ids)}
    assert meta["passes_requested"] == 2 and meta["passes_completed"] == 1
    assert meta["pass_summaries"][1]["skip_reason"] == "context_limit"
    # A new request without a completed candidate retains the real error.
    with pytest.raises(ContextOverflow):
        list(stream_best_work(ctrl, "vault " * capacity, passes=2))
    assert len(samples) == 1
