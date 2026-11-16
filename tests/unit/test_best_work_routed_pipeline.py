"""Full installed Pro/router/verifier chain, synthetic GGUF streams, no models."""
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

    backend = GGUFReasoningBackend.__new__(GGUFReasoningBackend)
    backend.model = Model()
    backend.is_gguf_available = True
    backend.n_ctx = 32768
    def entropy(prompt):
        entropies.append(prompt)
        return after if 'ACTUAL selected file content' in prompt else before
    backend.calculate_token_entropy = entropy
    settings = Settings(use_mock=False, auto_download=False, backend='gguf', max_new_tokens=1)
    engine = ProReasoningEngine(settings=settings)
    engine.mlx_backend = engine.bitnet_backend = engine.controller_backend = None
    engine.gguf_backend = backend
    engine.active_backend = 'gguf'
    engine.awake_consolidator = None
    engine._context_budget_tokens = 32768
    engine.speculative_engine = SimpleNamespace(get_telemetry=lambda: {})

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
                                  status_callback=decisions.append,
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
