"""Stop/steer interrupts the current Pro candidate, including unlimited output."""
from types import SimpleNamespace
import json
import threading
import pytest
from config.settings import Settings
from core.pro_engine import ProReasoningEngine
from core.engines.gguf_engine import GGUFReasoningBackend
from core.engines.bitnet_cpp_engine import BitNetCppReasoningBackend


def make_engine(backend, family='gguf'):
    engine = ProReasoningEngine(settings=Settings(use_mock=False, auto_download=False, backend=family))
    engine.mlx_backend = engine.gguf_backend = engine.bitnet_backend = engine.controller_backend = None
    setattr(engine, family+'_backend', backend)
    engine.active_backend = family
    engine.awake_consolidator = None
    engine._context_budget_tokens = 8192
    engine._best_work_stop_at_tools = True
    return engine


def test_stop_closes_active_gguf_candidate_before_next_token_and_anchor():
    cancelled = threading.Event()
    calls, closed = [], []
    class Model:
        def tokenize(self, text, **kwargs): return [1]*10
        def n_ctx(self): return 8192
        def __call__(self, prompt, **kwargs):
            calls.append(kwargs)
            assert kwargs['stream'] is True
            def events():
                try:
                    yield {'choices':[{'text':'Partial output '}]}
                    pytest.fail('Cancelled Pro candidate consumed another token')
                finally: closed.append(True)
            return events()
    backend = GGUFReasoningBackend.__new__(GGUFReasoningBackend)
    backend.model = Model(); backend.is_gguf_available = True; backend.n_ctx = 8192
    backend.calculate_token_entropy = lambda prompt:0.5
    engine = make_engine(backend)
    def progress(*args, **kwargs):
        if kwargs.get('finished') is False: cancelled.set()
    answer, meta = engine.solve('Analyze this request', cancel_event=cancelled, progress_callback=progress)
    assert answer == '' and meta.get('cancelled') is True
    assert len(calls) == 1 and closed == [True]


@pytest.mark.parametrize('family', ['controller','mlx','gguf','bitnet'])
def test_all_pro_backend_families_receive_the_live_cancel_callback(family):
    cancelled = threading.Event()
    calls = []
    class Backend:
        model = object()
        is_loaded = is_gguf_available = is_mlx_available = True
        tokenizer = SimpleNamespace(encode=lambda prompt:[1]*10)
        runtime = 'test-backend'
        def calculate_token_entropy(self, prompt): return 0.5
        def generate_branches(self, prompt, branch_count, max_tokens, temperature, top_p,
                              cancelled=None, progress_callback=None, stop_at_tool_boundary=False):
            assert callable(cancelled) and not cancelled()
            calls.append(branch_count)
            cancellation_event.set()
            assert cancelled()
            return []
    cancellation_event = cancelled
    engine = make_engine(Backend(),family)
    answer, meta = engine.solve('Analyze this request', cancel_event=cancelled)
    assert answer == '' and meta.get('cancelled') is True
    assert calls == [8]  # No extra anchor, backend fallback, or second request.


def test_cancellation_support_does_not_enable_unrequested_bitnet_tool_stopping(monkeypatch):
    call='<tool_call>{"name":"read_file","arguments":{}}</tool_call>'
    class Response:
        closed = False
        def __iter__(self):
            yield ('data: '+json.dumps({'content':call,'tokens_predicted':1})+'\n').encode()
            yield ('data: '+json.dumps({'content':' ordinary continuation','tokens_predicted':2,'stop':True})+'\n').encode()
        def close(self): self.closed=True
    response=Response()
    monkeypatch.setattr('urllib.request.urlopen',lambda *a,**k:response)
    backend=BitNetCppReasoningBackend.__new__(BitNetCppReasoningBackend)
    backend._base_url='http://127.0.0.1:1'
    result=backend.generate_branches('prompt',branch_count=1,cancelled=lambda:False,stop_at_tool_boundary=False)
    assert result==[call+' ordinary continuation'] and response.closed
