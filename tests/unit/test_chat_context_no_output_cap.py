"""No neural weights: output can use the entire remaining model context."""
from types import SimpleNamespace
import pytest
from core.awake_auto_hook import _remaining_generation_tokens, _active_trainable_backend, install_awake_auto_learning
from core.unified_context_budget import install


def test_direct_torch_uses_actual_model_window_and_tokenizer():
    engine = SimpleNamespace(model=SimpleNamespace(config=SimpleNamespace(max_position_embeddings=8192)),
                             tokenizer=SimpleNamespace(encode=lambda p: [1]*100),
                             active_backend='torch', _context_budget_tokens=32768)
    assert _active_trainable_backend(engine) is engine
    assert _remaining_generation_tokens(engine, 'prompt') == 8092


@pytest.mark.parametrize('legacy_cap', [0, 1, 512, 1536])
def test_single_stream_ignores_legacy_output_cap(legacy_cap):
    class Engine:
        settings = SimpleNamespace(max_new_tokens=legacy_cap)
        gguf_backend = None
        mlx_backend = None
        model = None
        _gui_force_single_stream = True
        _context_budget_tokens = 8192
        controller_backend = SimpleNamespace(runtime='generic', model=object(), tokenizer=SimpleNamespace(encode=lambda p: [1]*100))
        def _format_prompt_with_history(self, prompt, history): return prompt
        def stream_solve(self, prompt, **kwargs):
            yield str(self.settings.max_new_tokens)
        def solve(self, prompt, **kwargs): return str(self.settings.max_new_tokens), {}
    install_awake_auto_learning(Engine)
    engine=Engine()
    assert ''.join(engine.stream_solve('request')) == '8092'
    assert engine.solve('request')[0] == '8092'
    assert engine.settings.max_new_tokens == legacy_cap


def test_mlx_direct_default_does_not_reintroduce_512_token_cap():
    class Backend: pass
    install(Backend)
    backend=Backend()
    backend.model=SimpleNamespace(args=SimpleNamespace(max_seq_len=8192))
    backend.tokenizer=SimpleNamespace(encode=lambda p:[1]*100)
    assert backend._effective_generation_cap('prompt', 512)==8092


def test_full_context_is_not_truncated_to_make_room():
    engine=SimpleNamespace(model=SimpleNamespace(config=SimpleNamespace(max_position_embeddings=128)),
                           tokenizer=SimpleNamespace(encode=lambda p:[1]*128),
                           active_backend='torch', _context_budget_tokens=32768)
    with pytest.raises(RuntimeError, match='refusing truncation'):
        _remaining_generation_tokens(engine, 'full')
