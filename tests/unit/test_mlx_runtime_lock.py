import threading
import time

from core.mlx_runtime_lock import install_mlx_runtime_lock


class DummyMLX:
    def __init__(self):
        self.generation_entered = threading.Event()
        self.release_generation = threading.Event()
        self.training_entered = threading.Event()

    def calculate_token_entropy(self, prompt):
        return 0.5

    def generate_branches(self, prompt, branch_count=1):
        self.generation_entered.set()
        self.release_generation.wait(timeout=2.0)
        return ["ok"] * branch_count

    def stream_generate_tokens(self, prompt):
        self.generation_entered.set()
        self.release_generation.wait(timeout=2.0)
        yield "ok"

    def train_mini_batch(self, *args, **kwargs):
        self.training_entered.set()
        return {}, 0.1


install_mlx_runtime_lock(DummyMLX)


def test_background_training_waits_for_branch_generation_boundary():
    engine = DummyMLX()

    generation = threading.Thread(target=lambda: engine.generate_branches("x", 1))
    generation.start()
    assert engine.generation_entered.wait(timeout=1.0)

    training = threading.Thread(target=lambda: engine.train_mini_batch())
    training.start()
    time.sleep(0.05)
    assert not engine.training_entered.is_set()

    engine.release_generation.set()
    generation.join(timeout=1.0)
    training.join(timeout=1.0)
    assert engine.training_entered.is_set()


def test_background_training_waits_for_stream_completion():
    engine = DummyMLX()

    def consume():
        list(engine.stream_generate_tokens("x"))

    generation = threading.Thread(target=consume)
    generation.start()
    assert engine.generation_entered.wait(timeout=1.0)

    training = threading.Thread(target=lambda: engine.train_mini_batch())
    training.start()
    time.sleep(0.05)
    assert not engine.training_entered.is_set()

    engine.release_generation.set()
    generation.join(timeout=1.0)
    training.join(timeout=1.0)
    assert engine.training_entered.is_set()
