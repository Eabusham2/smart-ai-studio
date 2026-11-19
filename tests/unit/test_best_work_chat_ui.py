"""Real Tk rendering and steer handoff checks; synthetic text only."""
import ast
from pathlib import Path
from types import SimpleNamespace
import threading
import tkinter as tk
import pytest

from app_gui import SmartAIChatbotApp
from config.settings import Settings


def _cancel_ui_callbacks(root):
    # Each test owns a separate interpreter. Do not leave invalid after scripts
    # for a later test's Tk event loop to encounter after this root is destroyed.
    for pending in root.tk.splitlist(root.tk.call('after', 'info')):
        # Cancel scheduling only. The callback command may belong to a child;
        # that widget's destroy() must remove it from its own command registry.
        root.tk.call('after', 'cancel', pending)


@pytest.mark.parametrize('starts_with_pro', [False, True])
def test_selected_handoff_is_painted_once_and_all_thinking_sections_render(tmp_path, monkeypatch, starts_with_pro):
    monkeypatch.setenv('SMARTAI_DATA_DIR', str(tmp_path))
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip('A desktop display is required for this real Tk regression test')
    root.withdraw()
    app = SmartAIChatbotApp(root, settings=Settings(use_mock=True, auto_download=False,
                                                 database_path=str(tmp_path / 'memory.db')))
    app._check_and_run_next_queue = lambda: None
    app.is_model_loaded = True
    app._set_generating_state(True)
    observed = []
    failures = []
    done = threading.Event()
    def stream(prompt, **kwargs):
        status, progress = kwargs['status_callback'], kwargs['progress_callback']
        status({'kind':'route','total':9 if starts_with_pro else 1,'branch_count':8 if starts_with_pro else 1})
        if starts_with_pro:
            for n in range(1,10): progress(n,9,5,0.1)
            status({'kind':'selected','total':9,'winning_branch':2})
        yield '<think>First reasoning section.</think>Chosen text before invocation.\n'
        yield 'Calling read_file({"path":"selected.txt"})\n'
        status({'kind':'executing','name':'read_file'})
        # This executes on the generation worker; the real Tk event loop must
        # have painted both preceding yields before the handoff callback returns.
        snapshot_ready = threading.Event()
        def snapshot():
            try:
                observed.append(app.chat_stream.get('1.0','end'))
            finally:
                snapshot_ready.set()
        app._event_queue.put(('ui_callback', snapshot))
        assert snapshot_ready.wait(2), 'The Tk owner did not capture the painted tool handoff'
        yield 'Returned tool data.\n'
        status({'kind':'routing','stage':2})
        status({'kind':'route','total':1 if starts_with_pro else 9,'branch_count':1 if starts_with_pro else 8})
        if not starts_with_pro:
            for n in range(1,10): progress(n,9,7,0.1)
            status({'kind':'selected','total':9,'winning_branch':1})
        yield '<think>Second reasoning section.</think>Finished response.'
    app.multimodal.stream_solve = stream
    def run():
        try:
            app._process_message_thread('Please inspect this example', 'Please inspect this example', None, None, True)
        except BaseException as exc:
            failures.append(exc)
        finally:
            done.set()
    worker = threading.Thread(target=run, daemon=True)
    def poll():
        if done.is_set(): root.after(30, root.quit)
        else: root.after(20, poll)
    root.after(10, worker.start)
    root.after(20, poll)
    root.after(15000, root.quit)
    try:
        root.mainloop()
        assert done.is_set() and not failures
        assert len(observed) == 1
        assert 'Chosen text before invocation.' in observed[0]
        assert 'Calling read_file' in observed[0]
        assert 'Returned tool data.' not in observed[0]
        text = app.chat_stream.get('1.0','end')
        for part in ('Chosen text before invocation.', 'Calling read_file', 'Returned tool data.', 'Finished response.'):
            assert text.count(part) == 1
        assert '<think>' not in text and '</think>' not in text
        assert list(app.thinking_cache.values())[-2:] == ['First reasoning section.', 'Second reasoning section.']
        labels = [child.cget('text') for panel in app.chat_stream.winfo_children()
                  for child in panel.winfo_children() if isinstance(child, tk.Label)]
        assert any('Best Work 9/9' in label and 'complete' in label for label in labels)
    finally:
        app.cancel_event.set()
        if app.watchdog:
            app.watchdog.stop_monitoring()
        app.engine.unload_model()
        worker.join(timeout=2)
        _cancel_ui_callbacks(root)
        root.destroy()


def test_steer_waits_for_the_owned_worker_before_clearing_cancel():
    source = (Path(__file__).resolve().parents[2] / 'app_gui.py').read_text(encoding='utf-8')
    tree = ast.parse(source)
    method = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                  and node.name == '_execute_steered_generation')
    scheduled, starts, states = [], [], []
    class Thread:
        def __init__(self, target, args, daemon): self.args = args
        def start(self): starts.append(self.args)
    namespace = {'threading': SimpleNamespace(Thread=Thread)}
    exec(compile(ast.Module(body=[method], type_ignores=[]), 'actual_gui_method', 'exec'), namespace)
    alive = [True]
    cancel = threading.Event();cancel.set()
    app = SimpleNamespace(_pending_steer='new direction',
                          _generation_worker=SimpleNamespace(is_alive=lambda:alive[0]),
                          root=SimpleNamespace(after=lambda ms, callback:scheduled.append(callback)),
                          pro_branches_var=SimpleNamespace(get=lambda:True),
                          _process_message_thread=lambda *args:None)
    def set_state(value):
        states.append(value)
        if value: cancel.clear()
    app._set_generating_state = set_state
    app._execute_steered_generation = lambda text:namespace['_execute_steered_generation'](app, text)
    app._execute_steered_generation('new direction')
    assert cancel.is_set() and not starts and not states and len(scheduled) == 1
    alive[0] = False
    scheduled.pop(0)()
    assert not cancel.is_set() and states == [True]
    assert len(starts) == 1 and starts[0][-2:] == (True, True)
    assert app._pending_steer is None
    app._execute_steered_generation('new direction')
    assert len(starts) == 1  # An old scheduled callback cannot launch it twice.


@pytest.fixture
def work_app(tmp_path, monkeypatch):
    monkeypatch.setenv('SMARTAI_DATA_DIR', str(tmp_path))
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip('A desktop display is required for this real Tk regression test')
    root.withdraw()
    app = SmartAIChatbotApp(root, settings=Settings(use_mock=True, auto_download=False,
                                                 database_path=str(tmp_path / 'work.db')))
    app._check_and_run_next_queue = lambda: None
    try:
        yield app
    finally:
        app.cancel_event.set()
        if app.watchdog:
            app.watchdog.stop_monitoring()
        app.engine.unload_model()
        _cancel_ui_callbacks(root)
        root.destroy()
        import gc
        gc.collect()


def _descendants(widget):
    for child in widget.winfo_children():
        yield child
        yield from _descendants(child)


def test_hybrid_and_pro_branches_are_on_by_default_with_two_passes(work_app):
    from core.online_consolidator import HybridAwakeOnlineConsolidator
    app = work_app
    assert app.settings.learning_implementation == "hybrid"
    assert isinstance(app.engine.awake_consolidator, HybridAwakeOnlineConsolidator)
    assert app.pro_branches_var.get() is True
    assert app.settings.best_work_passes == app.engine.settings.best_work_passes == 2
    app.chk_pro_branches.invoke()
    assert app.pro_branches_var.get() is False
    app.chk_pro_branches.invoke()
    assert app.pro_branches_var.get() is True
    assert app.settings.best_work_passes == 2


def test_branch_dialog_adds_pass_control_without_removing_candidate_inspector(work_app):
    from tkinter import ttk
    app = work_app
    app.last_metadata = {"winning_branch": 0, "verified": True,
        "branches": [{"index": 0, "temp": .2, "passed": True, "reward": 1.0,
                      "candidate": "Measured candidate text", "code": "print('verified')", "stderr": ""}]}
    app._on_open_branch_visualizer()
    modal = next(w for w in app.root.winfo_children() if isinstance(w, tk.Toplevel))
    widgets = list(_descendants(modal))
    control = next(w for w in widgets if isinstance(w, ttk.Spinbox))
    apply = next(w for w in widgets if isinstance(w, ttk.Button) and w.cget('text') == 'Apply passes')
    assert control.get() == '2'
    for value in ('3', '1'):
        control.delete(0, 'end')
        control.insert(0, value)
        apply.invoke()
        assert app.settings.best_work_passes == app.engine.settings.best_work_passes == int(value)
    control.delete(0, 'end')
    control.insert(0, '0')
    apply.invoke()
    assert app.settings.best_work_passes == 1
    assert any(isinstance(w, tk.Label) and 'positive whole number' in w.cget('text') for w in widgets)
    assert any(isinstance(w, ttk.Treeview) and w.get_children() for w in widgets)
    assert any(isinstance(w, tk.Text) and 'Measured candidate text' in w.get('1.0', 'end') for w in widgets)


def test_branch_dialog_keeps_explicit_single_pass_for_the_next_best_work_request(work_app):
    from tkinter import ttk
    app = work_app
    app._on_open_branch_visualizer()
    modal = next(w for w in app.root.winfo_children() if isinstance(w, tk.Toplevel))
    widgets = list(_descendants(modal))
    control = next(w for w in widgets if isinstance(w, ttk.Spinbox))
    apply = next(w for w in widgets if isinstance(w, ttk.Button) and w.cget('text') == 'Apply passes')
    assert control.get() == '2'
    control.delete(0, 'end')
    control.insert(0, '1')
    apply.invoke()
    observed = []

    def stream(_prompt, **kwargs):
        observed.append(kwargs)
        yield 'The explicitly selected single-pass response.'

    app.multimodal.stream_solve = stream
    app.is_model_loaded = True
    app._set_generating_state(True)
    app._process_message_thread('Explain this.', 'Explain this.', None, None, app.pro_branches_var.get())
    app.root.update()
    assert len(observed) == 1
    assert observed[0]['best_work_enabled'] is True
    assert observed[0]['best_work_passes'] == 1
    assert app.settings.best_work_passes == app.engine.settings.best_work_passes == 1
    assert app.pro_branches_var.get() is True
    assert 'The explicitly selected single-pass response.' in app.chat_stream.get('1.0', 'end')


def test_branch_dialog_without_a_completed_run_keeps_an_empty_candidate_inspector(work_app):
    from tkinter import ttk
    app = work_app
    app.last_metadata = None
    app._on_open_branch_visualizer()
    modal = next(w for w in app.root.winfo_children() if isinstance(w, tk.Toplevel))
    widgets = list(_descendants(modal))
    tree = next(w for w in widgets if isinstance(w, ttk.Treeview))
    assert not tree.get_children()
    assert any(isinstance(w, tk.Label) and 'No completed run' in w.cget('text') for w in widgets)
    assert any(isinstance(w, tk.Text) for w in widgets)


@pytest.mark.parametrize('capacity_fallback', [False, True])
def test_gui_runs_sequential_best_work_and_displays_only_final_selected_pass(work_app, capacity_fallback):
    from core.generation_metrics import TokenMeter
    app = work_app
    assert app.settings.best_work_passes == 2  # Exercise the default through the real dispatcher.
    app.is_model_loaded = True
    app.chat_history[app.active_tab_id] = []
    app._set_generating_state(True)
    calls = []
    failure = []
    done = threading.Event()

    def stream(prompt, history=None, cancel_event=None, progress_callback=None, decision_callback=None):
        calls.append((prompt, history))
        if capacity_fallback and len(calls) == 2:
            from core.context_memory import ContextOverflow
            raise ContextOverflow('Refinement cannot fit the complete draft')
        decision_callback({'kind': 'route', 'total': 1, 'branch_count': 1})
        meter = TokenMeter()
        try:
            meter.observe(delta=2 if len(calls) == 1 else 3)
            yield 'HIDDEN FIRST DRAFT' if len(calls) == 1 else '<think>Final review.</think>Published answer.'
        finally:
            meter.finish()

    app.engine.stream_solve = stream

    def run():
        try:
            app._process_message_thread('Explain the result.', 'Explain the result.', None, None, True)
        except BaseException as exc:
            failure.append(exc)
        finally:
            done.set()

    worker = threading.Thread(target=run, daemon=True)
    def poll():
        if done.is_set():
            app.root.after(50, app.root.quit)
        else:
            app.root.after(10, poll)
    app.root.after(10, worker.start)
    app.root.after(20, poll)
    app.root.after(8000, app.root.quit)
    app.root.mainloop()
    worker.join(timeout=2)
    assert done.is_set() and not failure
    assert len(calls) == 2 and 'HIDDEN FIRST DRAFT' in calls[1][0]
    assert 'Explain the result.' in calls[1][0]
    visible = app.chat_stream.get('1.0', 'end')
    labels = [w.cget('text') for w in _descendants(app.chat_stream) if isinstance(w, tk.Label)]
    if capacity_fallback:
        assert visible.count('HIDDEN FIRST DRAFT') == 1 and 'Published answer.' not in visible
        assert app.last_metadata['passes_completed'] == 1
        assert app.last_metadata['pass_usage'] == {'invocations': 1, 'tokens': 2}
        assert app._live_generation_snapshot['tokens'] == 2
        assert app._live_generation_snapshot['invocations'] == 1
        assert any('used pass 1/2' in label and 'refinement reached Context' in label for label in labels)
        assert not any('pass 2/2' in label and 'complete' in label for label in labels)
    else:
        assert 'HIDDEN FIRST DRAFT' not in visible and visible.count('Published answer.') == 1
        assert app.last_metadata['passes_completed'] == 2
        assert app._live_generation_snapshot['tokens'] == 5
        assert app._live_generation_snapshot['invocations'] == 2
        assert any('pass 2/2' in label and 'complete' in label for label in labels)
