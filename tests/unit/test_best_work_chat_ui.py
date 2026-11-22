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


def _best_work_progress_harness():
    """Run the actual GUI callbacks without requiring a desktop interpreter."""
    source = (Path(__file__).resolve().parents[2] / 'app_gui.py').read_text(encoding='utf-8')
    method = next(node for node in ast.walk(ast.parse(source))
                  if isinstance(node, ast.FunctionDef) and node.name == '_process_message_thread')
    callbacks = [next(node for node in ast.walk(method)
                      if isinstance(node, ast.FunctionDef) and node.name == name)
                 for name in ('_pass_label', '_decision')]
    completion = next(node for node in ast.walk(method)
                      if isinstance(node, ast.If) and isinstance(node.test, ast.BoolOp)
                      and isinstance(node.test.values[0], ast.Name)
                      and node.test.values[0].id == 'pro_active')
    factory = ast.parse('def bind(self, best_work_ui, _owns_request, _paint_best_work, _post):\n'
                        '    pro_active = False\n'
                        '    def finish():\n'
                        '        pass\n'
                        '    return _decision, finish\n').body[0]
    factory.body[1].body = [completion]
    factory.body[1:1] = callbacks
    namespace = {'threading': threading}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[factory], type_ignores=[])),
                 'actual_gui_progress', 'exec'), namespace)
    state = {'done': 0, 'total': 0, 'pass_index': 1, 'passes': 1, 'refinement_skipped': False}
    labels = []
    app = SimpleNamespace(cancel_event=threading.Event(), _best_work_mode=False,
                          chat_stream=SimpleNamespace(update_idletasks=lambda: None))
    decision, finish = namespace['bind'](app, state, lambda: True, labels.append,
                                         lambda _delay, callback: callback())
    return decision, finish, state, labels


@pytest.mark.parametrize('branches, local_winner, skipped_loser', [
    (1, 'A', False), (1, 'B', False),
    (9, 'A', False), (9, 'B', False),
    (9, 'A', True), (9, 'B', True),
])
def test_best_work_selected_progress_uses_winning_branch_events(
        tmp_path, monkeypatch, branches, local_winner, skipped_loser):
    from core.awake_auto_hook import _remaining_generation_tokens
    from core.best_work import stream_best_work
    from core.context_memory import ContextOverflow
    from tests.unit.test_best_work_tournament import (
        branch_index, comparison, make_controller, previous_draft, request_kind,
    )

    originals = []
    def respond(_model, call):
        if call['kind'] == 'original':
            index = len(originals)
            originals.append(index)
            return f'<think>THOUGHT_{index}</think>BRANCH_{index}_ORIGINAL'
        if call['kind'] == 'refine':
            index = branch_index(previous_draft(call['prompt']))
            return f'<think>DISCARD_REVISION</think>BRANCH_{index}_REVISED'
        pair = comparison(call['prompt'])['candidates']
        same_branch = branch_index(pair[0]['text']) == branch_index(pair[1]['text'])
        return '{"winner":"' + (local_winner if same_branch else 'A') + '"}'

    ctrl, _model, entropies, _actions = make_controller(
        respond, entropy=0.1 if branches == 1 else 0.5,
        database_path=str(tmp_path / 'progress.db'))
    def capacity(engine, prompt):
        if (skipped_loser and request_kind(prompt) == 'refine'
                and branch_index(previous_draft(prompt)) == branches - 1):
            raise ContextOverflow('The last losing branch cannot fit another refinement')
        return _remaining_generation_tokens(engine, prompt)
    monkeypatch.setattr('core.awake_auto_hook._remaining_generation_tokens', capacity)
    decision, finish, state, labels = _best_work_progress_harness()
    result = ''.join(stream_best_work(ctrl, 'Select the answer', passes=2, status_callback=decision))
    finish()

    expected_pass = 1 if local_winner == 'A' else 2
    assert len(originals) == branches and len(entropies) == 1
    assert result == '<think>THOUGHT_0</think>BRANCH_0_' + ('ORIGINAL' if expected_pass == 1 else 'REVISED')
    assert state['done'] == state['total'] == branches
    assert state['pass_index'] == expected_pass and state['passes'] == 2
    assert state['refinement_skipped'] is False
    assert f'Best Work {branches}/{branches} · complete · pass {expected_pass}/2' in labels[-1]
    assert 'refinement reached Context' not in labels[-1]
    if skipped_loser:
        assert any('refinement reached Context' in label for label in labels[:-1])
        assert ctrl.last_best_work_metadata['best_work']['branches'][-1]['refinement_skipped'] == 'context_limit'


def _persist_gui_output(accumulated, history, *, use_pro_branches):
    """Execute the actual GUI text preparation, empty check, and persistence."""
    from core.gui_thinking import reasoning_segments, transcript_payload

    source = (Path(__file__).resolve().parents[2] / 'app_gui.py').read_text(encoding='utf-8')
    method = next(node for node in ast.walk(ast.parse(source))
                  if isinstance(node, ast.FunctionDef) and node.name == '_process_message_thread')
    assignment = next(node for node in ast.walk(method)
                      if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                      and node.targets[0].id == 'full_ans')
    block = next(value for node in ast.walk(method) for _name, value in ast.iter_fields(node)
                 if isinstance(value, list) and assignment in value)
    start = block.index(assignment)
    end = next(index for index in range(start + 1, len(block))
               if isinstance(block[index], ast.Assign)
               and isinstance(block[index].targets[0], ast.Name)
               and block[index].targets[0].id == 'duration_s')
    segments = next(node for node in ast.walk(method)
                    if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id == 'segments')
    append = next(node for node in ast.walk(method)
                  if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                  and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == 'append'
                  and isinstance(node.value.func.value, ast.Name)
                  and node.value.func.value.id == 'origin_history')
    function = ast.parse('def persist():\n    pass\n').body[0]
    function.body = block[start:end] + [segments, append]
    notices = []
    namespace = {'accumulated': accumulated, 'origin_history': history,
                 'use_pro_branches': use_pro_branches, 'initial_thinking': False,
                 'reasoning_segments': reasoning_segments, 'transcript_payload': transcript_payload,
                 'self': SimpleNamespace(cancel_event=threading.Event()),
                 '_post': lambda delay, callback: notices.append(callback)}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
                 'actual_gui_persistence', 'exec'), namespace)
    namespace['persist']()
    return notices


@pytest.mark.parametrize('use_pro_branches', [False, True])
def test_best_work_persistence_preserves_selected_whitespace(tmp_path, use_pro_branches):
    from core.best_work import stream_best_work
    from core.context_memory import ChatArchive
    from tests.unit.test_best_work_tournament import make_controller

    original = '    indented first line\n    indented second line\n\n'
    ctrl, _model, _entropies, _actions = make_controller(
        lambda _model, _call: original, database_path=str(tmp_path / 'model.db'))
    selected = ''.join(stream_best_work(ctrl, 'Preserve the answer whitespace', passes=1))
    assert selected == original
    history = ChatArchive(str(tmp_path / 'transcript.db'), 'whitespace')
    notices = _persist_gui_output([selected[:12], selected[12:]], history,
                                  use_pro_branches=use_pro_branches)
    expected = original if use_pro_branches else original.strip()
    assert history[-1] == {'role': 'assistant', 'content': expected}
    assert not notices


@pytest.mark.parametrize('use_pro_branches', [False, True])
@pytest.mark.parametrize('output', ['', ' \n\t '])
def test_empty_gui_response_is_not_persisted(tmp_path, use_pro_branches, output):
    from core.context_memory import ChatArchive

    history = ChatArchive(str(tmp_path / 'transcript.db'), 'empty')
    notices = _persist_gui_output([output], history, use_pro_branches=use_pro_branches)
    assert len(history) == 0
    assert len(notices) == 1


@pytest.mark.parametrize('local_winner', ['A', 'B'])
def test_best_work_transcript_preserves_selected_literal_tool_arguments(tmp_path, local_winner):
    import json
    from core.best_work import stream_best_work
    from core.context_memory import ChatArchive
    from tests.unit.test_best_work_tournament import make_controller

    literal = '<thinking>keep literally</thinking> and <analysis>keep exactly</analysis>'
    arguments = {'path': 'notes.txt', 'content': literal}
    invocation = '<tool_call>' + json.dumps({'name': 'write_file', 'arguments': arguments}) + '</tool_call>'
    def respond(_model, call):
        if call['kind'] == 'compare':
            return '<think>DISCARD_JUDGMENT</think>{"winner":"' + local_winner + '"}'
        if 'ACTUAL selected file contents' in call['prompt']:
            return ('<think>CONTINUATION_ORIGINAL</think>Finished.' if call['kind'] == 'original'
                    else '<think>DISCARD_CONTINUATION</think>Saved.')
        if call['kind'] == 'original':
            return '<think>ORIGINAL_THOUGHT</think>' + (invocation if local_winner == 'A' else 'Original answer.')
        return '<analysis>DISCARD_REVISION</analysis>' + (invocation if local_winner == 'B' else 'Revised answer.')

    ctrl, model, _entropies, actions = make_controller(
        respond, database_path=str(tmp_path / 'model.db'))
    ctrl.app.tools.list_available_tools = lambda: [{'name': 'write_file'}]
    selected = ''.join(stream_best_work(ctrl, 'Save the literal text', passes=2))
    assert actions == [('write_file', arguments)]
    assert literal in selected and '<think>ORIGINAL_THOUGHT</think>' in selected
    assert 'DISCARD_' not in selected and 'BEST_WORK_COMPARISON' not in selected
    assert model.context == ''

    # Exercise production preparation/persistence and the real archive. Display
    # parsing must not change the exact tool data that continues on the next turn.
    history = ChatArchive(str(tmp_path / 'transcript.db'), 'best-work')
    assert not _persist_gui_output([selected], history, use_pro_branches=True)
    assert history[-1] == {'role': 'assistant', 'content': selected}
    continuation = ctrl.app.engine._format_prompt_with_history('Continue', history)
    assert selected in continuation and 'DISCARD_' not in continuation

    # Ordinary chat keeps the existing display-only thinking contract.
    assert not _persist_gui_output(['<think>DISPLAY_ONLY</think>Ordinary answer.'], history,
                                  use_pro_branches=False)
    assert history[-1] == {'role': 'assistant', 'content': 'Ordinary answer.',
                           'display_thoughts': [{'offset': 0, 'text': 'DISPLAY_ONLY'}]}


@pytest.mark.parametrize('tournament', [False, True])
def test_branch_inspector_distinguishes_model_selection_from_verification(monkeypatch, tournament):
    from collections import defaultdict
    from tkinter import ttk
    import app_gui

    widgets = []
    class Widget:
        def __init__(self, *args, **options):
            self.options = options
            self.callbacks = {}
            widgets.append(self)
        def pack(self, **options): pass
        def title(self, value): pass
        def transient(self, value): pass
        def add(self, widget, **options): pass
        def configure(self, **options): self.options.update(options)
        def bind(self, event, callback): self.callbacks[event] = callback
    class Tree(Widget):
        def __init__(self, *args, **options):
            super().__init__(*args, **options)
            self.rows, self.headings, self.selected = {}, {}, []
        def heading(self, key, **options): self.headings[key] = options['text']
        def column(self, *args, **options): pass
        def insert(self, parent, position, *, iid, values): self.rows[iid] = values
        def selection_set(self, iid): self.selected = [iid]
        def selection(self): return self.selected
    class Text(Widget):
        def delete(self, *args): self.text = ''
        def insert(self, position, value): self.text += value

    for name in ('Toplevel', 'Frame', 'Label', 'PanedWindow'):
        monkeypatch.setattr(tk, name, Widget)
    monkeypatch.setattr(tk, 'StringVar', lambda master, value: SimpleNamespace(get=lambda: value))
    monkeypatch.setattr(tk, 'Text', Text)
    for name in ('Spinbox', 'Button'):
        monkeypatch.setattr(ttk, name, Widget)
    monkeypatch.setattr(ttk, 'Treeview', Tree)
    monkeypatch.setattr(app_gui, 'fit_window_to_screen', lambda *args, **kwargs: None)
    monkeypatch.setattr('core.responsive_toolbar.install_wrapping_controls', lambda *args: None)
    settings = SimpleNamespace(best_work_passes=2)
    meta = {'winning_branch': 0, 'branches': [
        {'index': 0, 'temp': .2, 'passed': not tournament, 'reward': 0.0 if tournament else 1.0,
         'candidate': 'Chosen exact answer', 'code': 'Chosen exact answer', 'stderr': ''},
        {'index': 1, 'temp': .6, 'passed': False, 'reward': 0.0,
         'candidate': 'Other answer', 'code': 'Other answer', 'stderr': ''},
    ]}
    if tournament:
        meta['best_work'] = {'method': 'model_tournament'}
    app = SimpleNamespace(root=object(), settings=settings, engine=SimpleNamespace(settings=settings),
                          C=defaultdict(lambda: '#000000'), last_metadata=meta,
                          _make_badge=lambda *args: None)
    SmartAIChatbotApp._on_open_branch_visualizer(app)
    tree = next(widget for widget in widgets if isinstance(widget, Tree))
    inspector = next(widget for widget in widgets if isinstance(widget, Text))
    assert 'Chosen exact answer' in inspector.text
    if tournament:
        assert tree.headings['status'] == 'Selection'
        assert tree.rows['0'][2] == '✓ SELECTED (Winner)'
        assert tree.rows['1'][2] == 'Not selected'
        assert 'selected winner; no sandbox assertions run' in inspector.text
    else:
        assert tree.headings['status'] == 'RLVR Status'
        assert tree.rows['0'][2] == '✓ PASSED (Winner)'
        assert tree.rows['1'][2] == '✗ FAILED'
        assert 'RLVR Sandbox Result: PASSED (Reward = 1.0)' in inspector.text
    tree.selection_set('1')
    tree.callbacks['<<TreeviewSelect>>'](None)
    assert 'Other answer' in inspector.text
    assert ('not selected; no sandbox assertions run' if tournament
            else 'RLVR Sandbox Result: FAILED') in inspector.text


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
    from core.best_work import BEST_WORK_GUIDE
    from tests.unit.test_best_work_tournament import make_controller, previous_draft, comparison

    app = work_app
    assert app.settings.best_work_passes == 2  # Exercise the default through the real dispatcher.
    question = 'Explain the result.'
    original = '<think>ORIGINAL_THOUGHT</think>HIDDEN FIRST DRAFT'
    failure = []
    done = threading.Event()

    def respond(model, call):
        if call['kind'] == 'original':
            return ['<think>ORIGINAL_THOUGHT</think>', 'HIDDEN FIRST DRAFT']
        if call['kind'] == 'refine':
            assert previous_draft(call['prompt']) == original
            assert question in call['prompt']
            return ['<think>DISCARD_REVISION_THOUGHT</think>', 'Published ', 'answer.']
        candidates = comparison(call['prompt'])['candidates']
        assert candidates[0]['text'] == original
        assert candidates[1]['text'] == '<think>ORIGINAL_THOUGHT</think>Published answer.'
        return '<think>DISCARD_JUDGMENT</think>{"winner":"B"}'

    ctrl, model, entropies, _ = make_controller(respond, database_path=app.settings.database_path)
    app.engine.unload_model()
    app.engine = ctrl.app.engine
    if capacity_fallback:
        # The actual context guard has room for the complete original but not
        # the full review prompt. No second decoder or comparison should run.
        packed = app.engine._format_prompt_with_history(
            question, [{'role': 'system', 'content': BEST_WORK_GUIDE}])
        app.engine._context_budget_tokens = len(model.tokenize(packed)) + len(model.tokenize(original))
    app.is_model_loaded = True
    app.chat_history[app.active_tab_id] = []
    app._set_generating_state(True)

    def run():
        try:
            app._process_message_thread(question, question, None, None, True)
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
    assert len(entropies) == 1  # Refining and judging cannot reroute Pro.
    assert len(model.closed) == len(model.calls)
    assert model.context == ''  # Discarded decoder state cannot survive adoption.
    visible = app.chat_stream.get('1.0', 'end')
    labels = [w.cget('text') for w in _descendants(app.chat_stream) if isinstance(w, tk.Label)]
    saved = app.chat_history[app.active_tab_id][-1]
    assert saved['role'] == 'assistant'
    assert '<think>ORIGINAL_THOUGHT</think>' in saved['content']
    assert 'DISCARD_' not in saved['content'] and 'BEST_WORK_COMPARISON' not in saved['content']
    assert 'DISCARD_' not in visible
    assert list(app.thinking_cache.values())[-1] == 'ORIGINAL_THOUGHT'
    if capacity_fallback:
        assert [call['kind'] for call in model.calls] == ['original']
        assert saved['content'] == original
        assert visible.count('HIDDEN FIRST DRAFT') == 1 and 'Published answer.' not in visible
        assert app.last_metadata['passes_completed'] == 1
        assert app.last_metadata['pass_usage'] == {'invocations': 1, 'tokens': 2}
        assert app._live_generation_snapshot['tokens'] == 2
        assert app._live_generation_snapshot['invocations'] == 1
        assert any('used pass 1/2' in label and 'refinement reached Context' in label for label in labels)
        assert not any('pass 2/2' in label and 'complete' in label for label in labels)
    else:
        assert [call['kind'] for call in model.calls] == ['original', 'refine', 'compare']
        assert saved['content'] == '<think>ORIGINAL_THOUGHT</think>Published answer.'
        assert 'HIDDEN FIRST DRAFT' not in visible and visible.count('Published answer.') == 1
        assert app.last_metadata['passes_completed'] == 2
        assert app.last_metadata['pass_usage'] == {'invocations': 3, 'tokens': 6}
        assert app._live_generation_snapshot['tokens'] == 6
        assert app._live_generation_snapshot['invocations'] == 3
        assert any('Best Work 1/1' in label and 'pass 2/2' in label and 'complete' in label for label in labels)
