"""Clean-package GUI/database smoke test; never starts model inference."""
from contextlib import nullcontext
import ctypes
import json
import os
from pathlib import Path
import sys
import tempfile


def verify_macos_identity():
    """Check the running application executable, independently of its Tk title."""
    if sys.platform != 'darwin':
        return
    bundle = Path(__file__).resolve().parents[3]
    assert os.environ.get('SMARTAI_MACOS_NATIVE_LAUNCH') == '1', 'GUI launched as bare Python'
    assert Path(os.environ['SMARTAI_MACOS_APP_BUNDLE']).resolve() == bundle
    assert os.environ.get('SMARTAI_MACOS_APP_NAME') == 'Smart AI Studio'
    assert os.environ.get('SMARTAI_MACOS_APP_IDENTIFIER') == 'com.smartai.studio'
    executable_path = ctypes.CDLL(None)._NSGetExecutablePath
    executable_path.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
    executable_path.restype = ctypes.c_int
    size = ctypes.c_uint32(0)
    executable_path(None, ctypes.byref(size))
    assert size.value > 0, 'Native application executable could not be resolved'
    buffer = ctypes.create_string_buffer(size.value)
    assert executable_path(buffer, ctypes.byref(size)) == 0
    assert Path(os.fsdecode(buffer.value)).resolve() == bundle / 'Contents/MacOS/applet'


def main():
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', RUN_REAL_WEIGHTS_EVAL='0')
    verify_macos_identity()
    persistent = os.environ.get('SMARTAI_SMOKE_DATA_DIR')
    context = nullcontext(persistent) if persistent else tempfile.TemporaryDirectory(prefix='smartai-launch-')
    with context as folder:
        folder = Path(folder).resolve()
        folder.mkdir(parents=True, exist_ok=True)
        os.environ['SMARTAI_DATA_DIR'] = str(folder)
        import tkinter as tk
        from config.settings import Settings
        from memory.db import EpisodicMemoryDB
        from app_gui import SmartAIChatbotApp
        launch = int(os.environ.get('SMARTAI_SMOKE_LAUNCH', '1'))
        preferences = folder / 'app_preferences.json'
        if launch == 2:
            assert preferences.is_file(), 'Relaunch did not reuse the first-launch preferences'
        db=EpisodicMemoryDB(str(folder/'missing'/'database'/'memory.db'))
        assert db.get_stats()['total_interactions'] == 0
        root=tk.Tk()
        root.withdraw()
        app=SmartAIChatbotApp(root, settings=Settings(use_mock=True))
        root.update_idletasks()
        assert 'Smart' in root.title(), root.title()
        assert not app.is_model_loaded
        if launch == 2:
            assert app._memory_limit_custom and not app._memory_limit_enabled
            assert app._memory_limit_gb == 3.25, 'Saved preferences were not restored on relaunch'
        app._memory_limit_enabled = False
        app._memory_limit_custom = True
        app._memory_limit_gb = 3.25
        app._save_memory_limit_preferences()
        assert json.loads(preferences.read_text())['memory_limit']['limit_gb'] == 3.25
        root.destroy()
        print(f'PACKAGE_SMOKE_PASS: launch {launch}, GUI and application identity verified, no model loaded')


if __name__ == '__main__':
    main()
