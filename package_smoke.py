"""Clean-package GUI/database smoke test; never starts model inference."""
import os
from pathlib import Path
import tempfile


def main():
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', RUN_REAL_WEIGHTS_EVAL='0')
    with tempfile.TemporaryDirectory(prefix='smartai-launch-') as folder:
        os.environ['SMARTAI_DATA_DIR'] = folder
        import tkinter as tk
        from config.settings import Settings
        from memory.db import EpisodicMemoryDB
        from app_gui import SmartAIChatbotApp
        db=EpisodicMemoryDB(str(Path(folder)/'missing'/'database'/'memory.db'))
        assert db.get_stats()['total_interactions'] == 0
        root=tk.Tk()
        root.withdraw()
        app=SmartAIChatbotApp(root, settings=Settings(use_mock=True))
        root.update_idletasks()
        assert 'Smart' in root.title(), root.title()
        assert not app.is_model_loaded
        root.destroy()
        print('PACKAGE_SMOKE_PASS: GUI initialized, missing database created, no model loaded')


if __name__ == '__main__':
    main()
