"""Exercise deferred Tk Scale callbacks without requiring a display server."""
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from core.gui_media import MediaPreview


@pytest.mark.parametrize('fps', [16.0, 29.97])
def test_programmatic_slider_callback_does_not_rewind_or_stall_video(fps):
    viewer = MediaPreview.__new__(MediaPreview)
    viewer._fps, viewer._duration = fps, 3.0
    viewer._frame_index = 0
    viewer._updating_slider = False
    viewer._programmatic_position = None
    viewer._playing = True
    viewer._job = None
    viewer._show_image = Mock()
    viewer._reader = SimpleNamespace(get_data=lambda index: np.full((2, 2, 3), index, dtype=np.uint8))
    viewer.play_button = SimpleNamespace(configure=Mock())
    callbacks, timers = [], {}

    class Scale:
        def set(self, seconds):
            self.value = round(seconds, 2)
            # Match Tk: command runs after set() and the surrounding callback.
            callbacks.append(lambda value=self.value: viewer._seek(str(value)))

        def get(self):
            return self.value

    def after(delay, callback):
        key = 'timer-' + str(len(timers))
        timers[key] = callback
        return key

    viewer.position = Scale()
    viewer.after = after
    for expected in range(1, 5):
        viewer._tick()
        while callbacks:
            callbacks.pop(0)()
        assert viewer._frame_index == expected
        assert viewer._job in timers

    viewer._seek('0.50')  # A real user seek must still move to another frame.
    while callbacks:
        callbacks.pop(0)()
    assert viewer._frame_index == int(0.5 * fps)
