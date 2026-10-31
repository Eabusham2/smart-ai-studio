"""Behavioral tests for native speed calibration without model dependencies."""
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("_speed_tuning_test_module", ROOT / "core/speed_tuning.py")
speed = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = speed
SPEC.loader.exec_module(speed)

BASE = {"n_threads": 4}
CANDIDATE = {"n_threads": 2}


def sample(prefill=100.0, decode=50.0, digest="same-complete-output"):
    return speed.SpeedSample(100, 25, 100.0 / prefill, 24.0 / decode, digest)


def improved_result():
    return speed.tune_speed(BASE, [CANDIDATE],
                            lambda opts: sample(120, 60) if opts == CANDIDATE else sample())


class TestMeasurements(unittest.TestCase):
    def test_selects_only_repeated_improvement_and_reports_real_rates(self):
        calls = []

        def measure(opts):
            calls.append(dict(opts))
            return sample(120, 60) if opts == CANDIDATE else sample()

        result = speed.tune_speed(BASE, [CANDIDATE], measure)
        self.assertEqual(result.status, "improved")
        self.assertEqual(result.settings, CANDIDATE)
        self.assertAlmostEqual(result.gain_pct, 20.0)
        self.assertAlmostEqual(result.selected["decode_tokens_per_second"], 60.0)
        self.assertEqual(calls, [BASE, BASE, BASE, BASE, CANDIDATE, BASE, BASE, CANDIDATE, BASE])
        self.assertEqual(BASE, {"n_threads": 4})

    def test_decode_rate_excludes_first_token(self):
        value = speed.SpeedSample(80, 5, 2.0, 2.0, "tokens").rates()
        self.assertEqual(value["decode_tokens_per_second"], 2.0)

    def test_cold_kernel_warmup_is_discarded_before_noise_estimation(self):
        calls = 0

        def measure(opts):
            nonlocal calls
            calls += 1
            if calls == 1:
                return sample(10, 5)
            return sample(110, 55) if opts == CANDIDATE else sample()

        result = speed.tune_speed(BASE, [CANDIDATE], measure)
        self.assertEqual(result.status, "improved")
        self.assertAlmostEqual(result.gain_pct, 10.0)
        self.assertEqual(result.trials[0]["noise_pct"], 0)

    def test_big_average_gain_cannot_hide_prefill_regression(self):
        result = speed.tune_speed(BASE, [CANDIDATE],
                                  lambda opts: sample(95, 100) if opts == CANDIDATE else sample())
        self.assertEqual(result.status, "baseline")
        self.assertEqual(result.settings, BASE)
        self.assertIn("one metric regressed", result.trials[0]["reason"])

    def test_candidate_must_pass_every_round(self):
        candidate_calls = 0

        def measure(opts):
            nonlocal candidate_calls
            if opts == CANDIDATE:
                candidate_calls += 1
                return sample(120, 60) if candidate_calls == 1 else sample(99, 49.5)
            return sample()

        result = speed.tune_speed(BASE, [CANDIDATE], measure)
        self.assertEqual(result.status, "baseline")
        self.assertTrue(result.trials[0]["accepted"])
        self.assertFalse(result.trials[1]["accepted"])

    def test_noise_margin_rejects_unsubstantiated_improvement(self):
        calls = 0

        def measure(opts):
            nonlocal calls
            calls += 1
            if calls == 2:
                return sample(96, 48)
            return sample(105, 52.5) if opts == CANDIDATE else sample()

        result = speed.tune_speed(BASE, [CANDIDATE], measure)
        self.assertEqual(result.status, "baseline")
        self.assertGreater(result.trials[0]["threshold_pct"], 8)

    def test_drift_margin_rejects_warming_or_thermal_bias(self):
        position = 0
        samples = [sample(), sample(), sample()]
        for _ in range(2):
            samples += [sample(100, 50), sample(120, 60), sample(115, 57.5)]

        def measure(_opts):
            nonlocal position
            current = samples[position]
            position += 1
            return current

        result = speed.tune_speed(BASE, [CANDIDATE], measure)
        self.assertEqual(result.status, "baseline")
        self.assertGreaterEqual(result.trials[0]["threshold_pct"], 29.9)

    def test_changed_output_and_changed_token_counts_reject_candidate(self):
        for changed in (sample(150, 75, "different-output"),
                        speed.SpeedSample(100, 24, 0.5, 0.25, "same-complete-output")):
            with self.subTest(changed=changed):
                result = speed.tune_speed(BASE, [CANDIDATE],
                                          lambda opts: changed if opts == CANDIDATE else sample())
                self.assertEqual(result.status, "baseline")
                self.assertTrue(all(row["reason"] == "output changed" for row in result.trials))

    def test_nonrepeatable_baseline_is_incomplete_and_not_cacheable(self):
        count = 0

        def measure(_opts):
            nonlocal count
            count += 1
            return sample(digest=str(count))

        result = speed.tune_speed(BASE, [CANDIDATE], measure)
        self.assertEqual(result.status, "error")
        self.assertEqual(result.settings, BASE)
        self.assertEqual(len(result.trials), 0)

    def test_error_after_winning_trial_discards_partial_winner(self):
        calls = 0

        def measure(opts):
            nonlocal calls
            calls += 1
            if calls == 7:
                raise RuntimeError("native backend failed")
            return sample(120, 60) if opts == CANDIDATE else sample()

        result = speed.tune_speed(BASE, [CANDIDATE], measure)
        self.assertEqual(result.status, "error")
        self.assertEqual(result.settings, BASE)
        self.assertEqual(result.gain_pct, 0)
        self.assertTrue(result.trials[0]["accepted"])

    def test_invalid_timings_or_one_token_output_never_win(self):
        samples = [speed.SpeedSample(10, 25, float("nan"), 1, "x"),
                   speed.SpeedSample(10, 25, 1, float("inf"), "x"),
                   speed.SpeedSample(10, 1, 1, 1, "x"),
                   speed.SpeedSample(10, 25, 0, 1, "x")]
        for bad in samples:
            with self.subTest(bad=bad):
                result = speed.tune_speed(BASE, [CANDIDATE], lambda opts: bad)
                self.assertEqual(result.status, "error")
                self.assertEqual(result.settings, BASE)

    def test_quality_knobs_are_never_passed_to_measure(self):
        calls = []
        result = speed.tune_speed(BASE, [{"n_threads": 2, "temperature": 0}],
                                  lambda opts: calls.append(opts))
        self.assertEqual(result.status, "error")
        self.assertEqual(calls, [])

    def test_callback_mutation_cannot_change_baseline(self):
        def measure(opts):
            candidate = opts == CANDIDATE
            opts["n_threads"] = 999
            return sample(120, 60) if candidate else sample()

        result = speed.tune_speed(BASE, [CANDIDATE], measure)
        self.assertEqual(result.settings, CANDIDATE)
        self.assertEqual(BASE["n_threads"], 4)

    def test_candidate_order_rotates_and_best_repeated_candidate_wins(self):
        other = {"n_threads": 8}
        calls = []

        def measure(opts):
            calls.append(dict(opts))
            rate = 1.1 if opts == CANDIDATE else 1.2 if opts == other else 1.0
            return sample(100 * rate, 50 * rate)

        result = speed.tune_speed(BASE, [CANDIDATE, other], measure)
        self.assertEqual(result.settings, other)
        self.assertEqual([opts for opts in calls if opts != BASE], [CANDIDATE, other, other, CANDIDATE])

    def test_cancel_timeout_and_memory_failure_discard_all_trials(self):
        for expected in ("cancelled", "timeout", "memory_pressure"):
            with self.subTest(expected=expected):
                state = {"calls": 0, "clock": 0.0}

                def measure(opts):
                    state["calls"] += 1
                    state["clock"] += 1
                    return sample(120, 60) if opts == CANDIDATE else sample()

                kwargs = {}
                if expected == "cancelled":
                    kwargs["cancelled"] = lambda: state["calls"] >= 6
                elif expected == "timeout":
                    kwargs["clock"] = lambda: state["clock"]
                    kwargs["budget_seconds"] = 6
                else:
                    kwargs["memory_ok"] = lambda: state["calls"] < 6
                result = speed.tune_speed(BASE, [CANDIDATE], measure, **kwargs)
                self.assertEqual(result.status, expected)
                self.assertEqual(result.settings, BASE)
                self.assertEqual(result.gain_pct, 0)

    def test_adapter_can_signal_a_token_boundary_stop(self):
        for expected in ("cancelled", "timeout", "memory_pressure"):
            with self.subTest(expected=expected):
                def measure(_opts):
                    raise speed.SpeedTuningStopped(expected, "measurement stopped")
                result = speed.tune_speed(BASE, [CANDIDATE], measure)
                self.assertEqual(result.status, expected)
                self.assertEqual(result.settings, BASE)

    def test_empty_candidate_list_is_not_a_measurement(self):
        result = speed.tune_speed(BASE, [BASE], lambda opts: self.fail("no measurement needed"))
        self.assertEqual(result.status, "unavailable")


class TestProfiles(unittest.TestCase):
    def test_roundtrip_uses_only_complete_results_and_returns_isolated_values(self):
        with tempfile.TemporaryDirectory() as folder:
            store = speed.SpeedProfileStore(str(Path(folder) / "profiles.json"))
            result = improved_result()
            self.assertTrue(store.put("fingerprint", result))
            record = store.get("fingerprint")
            self.assertEqual(record["settings"], CANDIDATE)
            self.assertEqual(record["fingerprint"], "fingerprint")
            self.assertEqual(result.fingerprint, "")
            record["settings"]["n_threads"] = 30
            self.assertEqual(store.get("fingerprint")["settings"], CANDIDATE)
            for status in ("cancelled", "timeout", "error", "memory_pressure", "unavailable"):
                incomplete = speed.SpeedResult(status, BASE, "incomplete")
                self.assertFalse(store.put("fingerprint", incomplete))
                self.assertEqual(store.get("fingerprint")["settings"], CANDIDATE)

    def test_corrupt_unknown_and_nonfinite_cache_records_fail_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "profiles.json"
            store = speed.SpeedProfileStore(str(path))
            for text in ("not JSON", '{"schema":99,"profiles":{}}', "[]"):
                path.write_text(text)
                self.assertIsNone(store.get("key"))
            for mutation in ({"settings": {"temperature": 0}}, {"gain_pct": float("nan")},
                             {"settings": {"n_threads": True}}, {"settings": {"n_threads": 100000}}):
                record = improved_result().to_dict()
                record.update(mutation)
                record["fingerprint"] = "key"
                path.write_text(json.dumps({"schema": 1, "profiles": {"key": record}}))
                self.assertIsNone(store.get("key"))

    def test_concurrent_writers_merge_instead_of_overwriting_other_profiles(self):
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / "profiles.json")

            def write(index):
                return speed.SpeedProfileStore(path).put(str(index), improved_result())

            with ThreadPoolExecutor(max_workers=8) as pool:
                self.assertTrue(all(pool.map(write, range(20))))
            store = speed.SpeedProfileStore(path)
            self.assertTrue(all(store.get(str(index)) is not None for index in range(20)))

    def test_atomic_replace_failure_preserves_existing_cache(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "profiles.json"
            store = speed.SpeedProfileStore(str(path))
            self.assertTrue(store.put("first", improved_result()))
            before = path.read_bytes()
            with patch.object(speed.os, "replace", side_effect=OSError("disk problem")):
                self.assertFalse(store.put("second", improved_result()))
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(list(Path(folder).glob("*.tmp")), [])

    def test_fingerprint_invalidates_runtime_model_context_adapter_and_gpu(self):
        with tempfile.TemporaryDirectory() as folder:
            model = Path(folder) / "model"
            model.mkdir()
            config = model / "config.json"
            config.write_text('{"layers":12}')
            (model / "weights.safetensors").write_bytes(b"dummy weights")
            adapter = Path(folder) / "adapter.npz"
            adapter.write_bytes(b"adapter one")
            projector = Path(folder) / "projector.gguf"
            projector.write_bytes(b"projector one")
            kwargs = {"context": 32768, "adapter_path": str(adapter), "projector_path": str(projector),
                      "hardware": {"gpu": "GPU A", "total_ram_gb": 32, "available_ram_gb": 10}}
            with patch.object(speed.metadata, "version", return_value="1.0"):
                initial = speed.speed_fingerprint("gguf", str(model), **kwargs)
                same = dict(kwargs, hardware={"gpu": "GPU A", "total_ram_gb": 32, "available_ram_gb": 2})
                self.assertEqual(initial, speed.speed_fingerprint("gguf", str(model), **same))
                for updates in ({"context": 16384}, {"adapter_path": None}, {"projector_path": None},
                                {"hardware": {"gpu": "GPU B", "total_ram_gb": 32}}):
                    self.assertNotEqual(initial, speed.speed_fingerprint("gguf", str(model), **dict(kwargs, **updates)))
                config.write_text('{"layers":13}')
                self.assertNotEqual(initial, speed.speed_fingerprint("gguf", str(model), **kwargs))
            with patch.object(speed.metadata, "version", return_value="2.0"):
                self.assertNotEqual(initial, speed.speed_fingerprint("gguf", str(model), **kwargs))

    def test_detected_gpu_identity_and_unavailable_package_file_invalidate_cache(self):
        with tempfile.TemporaryDirectory() as folder:
            native = Path(folder) / "native.py"
            native.write_text("runtime version one")
            module = type("Module", (), {"__file__": str(native)})()
            with patch.object(speed.metadata, "version", side_effect=speed.metadata.PackageNotFoundError), \
                 patch.dict(speed.sys.modules, {"llama_cpp": module}), \
                 patch.object(speed, "_hardware_identity", return_value={"gpus": [{"id": "A"}]}):
                first = speed.speed_fingerprint("gguf", "remote/model", context=4096)
                native.write_text("runtime version two has changed")
                second = speed.speed_fingerprint("gguf", "remote/model", context=4096)
                self.assertNotEqual(first, second)
                with patch.object(speed, "_hardware_identity", return_value={"gpus": [{"id": "B"}]}):
                    third = speed.speed_fingerprint("gguf", "remote/model", context=4096)
                    self.assertNotEqual(second, third)

    def test_hf_identifiers_use_only_local_snapshot_and_invalidate_after_update(self):
        import types
        with tempfile.TemporaryDirectory() as folder:
            snapshot = Path(folder) / "snapshot"
            snapshot.mkdir()
            weights = snapshot / "weights.safetensors"
            weights.write_bytes(b"first")
            calls = []
            def local_snapshot(**kwargs):
                calls.append(kwargs)
                return str(snapshot)
            module = types.SimpleNamespace(snapshot_download=local_snapshot)
            with patch.dict(sys.modules, {"huggingface_hub": module}):
                first = speed.speed_fingerprint("mlx", "org/model", context=4096)
                second = speed.speed_fingerprint("mlx", "org/model", context=4096)
                self.assertEqual(first, second)
                weights.write_bytes(b"updated model bytes")
                self.assertNotEqual(first, speed.speed_fingerprint("mlx", "org/model", context=4096))
            self.assertTrue(all(call == {"repo_id": "org/model", "local_files_only": True} for call in calls))

    def test_unresolved_model_identity_cannot_reuse_a_saved_profile(self):
        import types
        module = types.SimpleNamespace(snapshot_download=lambda **kwargs: (_ for _ in ()).throw(FileNotFoundError()))
        with patch.dict(sys.modules, {"huggingface_hub": module}):
            first = speed.speed_fingerprint("mlx", "missing/model", context=4096)
            second = speed.speed_fingerprint("mlx", "missing/model", context=4096)
            self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
