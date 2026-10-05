
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from yin_tests import run_yin_orientation as runner
spec = importlib.util.spec_from_file_location("queue_yin", runner.ROOT / "yin_tests/queue_yin_orientation.py")
queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue)


class OrientationTests(unittest.TestCase):
    def test_rates_reach_memory_and_both_test_probes(self):
        args = SimpleNamespace(seed=101, noise_phase="both", layer="h", sigma=2.,
                               study_fixations=1, test_fixations=1, num_test=1)
        expected = {("U", "U"): (.1, .1), ("I", "I"): (.3, .3),
                    ("U", "I"): (.1, .3), ("I", "U"): (.3, .1)}
        for (study, test), pair in expected.items():
            seen = []
            def encode(model, transform, crops, device, noise, *extra):
                seen.append((transform, noise))
                return torch.zeros(1, 256)
            with patch.object(runner.yin, "encode", side_effect=encode), \
                 patch.object(runner.yin, "load_item_fixations", return_value=torch.zeros(1, 3, 180, 180)), \
                 patch.object(runner.yin, "compute_familiarity_score", side_effect=[1., 0.]):
                ps, pt = runner.noise_pair(study, test, .1, .3)
                acc = runner.yin.run_condition(None, "cpu", args, None, [(0, "old")], [(1, "new")],
                                               study, test, ps, p_test_in=pt)
            self.assertEqual((ps, pt), pair)
            self.assertEqual(seen, [(study, pair[0]), (test, pair[1]), (test, pair[1])])
            self.assertEqual(acc, 1.)

    def test_per_seed_item_splits_are_repeatable_and_disjoint(self):
        items = list(range(64))
        a, b = runner.split_items(items, 101)
        self.assertEqual((len(a), len(b)), (40, 24))
        self.assertFalse(set(a) & set(b))
        self.assertEqual((a, b), runner.split_items(items, 101))
        self.assertNotEqual((a, b), runner.split_items(items, 102))
        self.assertEqual(items, list(range(64)))

    def test_fits_independent_and_unreachable_is_recorded(self):
        uu = runner.choose_fit({0.: .99, .2: .9629, .4: .6}, .9629)
        ii = runner.choose_fit({0.: .8188, .2: .7, .4: .5}, .8188)
        self.assertEqual((uu["p"], ii["p"]), (.2, 0.))
        bad = runner.choose_fit({0.: .8, .2: .7, .5: .5}, .9629)
        self.assertEqual(bad["status"], "target_outside_sampled_range")
        self.assertEqual(bad["p"], 0.)

    def test_calibration_resumes_samples_without_repeating(self):
        class Fake:
            calls = []
            def measure(self, seed, study, test, ps, pt):
                self.calls.append((seed, study, test, ps, pt))
                return 1. - ps
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cache.json"
            args = argparse.Namespace(coarse_step=.25, fine_step=.01, fine_radius=0.)
            fake, cache = Fake(), {"U": {"0.000000": {"101": 1.}}}
            fit = runner.calibrate(fake, [101, 102], "U", .75, args, cache, path)
            self.assertEqual(fit["p"], .25)
            self.assertEqual(len(fake.calls), 5)
            runner.calibrate(fake, [101, 102], "U", .75, args, json.loads(path.read_text()), path)
            self.assertEqual(len(fake.calls), 5)


class QueueTests(unittest.TestCase):
    def test_queue_ordering_and_final_barrier(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            ck = {"epoch": 7, "run_dir": tmp, "checkpoint": str(out/"weights.pth"), "sha256": "abc"}
            log = out / "train.log"
            log.write_text("Done. Best valid acc 90%")
            models = [{"model": "model2", "pid": None, "token": None, "model2": ck}]
            models += [{"model": f"train{i}", "pid": i, "token": "train", "latest": ck,
                        "log": str(log), "source_run_dir": tmp} for i in (1, 2, 3)]
            manifest = dict(source_hashes={}, source=tmp, models=models, initial_wait=[],
                            categories=queue.CATEGORIES, seeds="101-150", target_uu=.9629,
                            target_ii=.8188, python="python", gpus=[0, 3], min_free_mb=4500)
            runner.atomic_json(out / "manifest.json", manifest)
            pid_count = iter(range(9000, 9010))
            def spawn(*a, **kw):
                return SimpleNamespace(pid=next(pid_count), poll=lambda: None)
            with patch.object(queue, "free_gpu_memory", return_value={0: 9000, 3: 9000}), \
                 patch.object(queue.subprocess, "Popen", side_effect=spawn), \
                 patch.object(queue, "process_token", return_value="train"), \
                 patch.object(queue, "snapshot", return_value=ck):
                queue.run_queue(out, once=True)
                path = out / "queue_state.json"
                state = json.loads(path.read_text())
                running = [j for j in state["jobs"].values() if j["status"] == "running"]
                self.assertEqual(len(running), 2)
                self.assertEqual({j["phase"] for j in running}, {"model2"})
                for job in state["jobs"].values():
                    if job["phase"] == "model2": job["status"] = "done"
                runner.atomic_json(path, state)
                queue.run_queue(out, once=True)
                state = json.loads(path.read_text())
                self.assertEqual({j["phase"] for j in state["jobs"].values() if j["status"] == "running"}, {"latest"})
                for job in state["jobs"].values(): job["status"] = "done"
                runner.atomic_json(path, state)
                queue.run_queue(out, once=True)
                state = json.loads(path.read_text())
                self.assertEqual(state["status"], "waiting_for_all_training")
                self.assertFalse(state["final_prepared"])
                with patch.object(queue, "all_trainings_finished", return_value=True):
                    queue.run_queue(out, once=True)
                state = json.loads(path.read_text())
                self.assertTrue(state["final_prepared"])
                self.assertEqual(sum(j["phase"] == "final" for j in state["jobs"].values()), 6)

    def test_training_barrier_and_pid_reuse(self):
        models = [{"pid": 1, "token": "a"}, {"pid": 2, "token": "b"}]
        with patch.object(queue, "process_token", side_effect=lambda pid: {1: "a", 2: None}[pid]):
            self.assertFalse(queue.all_trainings_finished(models))
        with patch.object(queue, "process_token", return_value="reused"):
            self.assertTrue(queue.all_trainings_finished(models))
        self.assertEqual(queue.process_token(os.getpid()), queue.process_token(os.getpid()))
        self.assertIsNotNone(queue.process_token(os.getpid()))

    def test_snapshot_uses_last_epoch_and_is_immutable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, dest = root / "run", root / "snapshot"
            source.mkdir()
            for name in ("config.json", "label_map.json"):
                (source / name).write_text("{}")
            state = {"epoch": 17, "model": {"w": torch.ones(2)}}
            torch.save(state, source / "checkpoint_last.pth")
            got = queue.snapshot(source, dest, "latest")
            torch.save({"epoch": 18, "model": {"w": torch.zeros(2)}}, source / "checkpoint_last.pth")
            self.assertEqual(got["epoch"], 17)
            self.assertEqual(queue.snapshot(source, dest, "latest"), got)
            self.assertTrue(torch.equal(torch.load(got["checkpoint"], weights_only=True)["w"], torch.ones(2)))


if __name__ == "__main__":
    unittest.main()
