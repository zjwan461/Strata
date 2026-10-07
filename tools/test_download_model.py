"""Tests for tools/download_model.py: the choices come from setup.py (no model is hard-coded), the files land where
setup.py looks for them (models/<tag>/<file>, each with its .done mark), the ModelScope patterns carry the repository's
own folder, and a size the chosen family does not have - or the image encoder where setup never uses it - is refused.
Nothing is downloaded: a fake snapshot_download and a fake setup.download record what they were asked for.

    python -m unittest tools.test_download_model
"""
from __future__ import annotations

import argparse
import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import download_model as D  # noqa: E402
import setup as S  # noqa: E402


def args(**kw) -> argparse.Namespace:
    base = dict(family="qwen", model=None, models_dir=None, source="modelscope", repo=None, revision="master",
                endpoint=None, vision=False, shard=None, list=False, check=False, dry_run=False)
    base.update(kw)
    return argparse.Namespace(**base)


class Options(unittest.TestCase):
    def test_the_defaults_follow_setup_py(self):
        self.assertEqual(D.default_model("qwen"), "IQ3_XXS")           # setup's default size
        self.assertEqual(D.default_model("coder"), "IQ1_M")            # the Coder's only size
        self.assertEqual(D.default_model("unsloth"), "UD-IQ4_XS")      # no IQ3_XXS there: the first choice
        self.assertEqual(D.families_of("IQ2_XS"), ("qwen", "swift"))
        self.assertEqual(D.families_of("IQ3_S"), ("qwen",))            # the original only
        self.assertEqual([m for m in S.MODELS if "coder" in D.families_of(m)], ["IQ1_M"])

    def test_the_repository_and_folder_of_each_family(self):
        self.assertEqual(D.repo_subdir(S.FAMILIES["qwen"], "IQ3_XXS"), "IQ3_XXS")
        self.assertEqual(D.repo_subdir(S.FAMILIES["coder"], "IQ1_M"), "IQ1_M")
        self.assertEqual(D.repo_subdir(S.FAMILIES["unsloth"], "UD-IQ4_XS"), "UD-IQ4_XS")
        self.assertEqual(D.repo_subdir(S.FAMILIES["swift"], "IQ2_XS"), "")     # at the repository's root
        self.assertEqual(D.hf_repo(S.FAMILIES["qwen"]["hf"]), "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF")
        self.assertEqual(D.hf_repo(S.FAMILIES["swift"]["hf"]), "ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF")
        self.assertEqual(D.hf_repo(S.FAMILIES["unsloth"]["mmproj_hf"]),
                         "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF")       # another repository than its shards
        for fam in S.FAMILIES.values():
            self.assertIn(D.hf_repo(fam["hf"]), S.HF_REVISIONS,
                          "every family's repository has a pinned revision in setup.py")

    def test_the_data_folder_setup_remembers_is_the_default_target(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(S, "load_settings", lambda: {"data_dir": d}):
            self.assertEqual(D.default_models_dir(), Path(d) / "models")


class CommandLine(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def run_main(self, *argv):
        """(exit code, what it printed): setup's fail() exits 1, so SystemExit is an answer too."""
        buf = io.StringIO()
        with mock.patch.object(sys, "argv", ["download_model.py", *argv]), contextlib.redirect_stdout(buf):
            try:
                code = D.main()
            except SystemExit as e:
                code = e.code
        return code, buf.getvalue()

    def test_refuses_a_size_this_family_does_not_have(self):
        code, text = self.run_main("--family", "coder", "--model", "IQ3_XXS")
        self.assertEqual(code, 1)
        self.assertIn("has no IQ3_XXS", text)
        self.assertIn("IQ1_M", text)
        code, text = self.run_main("--family", "swift", "--model", "Q2_0")
        self.assertEqual(code, 1)
        self.assertIn("has no Q2_0", text)

    def test_refuses_the_image_encoder_where_setup_never_uses_it(self):
        code, text = self.run_main("--family", "unsloth", "--model", "UD-Q4_K_XL", "--vision", "--dry-run")
        self.assertEqual(code, 1)
        self.assertIn("no image support", text)
        self.assertIn("UD-IQ4_XS", text)                              # what does support images

    def test_refuses_a_shard_that_does_not_exist(self):
        code, text = self.run_main("--model", "IQ3_XXS", "--shard", "3", "--dry-run")
        self.assertEqual(code, 1)
        self.assertIn("--shard takes 1..2", text)

    def test_dry_run_prints_the_plan_and_downloads_nothing(self):
        with mock.patch.object(D, "ms_snapshot") as ms, mock.patch.object(S, "download") as hf:
            code, text = self.run_main("--family", "coder", "--model", "IQ1_M", "--models-dir", self.tmp.name,
                                       "--dry-run")
        self.assertEqual(code, 0)
        self.assertEqual((ms.call_count, hf.call_count), (0, 0), "a dry run downloads nothing, from either source")
        self.assertIn("coder-IQ1_M", text)
        self.assertEqual(list(Path(self.tmp.name).rglob("*")), [], "and writes nothing either")


class Placement(unittest.TestCase):
    """What ModelScope writes (the repository's own folder) must end up where setup.py looks for it."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.calls = []

    def tearDown(self):
        self.tmp.cleanup()

    def fake_ms(self, repo, revision, endpoint, dest, patterns, dry=False):
        """ModelScope's side of the contract: a file under its repository path, <dest>/<pattern>."""
        self.calls.append((repo, revision, patterns))
        for p in patterns:
            f = Path(dest) / p
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_bytes(b"x" * 64)

    def test_modelscope_files_move_into_the_folder_setup_expects(self):
        fam, model, tag = S.FAMILIES["coder"], "IQ1_M", "coder-IQ1_M"
        name = S.model_file(fam, model, 1)
        target = self.root / tag / name
        with mock.patch.object(D, "ms_snapshot", self.fake_ms), mock.patch.object(S, "whole_shard", lambda p: True):
            D.downloads(fam, model, self.root, tag, [target], False, args(vision=False))
        self.assertEqual(self.calls, [("ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF", "master", [f"IQ1_M/{name}"])])
        self.assertEqual(target.read_bytes(), b"x" * 64)
        self.assertTrue(S.done(target))                               # setup.py finds it and skips the download
        self.assertFalse((self.root / "IQ1_M").exists(), "the folder ModelScope made is left behind")

    def test_the_original_sizes_stay_where_modelscope_wrote_them(self):
        fam, model, tag = S.FAMILIES["qwen"], "IQ3_XXS", "IQ3_XXS"
        name = S.model_file(fam, model, 2)
        target = self.root / tag / name
        with mock.patch.object(D, "ms_snapshot", self.fake_ms), mock.patch.object(S, "whole_shard", lambda p: True):
            D.downloads(fam, model, self.root, tag, [target], False, args())
        self.assertEqual(self.calls[0][2], [f"IQ3_XXS/{name}"])
        self.assertTrue(target.exists())
        self.assertEqual(self.calls[0][2][0].split("/")[0], tag)      # the pattern and the folder are the same

    def test_a_short_file_is_not_marked_downloaded(self):
        fam, model, tag = S.FAMILIES["qwen"], "IQ3_XXS", "IQ3_XXS"
        target = self.root / tag / S.model_file(fam, model, 1)
        with mock.patch.object(D, "ms_snapshot", self.fake_ms), \
                self.assertRaises(SystemExit):
            D.downloads(fam, model, self.root, tag, [target], False, args())
        self.assertFalse(S.done(target), "a file that is not whole must not get a finish mark")

    def test_the_huggingface_source_uses_setups_own_downloader(self):
        fam, model, tag = S.FAMILIES["qwen"], "IQ3_S", "IQ3_S"
        name = S.model_file(fam, model, 1)
        target = self.root / tag / name
        with mock.patch.object(S, "download") as dl:
            D.downloads(fam, model, self.root, tag, [target], True, args(source="huggingface"))
        urls = [c.args[0] for c in dl.call_args_list]
        self.assertEqual(urls[0], fam["hf"].format(q=model) + name)
        self.assertIn(S.HF_REVISIONS[D.hf_repo(fam["hf"])], urls[0], "the revision setup.py pins")
        self.assertEqual(urls[1], fam["mmproj_hf"] + fam["mmproj"])
        self.assertEqual(dl.call_args_list[0].args[1], target)


class Check(unittest.TestCase):
    def run_check(self, *argv):
        buf = io.StringIO()
        with mock.patch.object(sys, "argv", ["download_model.py", "--model", "IQ3_XXS", "--shard", "1",
                                             "--check", *argv]), contextlib.redirect_stdout(buf):
            return D.main(), buf.getvalue()

    def test_check_says_what_is_there_and_exits_non_zero_when_it_is_not(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(S, "whole_shard", lambda p: True):
            code, text = self.run_check("--models-dir", d)
            self.assertEqual(code, 1)                                  # nothing downloaded yet
            self.assertIn("missing", text)
            f = Path(d) / "IQ3_XXS" / S.model_file(S.FAMILIES["qwen"], "IQ3_XXS", 1)
            f.parent.mkdir(parents=True)
            f.write_bytes(b"x" * 16)
            code, text = self.run_check("--models-dir", d)
            self.assertEqual(code, 1)                                  # there, but with no finish mark yet
            self.assertIn("no finish mark", text)
            S.mark(f)
            code, text = self.run_check("--models-dir", d)
            self.assertEqual(code, 0)
            self.assertIn("whole", text)


if __name__ == "__main__":
    unittest.main()