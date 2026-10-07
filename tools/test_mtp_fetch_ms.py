"""Tests for tools/mtp_fetch_ms.py: it must produce exactly what tools/mtp_fetch.py produces (the same tensor
bytes and manifest, so mtp_pack/mtp_rt are unaffected), keep the #327 range discipline against a ModelScope
endpoint that ignores Range, slice a whole shard when told to, join --jobs parts and resume them, and check a
shard against ModelScope's own sha256.  A fake ModelScope endpoint behind a mocked urlopen: nothing is downloaded.

    python -m unittest tools.test_mtp_fetch_ms
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import test_mtp_fetch as T  # noqa: E402  the fake checkpoint and its HF-style mirror
import mtp_fetch_ms as MS  # noqa: E402

M, FILES, HASHES = T.M, T.FILES, T.HASHES


class Endpoint:
    """urlopen for the fake ModelScope: /api/v1/models/<id>/repo?Revision=&FilePath= and the /repo/files listing.
    `ignore_range`: 200 with the whole file (`cut`: cut to the asked length, as the proxy in #327 did);
    `wrong_shard_sha`: the listing claims another sha256 for these files."""

    def __init__(self, listing=True, ignore_range=False, cut=False, wrong_shard_sha=()):
        self.listing, self.ignore_range, self.cut = listing, ignore_range, cut
        self.wrong_shard_sha = set(wrong_shard_sha)
        self.ranges, self.fulls = [], []

    @staticmethod
    def body(path):
        return T.INDEX if path == "model.safetensors.index.json" else FILES[path]

    def files(self):
        return [dict(Path=p, Type="blob", Size=len(self.body(p)),
                     Sha256="0" * 64 if p in self.wrong_shard_sha else hashlib.sha256(self.body(p)).hexdigest())
                for p in sorted(FILES) + ["model.safetensors.index.json"]]

    def __call__(self, req, timeout=None):
        u = urlparse(req.full_url)
        q = parse_qs(u.query)
        if u.path.endswith("/repo/files"):
            if not self.listing:
                return T.Response(b"{}", 404, {})
            return T.Response(json.dumps({"Data": {"Files": self.files()}}).encode(), 200, {})
        path, body, rng = q["FilePath"][0], self.body(q["FilePath"][0]), req.get_header("Range")
        if rng is None:
            self.fulls.append(path)
            return T.Response(body, 200, {"Content-Length": str(len(body))})
        a, b = map(int, rng.split("=")[1].split("-"))
        self.ranges.append((path, a, b))
        if self.ignore_range:
            return T.Response(body[:b - a + 1] if self.cut else body, 200, {})
        return T.Response(body[a:b + 1], 206, {"Content-Range": "bytes %d-%d/%d" % (a, b, len(body))})


class FetchCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name)
        self.patches = [mock.patch.object(M, "SHA256", HASHES), mock.patch.object(M.time, "sleep", lambda s: None),
                        mock.patch.object(MS, "CHUNK", 16), mock.patch.object(MS, "CHECK_HASHES", True)]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        self.tmp.cleanup()

    def ms_fetch(self, endpoint, jobs=1, fallback=False, out=None, only=None):
        out = Path(out or self.out)
        urlopen = mock.patch.object(M.urllib.request, "urlopen", endpoint)
        with urlopen, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as err:
            src = MS.Source(out, fallback=fallback)
            try:
                MS.fetch(str(out), only, jobs, src)
            finally:
                src.done()
        return err.getvalue()

    def ms_inventory(self, endpoint, out=None):
        out = Path(out or self.out)
        with mock.patch.object(M.urllib.request, "urlopen", endpoint), contextlib.redirect_stdout(io.StringIO()):
            src = MS.Source(out)
            try:
                MS.inventory(str(out), src)
            finally:
                src.done()

    def tensor(self, name, out=None):
        return (Path(out or self.out) / "tensors" / (name + ".bin")).read_bytes()

    def manifest(self, out=None):
        return {r["name"]: r["sha256"] for r in json.loads((Path(out or self.out) / "mtp-manifest.json").read_text())}


class Output(FetchCase):
    def test_matches_tools_mtp_fetch_byte_for_byte(self):
        """mtp_pack.py reads a manifest and memmaps tensors/*.bin: both must be the pinned tool's."""
        hf = Path(self.tmp.name) / "hf"
        with mock.patch.object(M.urllib.request, "urlopen", T.Mirror()), mock.patch.object(M, "REPO", M.PINNED), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            M.fetch(str(hf), None)
        self.ms_fetch(Endpoint())
        self.assertEqual([self.tensor("mtp.a"), self.tensor("mtp.b")],
                         [(hf / "tensors" / "mtp.a.bin").read_bytes(), (hf / "tensors" / "mtp.b.bin").read_bytes()])
        self.assertEqual(self.manifest(), {r["name"]: r["sha256"] for r in json.loads(
            (hf / "mtp-manifest.json").read_text())})
        self.assertEqual(self.manifest(), HASHES)

    def test_inventory_records_the_modelscope_source(self):
        self.ms_inventory(Endpoint())
        inv = json.loads((self.out / "mtp-inventory.json").read_text())
        self.assertEqual(inv["repo"], MS.REPO)
        self.assertEqual([r["name"] for r in inv["tensors"]], ["mtp.a", "mtp.b"])
        self.assertTrue((self.out / "mtp-inventory.md").read_text().startswith("# MTP block"))

    def test_fetch_reuses_its_inventory_and_an_already_fetched_tensor(self):
        endpoint = Endpoint()
        self.ms_fetch(endpoint)
        asked = list(endpoint.ranges)
        err = self.ms_fetch(endpoint)                            # the inventory and the tensors are both there
        self.assertIn("mtp.a: already fetched, kept", err)
        self.assertEqual(endpoint.ranges, asked, "a second run asks for nothing at all")
        self.assertEqual(len(set(asked)), len(asked), "no range was ever asked for twice")


class RangeDiscipline(FetchCase):
    def test_an_endpoint_that_ignores_range_is_refused(self):
        for cut in (False, True):
            with self.subTest(cut=cut), self.assertRaisesRegex(IOError, "not honoured"):
                self.ms_fetch(Endpoint(ignore_range=True, cut=cut))

    def test_fallback_full_shard_slices_the_downloaded_shard(self):
        endpoint = Endpoint(ignore_range=True)
        err = self.ms_fetch(endpoint, fallback=True)
        self.assertIn("reading the whole shard once", err)
        self.assertEqual((self.tensor("mtp.a"), self.tensor("mtp.b")), (T.A, T.B))
        self.assertEqual(sorted(p for p in endpoint.fulls if p.endswith(".safetensors")),
                         ["model-1.safetensors", "model-2.safetensors"])
        shards = self.out / "_shards"
        self.assertEqual(list(shards.glob("*")) if shards.exists() else [], [], "the shards are removed")

    def test_a_shard_that_is_not_modelscope_sha256_is_refused(self):
        with self.assertRaisesRegex(IOError, "not ModelScope's sha256"):
            self.ms_fetch(Endpoint(ignore_range=True, wrong_shard_sha=["model-2.safetensors"]), fallback=True)

    def test_a_listing_that_is_not_available_is_only_a_warning(self):
        err = self.ms_fetch(Endpoint(listing=False, ignore_range=True), fallback=True)
        self.assertIn("file listing is unavailable", err)
        self.assertEqual((self.tensor("mtp.a"), self.tensor("mtp.b")), (T.A, T.B))


class Jobs(FetchCase):
    def test_parts_are_joined_in_order_and_removed(self):
        endpoint = Endpoint()
        for jobs in (2, 4):
            out = Path(self.tmp.name) / ("jobs%d" % jobs)
            with self.subTest(jobs=jobs):
                self.ms_fetch(endpoint, jobs=jobs, out=out)
                self.assertEqual((self.tensor("mtp.a", out), self.tensor("mtp.b", out)), (T.A, T.B))
                self.assertEqual(sorted((out / "tensors").glob("*.part*")), [])
                self.assertEqual(json.loads((out / "mtp-manifest.json").read_text())[0]["file"], "tensors/mtp.a.bin")

    def test_a_part_already_downloaded_is_not_fetched_again(self):
        endpoint = Endpoint()
        tdir = self.out / "tensors"
        tdir.mkdir(parents=True)
        start = T.Mirror.span("model-1.safetensors", "mtp.a")[0]
        (tdir / "mtp.a.bin.part0001").write_bytes(FILES["model-1.safetensors"][start + 16:start + 32])
        self.ms_fetch(endpoint, jobs=2)
        self.assertEqual(self.tensor("mtp.a"), T.A)
        self.assertNotIn(("model-1.safetensors", start + 16, start + 31), endpoint.ranges)

    def test_wrong_bytes_are_fetched_again_and_then_refused(self):
        err = None
        with self.assertRaisesRegex(SystemExit, "not the pinned checkpoint's"), \
                mock.patch.object(M.urllib.request, "urlopen", _WrongEndpoint()), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as e:
            src = MS.Source(self.out)
            try:
                MS.fetch(str(self.out), None, 2, src)
            finally:
                src.done()
        err = e.getvalue()
        self.assertIn("wrong bytes", err)                        # the whole tensor was fetched again, once


class _WrongEndpoint(Endpoint):
    """Every tensor's data range comes back flipped (#327's bad source: right size, wrong bytes): the whole
    tensor is fetched again, once, and then the tool refuses the source."""

    def __call__(self, req, timeout=None):
        q = parse_qs(urlparse(req.full_url).query)
        path, rng = q.get("FilePath", [None])[0], req.get_header("Range")
        r = super().__call__(req, timeout)
        if r.status != 206 or rng is None:
            return r
        raw = self.body(path)
        base = 8 + struct.unpack("<Q", raw[:8])[0]              # the shard's data section starts here
        if int(rng.split("=")[1].split("-")[0]) < base:
            return r                                            # a header read is left alone
        return T.Response(bytes(x ^ 0xFF for x in r.read()), r.status, r.headers)


class Verify(FetchCase):
    def test_verify_finds_a_corrupt_install_and_shares_the_stamp_file(self):
        self.ms_fetch(Endpoint())
        self.assertEqual(MS.verify(str(self.out)), [])
        stamps = json.loads((self.out / "tensors" / "verified.json").read_text())
        self.assertEqual(sorted(stamps), ["mtp.a", "mtp.b"])
        (self.out / "tensors" / "mtp.a.bin").write_bytes(bytes(len(T.A)))     # right size, wrong bytes
        self.assertEqual(MS.verify(str(self.out)), ["mtp.a"])
        (self.out / "tensors" / "mtp.b.bin").unlink()
        self.assertEqual(MS.verify(str(self.out)), ["mtp.a", "mtp.b"])
        with mock.patch.object(MS, "CHECK_HASHES", False):
            self.assertEqual(MS.verify(str(self.out)), [])                   # another revision: nothing to check


if __name__ == "__main__":
    unittest.main()