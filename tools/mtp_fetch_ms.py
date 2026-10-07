"""tools/mtp_fetch_ms.py - the MTP block from the checkpoint as ModelScope serves it.

tools/mtp_fetch.py reads the block from huggingface.co with HTTP range requests.  The same checkpoint is
mirrored on ModelScope (modelscope.cn), which from China is usually several MB/s with no proxy at all,
and whose file API carries a per-file sha256.  This is the same tool against those endpoints: it writes
exactly what tools/mtp_fetch.py writes (one raw file per tensor, mtp-inventory.json, mtp-manifest.json),
so tools/mtp_pack.py and tools/mtp_rt.py read its output unchanged.

    python tools/mtp_fetch_ms.py probe --out DIR                    # endpoint, revision, does it honour Range?
    python tools/mtp_fetch_ms.py inventory --out DIR                # headers only (a few KB per shard)
    python tools/mtp_fetch_ms.py fetch --out DIR [--only SUBSTR] [--jobs N]
    python tools/mtp_fetch_ms.py verify --out DIR                   # offline; exit 3: missing or corrupt tensors

Environment:
    MODELSCOPE_MTP_REPO=Qwen/Qwen3.8-Flash-Next    the repository id (default)
    MODELSCOPE_MTP_REVISION=master                 a branch, tag or commit of that repository (default)
    MODELSCOPE_ENDPOINT=https://modelscope.cn      another ModelScope deployment or mirror (default)
    MTP_MS_SCHEME=api|resolve                      how a file's URL is built (default api:
                                                   /api/v1/models/<id>/repo?Revision=&FilePath=)

#327 still holds, and is why this refuses a sloppy endpoint rather than guessing: a range read needs a
206 whose Content-Range is the range asked for, and every tensor is checked against mtp_fetch.SHA256 -
the pinned checkpoint's *bytes*, so the check is about the tensors, not about which service sent them
(that is also why it applies here at all: ModelScope has no equivalent of the pinned HF commit).  If the
endpoint ignores Range, --fallback-full-shard downloads the whole shard and reads the ranges out of it
locally - GBs per shard, a last resort, and each shard is checked against ModelScope's own sha256 first.

`fetch` is resumable: a tensor is kept whole in one file (--jobs 1, the default, appends to it and
resumes at its size), and with --jobs > 1 it is downloaded as numbered parts that survive an interrupt
and are joined once they are all there (--jobs 4 holds up to 4 x 64 MiB in RAM).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
import mtp_fetch as H  # noqa: E402  the range discipline, the retries, the pinned SHA256, the exit code

REPO_ID = (os.environ.get("MODELSCOPE_MTP_REPO") or "Qwen/Qwen3.8-Flash-Next").strip("/")
REVISION = (os.environ.get("MODELSCOPE_MTP_REVISION") or "master").strip()
ENDPOINT = (os.environ.get("MODELSCOPE_ENDPOINT") or "https://modelscope.cn").strip().rstrip("/")
SCHEME = (os.environ.get("MTP_MS_SCHEME") or "api").strip().lower()
API = ENDPOINT + "/api/v1/models/" + REPO_ID
REPO = "modelscope://%s@%s" % (REPO_ID, REVISION)       # what mtp-inventory.json records as its source
CHECK_HASHES = not os.environ.get("MTP_MS_NO_SHA256")
CHUNK = 64 << 20                                            # the range size of tools/mtp_fetch.py


def file_url(path, revision=None):
    """Where ModelScope serves one file of the repository."""
    rev = quote(revision or REVISION, safe="")
    if SCHEME == "resolve":
        return "%s/models/%s/resolve/%s/%s" % (ENDPOINT, REPO_ID, rev, quote(path, safe="/"))
    return "%s/repo?Revision=%s&FilePath=%s" % (API, rev, quote(path, safe=""))


def api_files():
    """path -> (size, sha256) from ModelScope's file listing; {} when the listing is not available."""
    try:
        raw = H.get("%s/repo/files?Revision=%s&Recursive=True" % (API, quote(REVISION, safe="")))
        files = json.loads(raw)["Data"]["Files"]
    except Exception as e:
        print("ModelScope's file listing is unavailable (%s): its per-file sha256 is not cross-checked" % e,
              file=sys.stderr)
        return {}
    out = {}
    for f in files:
        if f.get("Path") and f.get("Type") in (None, "blob"):
            out[f["Path"]] = (f.get("Size"), (f.get("Sha256") or "").lower() or None)
    return out


def download(url, dest, retries=3):
    """A whole file, streamed to dest.  Only the full-shard fallback needs this: the range path is mtp_fetch.get."""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "strata-mtp-fetch-ms"})
            with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
                for block in iter(lambda: r.read(1 << 22), b""):
                    f.write(block)
            return
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
            print("retry %s: %s" % (url, e), file=sys.stderr)


class Source:
    """The byte ranges of one ModelScope revision, plus the whole-shard fallback of a Range-ignoring endpoint."""

    def __init__(self, out, fallback=False, keep_shards=False):
        self.dir = Path(out) / "_shards"
        self.fallback, self.keep_shards = fallback, keep_shards
        self.listing = api_files() if fallback else {}
        self.lock = threading.Lock()
        self.local = {}

    def range(self, shard, a, b):
        """The bytes a..b (inclusive) of a shard: a 206 with the asked Content-Range, or the fallback."""
        try:
            return H.get(file_url(shard), a, b)
        except IOError as e:
            if not self.fallback or "not honoured" not in str(e):
                raise
            print("%s: this endpoint ignores Range (%s): reading the whole shard once" % (shard, e), file=sys.stderr)
            return self.slice(self.shard_file(shard), a, b)

    def shard_file(self, shard):
        """The shard, whole, on disk (checked against ModelScope's own size and sha256 when the listing has them)."""
        with self.lock:
            if shard in self.local:
                return self.local[shard]
            self.dir.mkdir(parents=True, exist_ok=True)
            path, (size, want) = self.dir / shard, self.listing.get(shard, (None, None))
            if not (path.exists() and size is not None and path.stat().st_size == size
                    and (want is None or H.sha256_of(path) == want)):
                download(file_url(shard), path)
                if size is not None and path.stat().st_size != size:
                    raise IOError("%s: %d bytes, ModelScope's listing says %d" % (shard, path.stat().st_size, size))
                if want is not None and H.sha256_of(path) != want:
                    os.remove(path)
                    raise IOError("%s: not ModelScope's sha256 for this file" % shard)
            self.local[shard] = path
            return path

    @staticmethod
    def slice(path, a, b):
        with open(path, "rb") as f:
            f.seek(a)
            data = f.read(b - a + 1)
        if len(data) != b - a + 1:
            raise IOError("%s: %d bytes for %d-%d" % (path, len(data), a, b))
        return data

    def done(self):
        if self.local and not self.keep_shards:
            for path in self.local.values():
                os.remove(path)
            self.local.clear()
            try:
                self.dir.rmdir()
            except OSError:
                pass


def shard_header(src, shard):
    """Safetensors: [u64 header length][JSON header][data], so two small range reads place every tensor."""
    n = struct.unpack("<Q", src.range(shard, 0, 7))[0]
    header = json.loads(src.range(shard, 8, 8 + n - 1))
    return 8 + n, header


def rows_of(src):
    """The mtp.* tensors as the pinned tool describes them: name, shard, dtype, shape, byte range."""
    index = json.loads(H.get(file_url("model.safetensors.index.json")))["weight_map"]
    mtp = {k: v for k, v in index.items() if k.startswith("mtp.")}
    rows, total = [], 0
    for shard in sorted(set(mtp.values())):
        base, header = shard_header(src, shard)
        for name, meta in header.items():
            if name in mtp and mtp[name] == shard:
                a, b = meta["data_offsets"]
                rows.append(dict(name=name, shard=shard, dtype=meta["dtype"], shape=meta["shape"],
                                 start=base + a, end=base + b - 1, bytes=b - a))
                total += b - a
    missing = sorted(set(mtp) - {r["name"] for r in rows})
    if missing:
        sys.exit("tensors named in the index but absent from their shard headers: %s" % missing)
    rows.sort(key=lambda r: r["name"])
    return rows, total


def inventory(out, src):
    """The same inventory files tools/mtp_fetch.py writes (mtp-inventory.json and .md), from ModelScope."""
    rows, total = rows_of(src)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "mtp-inventory.json"), "w", encoding="utf-8") as f:
        json.dump(dict(repo=REPO, endpoint=ENDPOINT, revision=REVISION, total_bytes=total, tensors=rows), f, indent=1)
    lines = ["# MTP block in the BF16 checkpoint", "",
             "%d tensors, %.3f GB, in %d shards." % (len(rows), total / 1e9, len(set(r["shard"] for r in rows))), "",
             "| tensor | dtype | shape | MB |", "|---|---|---|---:|"]
    for r in rows:
        lines.append("| `%s` | %s | %s | %.1f |" % (r["name"], r["dtype"], "x".join(map(str, r["shape"])), r["bytes"] / 1e6))
    text = "\n".join(lines) + "\n"
    with open(os.path.join(out, "mtp-inventory.md"), "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    return rows


def serial(src, r, path, want):
    """The algorithm of mtp_fetch.fetch: append to one file, resume at its size, sha256, one whole retry."""
    for attempt in range(2):
        have = os.path.getsize(path) if os.path.exists(path) else 0
        if have > r["bytes"]:
            os.remove(path)
            have = 0
        with open(path, "ab") as f:
            pos = r["start"] + have
            while pos <= r["end"]:
                end = min(pos + CHUNK - 1, r["end"])
                f.write(src.range(r["shard"], pos, end))
                pos = end + 1
                print("%s %.0f%%" % (r["name"], 100 * (pos - r["start"]) / r["bytes"]), file=sys.stderr)
        if os.path.getsize(path) != r["bytes"]:
            sys.exit("%s: size %d != %d" % (path, os.path.getsize(path), r["bytes"]))
        digest = H.sha256_of(path)
        if want is None or digest == want:
            return digest
        os.remove(path)
        if attempt:
            break
        print("%s: wrong bytes (sha256 %s...), fetching it again" % (r["name"], digest[:12]), file=sys.stderr)
    sys.exit("%s: sha256 %s is not the pinned checkpoint's %s - this ModelScope revision does not carry the "
             "checkpoint's bytes (try another revision, or --no-sha256 to accept it)" % (r["name"], digest, want))


def parallel(src, r, path, jobs, want):
    """--jobs: the tensor as numbered 64 MiB parts, downloaded together, joined once, sha256 then (twice at most).
    The parts survive an interrupt, so a rerun downloads only what is missing."""
    n = (r["bytes"] + CHUNK - 1) // CHUNK
    parts = ["%s.part%04d" % (path, i) for i in range(n)]
    lock, done = threading.Lock(), [0]

    def one(i):
        a, b = r["start"] + i * CHUNK, min(r["start"] + (i + 1) * CHUNK - 1, r["end"])
        if not (os.path.exists(parts[i]) and os.path.getsize(parts[i]) == b - a + 1):
            data = src.range(r["shard"], a, b)
            if len(data) != b - a + 1:
                raise IOError("%s: %d bytes for %d-%d" % (r["shard"], len(data), a, b))
            with open(parts[i], "wb") as f:
                f.write(data)
        with lock:
            done[0] += 1
            print("%s %d/%d" % (r["name"], done[0], n), file=sys.stderr)

    for attempt in range(2):
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            for _ in pool.map(one, range(n)):
                pass
        h = hashlib.sha256()
        with open(path, "wb") as f:
            for p in parts:
                with open(p, "rb") as g:
                    for block in iter(lambda: g.read(1 << 24), b""):
                        h.update(block)
                        f.write(block)
        digest = h.hexdigest()
        if want is None or digest == want:
            for p in parts:
                os.remove(p)
            return digest
        for p in parts + [path]:
            if os.path.exists(p):
                os.remove(p)
        if attempt:
            break
        done[0] = 0
        print("%s: wrong bytes (sha256 %s...), fetching it again" % (r["name"], digest[:12]), file=sys.stderr)
    sys.exit("%s: sha256 %s is not the pinned checkpoint's %s - this ModelScope revision does not carry the "
             "checkpoint's bytes (try another revision, or --no-sha256 to accept it)" % (r["name"], digest, want))


def fetch(out, only, jobs, src):
    """Every mtp.* tensor into out/tensors plus out/mtp-manifest.json - the input tools/mtp_pack.py expects."""
    inv_path = os.path.join(out, "mtp-inventory.json")
    inv = None
    if os.path.exists(inv_path):
        try:
            with open(inv_path, encoding="utf-8") as f:
                inv = json.load(f)
        except ValueError:
            inv = None
    # The byte ranges belong to one revision: an inventory of another repository or revision is read again.
    same = isinstance(inv, dict) and inv.get("repo") == REPO
    rows = inv["tensors"] if same else inventory(out, src)
    tdir = os.path.join(out, "tensors")
    os.makedirs(tdir, exist_ok=True)
    if CHECK_HASHES:
        extra = sorted({r["name"] for r in rows} - set(H.SHA256))
        gone = sorted(set(H.SHA256) - {r["name"] for r in rows})
        if extra:
            print("not in the pinned checkpoint's hash table, they are taken as they come: %s" % extra, file=sys.stderr)
        if gone:
            print("%s is missing from this revision: %s" % (REPO, gone), file=sys.stderr)
    manifest = []
    for r in rows:
        if only and only not in r["name"]:
            continue
        path = os.path.join(tdir, r["name"] + ".bin")
        want = H.SHA256.get(r["name"]) if CHECK_HASHES else None
        if not same and os.path.exists(path) and (
                want is None or os.path.getsize(path) != r["bytes"] or H.sha256_of(path) != want):
            os.remove(path)                         # the bytes of another revision: not resumed
        digest = None
        if os.path.exists(path) and os.path.getsize(path) == r["bytes"]:
            digest = H.sha256_of(path)
            if want is None or digest == want:
                print("%s: already fetched, kept" % r["name"], file=sys.stderr)
        if digest is None:
            digest = parallel(src, r, path, jobs, want) if jobs > 1 else serial(src, r, path, want)
        # a forward slash, so a manifest written on Windows is read by tools/mtp_pack.py anywhere
        manifest.append(dict(r, file=os.path.relpath(path, out).replace(os.sep, "/"), sha256=digest))
    with open(os.path.join(out, "mtp-manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    print("%d tensors, %.2f GB -> %s" % (len(manifest), sum(r["bytes"] for r in manifest) / 1e9, os.path.join(out, "tensors")))
    return manifest


def verify(out):
    """#327 offline: the fetched tensors against mtp_fetch.SHA256 -> the names that are missing or wrong.  [] when
    they are right, when there is nothing to check (no tensors/, or --no-sha256) or when the plain tool would not
    check them (another STRATA_MTP_REVISION).  A verdict is kept in tensors/verified.json by size and mtime, so a
    later run hashes only what changed - this writes the same stamp file, so the two tools share it."""
    tdir = os.path.join(out, "tensors")
    if not CHECK_HASHES or not os.path.isdir(tdir):
        return []
    stamp_path = os.path.join(tdir, "verified.json")
    try:
        with open(stamp_path, encoding="utf-8") as f:
            stamps = json.load(f)
    except (OSError, ValueError):
        stamps = {}
    bad, good = [], {}
    for name, want in sorted(H.SHA256.items()):
        path = os.path.join(tdir, name + ".bin")
        if not os.path.exists(path):
            bad.append(name)
            continue
        st = os.stat(path)
        key = [st.st_size, st.st_mtime_ns, want]
        if stamps.get(name) != key and H.sha256_of(path) != want:
            bad.append(name)
            continue
        good[name] = key
    try:
        with open(stamp_path, "w", encoding="utf-8") as f:
            json.dump(good, f, indent=1)
    except OSError:
        pass
    return bad


def probe(src):
    """Is this endpoint usable?  Its listing, the pinned index and a real range read of one shard's header."""
    print("ModelScope endpoint %s, repo %s, revision %s, scheme %s" % (ENDPOINT, REPO_ID, REVISION, SCHEME))
    if src.listing:
        print("the file listing has %d files" % len(src.listing))
    try:
        rows, total = rows_of(src)
    except Exception as e:
        sys.exit("cannot read the checkpoint's headers from %s (%s): is it mirrored there under this revision?"
                 % (REPO, e))
    print("%d mtp tensors, %.3f GB, in %d shards; range reads answer 206 as they must"
          % (len(rows), total / 1e9, len(set(r["shard"] for r in rows))))
    print("next: python %s inventory --out DIR" % Path(__file__).name)


def main():
    global CHECK_HASHES
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["probe", "inventory", "fetch", "verify"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", help="fetch: only the tensors whose name contains this (the manifest is rewritten: "
                                   "run fetch again without it before tools/mtp_pack.py)")
    ap.add_argument("--jobs", type=int, default=1, help="fetch: chunks downloaded together (default 1; 4-8 saturates "
                                                        "a fast link, each job holds up to 64 MiB)")
    ap.add_argument("--fallback-full-shard", action="store_true",
                    help="if the endpoint ignores Range, download the whole shard (GBs) and slice it locally")
    ap.add_argument("--keep-shards", action="store_true", help="keep the shards that fallback downloaded")
    ap.add_argument("--no-sha256", action="store_true",
                    help="do not check the tensors against the pinned checkpoint's sha256 (another revision)")
    a = ap.parse_args()
    CHECK_HASHES = CHECK_HASHES and not a.no_sha256
    os.makedirs(a.out, exist_ok=True)
    if a.cmd == "verify":                           # offline: nothing here opens a socket
        bad = verify(a.out)
        for name in bad:
            print("MTP tensor missing or corrupt: %s" % name, file=sys.stderr)
        sys.exit(H.BAD if bad else 0)
    src = Source(a.out, a.fallback_full_shard, a.keep_shards)
    try:
        if a.cmd == "probe":
            probe(src)
        elif a.cmd == "inventory":
            inventory(a.out, src)
        else:
            fetch(a.out, a.only, max(1, a.jobs), src)
    finally:
        src.done()


if __name__ == "__main__":
    main()