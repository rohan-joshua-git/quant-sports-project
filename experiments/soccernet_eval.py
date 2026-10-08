"""SoccerNet-Tracking proxy check for the Stage 1 tracker.

Scores the tracker against SoccerNet-Tracking (SN-Tracking-2023 on Hugging Face,
ungated; the paper publishes test ground truth "so that researchers can benchmark
their results locally"). Different league and footage from DFL, so this is a proxy
for the design, not a measure of error on the project's own clip. Run from the
project root:

    python experiments/soccernet_eval.py fetch      # pick sequences (fixed seed), download only those
    python experiments/soccernet_eval.py detect     # YOLO once per frame, cached
    python experiments/soccernet_eval.py track      # every tracker variant over the cached detections
    python experiments/soccernet_eval.py score      # HOTA, ball metrics, ID switches by type

Train sequences are for tuning; test sequences are for validation only, scored once
against acceptance rules written into decision_log.md beforehand. Everything goes to
experiments/soccernet/, which is gitignored (derived frames, no redistribution).
"""
import argparse
import configparser
import io
import json
import struct
import sys
import time
import urllib.request
import zipfile
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT = ROOT / "experiments" / "soccernet"
DATA = OUT / "data"
SELECTION = OUT / "selection.json"
URL = "https://huggingface.co/datasets/SoccerNet/SN-Tracking-2023/resolve/main/{}.zip"
SEED = 20261009
N_SEQ = {"train": 6, "test": 12}


def load(path, default):
    return json.loads(path.read_text()) if path.exists() else default


def save(path, data):
    path.write_text(json.dumps(data, indent=1))


# --------------------------------------------------------------------------- fetch

class HttpFile(io.RawIOBase):
    """Read-only, seekable view of a remote file through HTTP range requests.

    zipfile only needs the central directory plus the members asked for, so a few
    sequences come out of an 8.7 GB archive without downloading the rest.
    """

    def __init__(self, url, block=1 << 20):
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req) as r:
            self.url = r.geturl()          # follow the CDN redirect once
            self.size = int(r.headers["Content-Length"])
        self.pos = 0
        self.block = block
        self.cache = {}

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def _get(self, start, end):
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end - 1}"})
        for attempt in range(5):
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    return r.read()
            except OSError:
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt)

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        if n > self.block:                  # big member data: one direct request
            data = self._get(self.pos, self.pos + n)
        else:                               # small reads (headers): cached blocks
            data = b""
            while len(data) < n:
                b0 = (self.pos + len(data)) // self.block
                if b0 not in self.cache:
                    s = b0 * self.block
                    self.cache[b0] = self._get(s, min(s + self.block, self.size))
                    if len(self.cache) > 64:          # keep memory bounded
                        self.cache.pop(next(iter(self.cache)))
                off = self.pos + len(data) - b0 * self.block
                data += self.cache[b0][off:off + n - len(data)]
        self.pos += len(data)
        return data

    def readinto(self, b):
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)


def fetch():
    DATA.mkdir(parents=True, exist_ok=True)
    selection = load(SELECTION, None)
    zips = {split: zipfile.ZipFile(HttpFile(URL.format(split))) for split in N_SEQ}

    if selection is None:
        # Chosen by seed from the full list before any frame is looked at.
        rng = np.random.default_rng(SEED)
        selection = {}
        for split, z in zips.items():
            seqs = sorted({n.split("/")[1] for n in z.namelist()
                           if n.count("/") >= 2 and n.split("/")[1]})
            pick = rng.choice(len(seqs), size=N_SEQ[split], replace=False)
            selection[split] = sorted(seqs[i] for i in pick)
            print(f"{split}: {len(seqs)} sequences, picked {selection[split]}")
        save(SELECTION, selection)

    for split, seqs in selection.items():
        z = zips[split]
        http = z.fp
        for seq in seqs:
            infos = [i for i in z.infolist()
                     if i.filename.startswith(f"{split}/{seq}/") and not i.is_dir()]
            todo = [i for i in infos if not (DATA / i.filename).exists()]
            if not todo:
                print(f"{split}/{seq}: complete", flush=True)
                continue
            # A sequence's members sit next to each other in the archive: fetch the
            # whole span in one request (many small requests ran ~10x slower).
            t0 = time.time()
            start = min(i.header_offset for i in todo)
            end = max(i.header_offset + 30 + len(i.orig_filename.encode()) + 1024 + i.compress_size
                      for i in todo)
            buf = http._get(start, min(end, http.size))
            for i in todo:
                h = i.header_offset - start
                assert buf[h:h + 4] == b"PK\x03\x04", f"bad local header for {i.filename}"
                name_len, extra_len = struct.unpack("<HH", buf[h + 26:h + 30])
                d = h + 30 + name_len + extra_len
                raw = buf[d:d + i.compress_size]
                if i.compress_type == zipfile.ZIP_DEFLATED:
                    data = zlib.decompress(raw, -15)
                elif i.compress_type == zipfile.ZIP_STORED:
                    data = raw
                else:
                    raise ValueError(f"unsupported compression {i.compress_type}")
                assert zlib.crc32(data) == i.CRC, f"CRC mismatch for {i.filename}"
                dest = DATA / i.filename
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(data)
            print(f"{split}/{seq}: {len(todo)} files, {len(buf) / 1e6:.0f} MB "
                  f"in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["fetch"])
    step = parser.parse_args().step
    if step == "fetch":
        fetch()
