"""Job-scoped, bounded-memory state for streaming anonymization.

A streaming job still has to remember things across a whole file: which
values were replaced, which surrogate each value received, which distinct
texts a model must read and what it answered. Held in dictionaries, that is
exactly the unbounded memory streaming exists to remove - and it is raw PHI.

``JobSpill`` keeps a bounded least-recently-used working set in memory and
spills the rest to a private SQLite file for this job only:

* Lookup keys are keyed digests: keyed BLAKE2b (a MAC) under a random per-job key that
  never leaves the process. A spilled key names no value.
* Values that can hold PHI - a source text, its sanitized form - are sealed
  with AES-GCM under a random per-job data key, bound to the job, the
  namespace and the key as associated data. The data key is itself wrapped by
  the publication master key under a job-specific context
  (``publication.encrypt``), so a spill cannot be opened without it, and no
  spill is written at all when that key is not configured: the job fails
  rather than put plaintext PHI on disk.
* Namespaces that hold only digests or generated tokens store no value.
* The file lives under the artifact store's private ``.spill`` sibling, one
  directory per job, created owner-only where the platform supports it. No
  journal is kept (a crash restarts the job from the beginning and discards
  the spill), and ``close`` removes the directory with anything SQLite left.

Small files never touch disk: nothing spills until the working set is full.
Nothing here logs a value.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import shutil
import sqlite3
import stat
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

SPILL_CONTEXT = "stream-spill-v1"
DEFAULT_MEMORY_ITEMS = 50_000
KEY_BYTES = 20
NONCE_BYTES = 12
MEMORY_ITEMS_ENV_VAR = "BIOBLOCK_STREAM_MEMORY_ITEMS"


class SpillUnavailable(RuntimeError):
    """Spilling would put PHI on disk without the key to seal it."""

    error_code = "stream_spill_key_unavailable"

    def __init__(self):
        super().__init__(self.error_code)


def memory_items() -> int:
    raw = (os.getenv(MEMORY_ITEMS_ENV_VAR) or "").strip()
    if not raw:
        return DEFAULT_MEMORY_ITEMS
    if not raw.isdigit() or not 1_000 <= int(raw) <= 5_000_000:
        from services.resource_policy import CapacityConfigurationError

        raise CapacityConfigurationError("invalid_bioblock_stream_memory_items")
    return int(raw)


def spill_root() -> Path:
    from services import sanitized_artifacts

    root = sanitized_artifacts.output_directory()
    directory = root.with_name(root.name + ".spill")
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _private_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path, stat.S_IRWXU)
    except OSError:
        pass


class JobSpill:
    """The disk side of one job's working state. Opened lazily, removed by ``close``."""

    def __init__(
        self,
        job_id: str,
        owner: str = "",
        directory: Optional[Path] = None,
        capacity: Optional[int] = None,
    ):
        self.job_id = job_id
        self.owner = (owner or "stream").lower()
        self.capacity = capacity or memory_items()
        self._directory = directory
        self._hmac_key = secrets.token_bytes(32)
        self._data_key: Optional[bytes] = None
        self._db: Optional[sqlite3.Connection] = None
        self._lock = threading.RLock()
        self.peak_bytes = 0
        self.spilled_items = 0

    # -- keys and sealing ---------------------------------------------------------------------

    def key(self, namespace: str, value: str) -> bytes:
        # Keyed BLAKE2b is a MAC in its own right (no HMAC construction needed)
        # and several times faster here, where every cell is keyed.
        return hashlib.blake2b(
            value.encode("utf-8", "surrogatepass"),
            key=self._hmac_key,
            person=namespace.encode("ascii")[:16],
            digest_size=KEY_BYTES,
        ).digest()

    def _aad(self, namespace: str, key: bytes) -> bytes:
        return b"\0".join(
            (SPILL_CONTEXT.encode(), self.job_id.encode(), namespace.encode(), key)
        )

    def _key_for_sealing(self) -> bytes:
        if self._data_key is None:
            from services import stream_crypto as publication

            data_key = secrets.token_bytes(32)
            try:
                wrapped = publication.encrypt(
                    data_key,
                    self.owner,
                    hashlib.sha256(self.job_id.encode()).hexdigest(),
                    "application/octet-stream",
                    context=f"{SPILL_CONTEXT}:{self.job_id}",
                )
            except publication.PublicationError:
                raise SpillUnavailable() from None
            self._connection().execute(
                "INSERT OR REPLACE INTO meta VALUES ('data_key', ?)",
                (
                    json.dumps(
                        {
                            "key_version": wrapped["key_version"],
                            "wrapped_key": wrapped["wrapped_key"],
                        }
                    ),
                ),
            )
            self._data_key = data_key
        return self._data_key

    def seal(self, namespace: str, key: bytes, plaintext: bytes) -> bytes:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        nonce = secrets.token_bytes(NONCE_BYTES)
        return nonce + AESGCM(self._key_for_sealing()).encrypt(
            nonce, plaintext, self._aad(namespace, key)
        )

    def unseal(self, namespace: str, key: bytes, sealed: bytes) -> bytes:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        return AESGCM(self._key_for_sealing()).decrypt(
            sealed[:NONCE_BYTES], sealed[NONCE_BYTES:], self._aad(namespace, key)
        )

    # -- storage ------------------------------------------------------------------------------

    @property
    def directory(self) -> Path:
        if self._directory is None:
            self._directory = spill_root() / self.job_id
        return self._directory

    def _connection(self) -> sqlite3.Connection:
        with self._lock:
            if self._db is None:
                _private_directory(self.directory)
                path = self.directory / "spill.db"
                self._db = sqlite3.connect(
                    str(path), isolation_level=None, check_same_thread=False
                )
                try:
                    os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
                except OSError:
                    pass
                # One job, one process, one connection: holding the file lock
                # spares every statement a lock round trip (16x faster here).
                for pragma in (
                    "PRAGMA locking_mode = EXCLUSIVE",
                    "PRAGMA journal_mode = OFF",
                    "PRAGMA synchronous = OFF",
                    "PRAGMA temp_store = MEMORY",
                    "PRAGMA cache_size = -16384",
                ):
                    self._db.execute(pragma)
                self._db.execute(
                    "CREATE TABLE IF NOT EXISTS meta (name TEXT PRIMARY KEY, "
                    "value TEXT)"
                )
                self._db.execute(
                    "CREATE TABLE IF NOT EXISTS kv (ns TEXT, k BLOB, v BLOB, "
                    "PRIMARY KEY (ns, k)) WITHOUT ROWID"
                )
                self._db.execute(
                    "CREATE TABLE IF NOT EXISTS seq (ns TEXT, i INTEGER, "
                    "k BLOB, v BLOB, PRIMARY KEY (ns, i)) WITHOUT ROWID"
                )
            return self._db

    @property
    def opened(self) -> bool:
        return self._db is not None

    def write(self, namespace: str, items: List[Tuple[bytes, Optional[bytes]]]) -> None:
        with self._lock:
            db = self._connection()
            db.execute("BEGIN")
            db.executemany(
                "INSERT OR REPLACE INTO kv VALUES (?, ?, ?)",
                [(namespace, key, value) for key, value in items],
            )
            db.execute("COMMIT")
            self.spilled_items += len(items)
            self._measure()

    def read(self, namespace: str, key: bytes) -> Tuple[bool, Optional[bytes]]:
        with self._lock:
            if self._db is None:
                return False, None
            row = self._db.execute(
                "SELECT v FROM kv WHERE ns = ? AND k = ?", (namespace, key)
            ).fetchone()
        return (row is not None), (row[0] if row else None)

    def keys(self, namespace: str) -> Iterator[bytes]:
        """Every key of a namespace, read with a cursor rather than all at once."""
        if self._db is None:
            return
        with self._lock:
            cursor = self._db.execute("SELECT k FROM kv WHERE ns = ?", (namespace,))
            while True:
                rows = cursor.fetchmany(10_000)
                if not rows:
                    return
                for row in rows:
                    yield row[0]

    def append_seq(self, namespace: str, items: List[Tuple[int, bytes, bytes]]) -> None:
        with self._lock:
            db = self._connection()
            db.execute("BEGIN")
            db.executemany(
                "INSERT OR REPLACE INTO seq VALUES (?, ?, ?, ?)",
                [(namespace, index, key, value) for index, key, value in items],
            )
            db.execute("COMMIT")
            self.spilled_items += len(items)
            self._measure()

    def read_seq(
        self, namespace: str, start: int, stop: int
    ) -> List[Tuple[int, bytes, bytes]]:
        if self._db is None:
            return []
        with self._lock:
            return self._db.execute(
                "SELECT i, k, v FROM seq WHERE ns = ? AND i >= ? AND i < ? ORDER BY i",
                (namespace, start, stop),
            ).fetchall()

    def _measure(self) -> None:
        total = 0
        try:
            for path in self.directory.iterdir():
                total += path.stat().st_size
        except OSError:
            return
        self.peak_bytes = max(self.peak_bytes, total)

    def close(self) -> None:
        """Drop the working state: the database and anything SQLite left beside it."""
        with self._lock:
            if self._db is not None:
                try:
                    self._db.close()
                finally:
                    self._db = None
            self._data_key = None
            if self._directory is not None and self._directory.exists():
                shutil.rmtree(self._directory, ignore_errors=True)

    def __enter__(self) -> "JobSpill":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class SpillMap:
    """value -> JSON-serializable object, bounded in memory, the rest spilled.

    ``sealed`` namespaces encrypt what they spill; the others store generated
    tokens or counters only.
    """

    def __init__(
        self,
        spill: JobSpill,
        namespace: str,
        sealed: bool = True,
        capacity: Optional[int] = None,
    ):
        self.spill = spill
        self.namespace = namespace
        self.sealed = sealed
        self.capacity = capacity or spill.capacity
        self._memory: "OrderedDict[bytes, Any]" = OrderedDict()
        #: Evicted entries not yet written, and the keys changed since read.
        self._evicted: Dict[bytes, Any] = {}
        self._dirty: set = set()
        self.puts = 0
        #: Keys ever written to disk. Absent here means absent on disk: the
        #: read a never-spilled value would cost is skipped.
        self._written = _Bloom()

    def _encode(self, key: bytes, value: Any) -> bytes:
        raw = json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode(
            "utf-8", "surrogatepass"
        )
        return self.spill.seal(self.namespace, key, raw) if self.sealed else raw

    def _decode(self, key: bytes, raw: bytes) -> Any:
        if self.sealed:
            raw = self.spill.unseal(self.namespace, key, raw)
        return json.loads(raw.decode("utf-8", "surrogatepass"))

    def _flush_evicted(self) -> None:
        if self._evicted:
            items = [
                (key, self._encode(key, value)) for key, value in self._evicted.items()
            ]
            self._evicted = {}
            self.spill.write(self.namespace, items)
            for key, _value in items:
                self._written.add(key)
            if self._written.full:
                # Grown from what is on disk, so no written key is ever forgotten.
                grown = _Bloom(self._written.bits * 4)
                for key in self.spill.keys(self.namespace):
                    grown.add(key)
                self._written = grown

    def get_key(self, key: bytes, default: Any = None) -> Any:
        if key in self._memory:
            self._memory.move_to_end(key)
            return self._memory[key]
        if key in self._evicted:
            return self._evicted[key]
        if not self._written.may_contain(key):
            return default
        found, raw = self.spill.read(self.namespace, key)
        if not found:
            return default
        value = self._decode(key, raw)
        self._remember(key, value, dirty=False)
        return value

    def get(self, value: str, default: Any = None) -> Any:
        return self.get_key(self.spill.key(self.namespace, value), default)

    def put(self, value: str, item: Any) -> None:
        self.put_key(self.spill.key(self.namespace, value), item)

    def put_key(self, key: bytes, item: Any) -> None:
        self.puts += 1
        self._evicted.pop(key, None)
        self._remember(key, item, dirty=True)

    def _remember(self, key: bytes, item: Any, dirty: bool) -> None:
        self._memory[key] = item
        self._memory.move_to_end(key)
        if dirty:
            self._dirty.add(key)
        while len(self._memory) > self.capacity:
            old_key, old_value = self._memory.popitem(last=False)
            if old_key in self._dirty:
                self._dirty.discard(old_key)
                self._evicted[old_key] = old_value
        if len(self._evicted) >= 1_000:
            self._flush_evicted()

    def __contains__(self, value: str) -> bool:
        return self.get(value, _MISSING) is not _MISSING


_MISSING = object()


class _Bloom:
    """A Bloom filter over keyed digests. False positives only cost a read."""

    HASHES = 7

    def __init__(self, bits: int = 1 << 20):
        self.bits = bits
        self.array = bytearray(bits // 8 + 1)
        self.count = 0

    def _positions(self, key: bytes) -> Iterator[int]:
        low = int.from_bytes(key[:8], "little")
        high = int.from_bytes(key[8:16], "little") | 1
        for index in range(self.HASHES):
            yield (low + index * high) % self.bits

    def add(self, key: bytes) -> None:
        for position in self._positions(key):
            self.array[position >> 3] |= 1 << (position & 7)
        self.count += 1

    @property
    def full(self) -> bool:
        return self.count * 12 > self.bits

    def may_contain(self, key: bytes) -> bool:
        if not self.count:
            return False
        return all(
            self.array[position >> 3] & (1 << (position & 7))
            for position in self._positions(key)
        )


class SpillSet:
    """Membership of values by keyed digest. Nothing about a value is stored but its digest.

    Up to the working-set size, digests sit in a set; beyond it every digest
    goes to disk and a Bloom filter answers the (many) misses without a read.
    """

    BITS_PER_ITEM = 12
    HASHES = 7

    def __init__(self, spill: JobSpill, namespace: str, capacity: Optional[int] = None):
        self.spill = spill
        self.namespace = namespace
        self.capacity = capacity or spill.capacity
        self._memory: set = set()
        self._bloom: Optional[bytearray] = None
        self._bloom_bits = 0
        self.size = 0
        self._pending: set = set()

    def _positions(self, key: bytes) -> Iterator[int]:
        low = int.from_bytes(key[:8], "little")
        high = int.from_bytes(key[8:16], "little") | 1
        for index in range(self.HASHES):
            yield (low + index * high) % self._bloom_bits

    def _bloom_add(self, key: bytes) -> None:
        for position in self._positions(key):
            self._bloom[position >> 3] |= 1 << (position & 7)

    def _bloom_has(self, key: bytes) -> bool:
        return all(
            self._bloom[position >> 3] & (1 << (position & 7))
            for position in self._positions(key)
        )

    def _rebuild(self, expected: int) -> None:
        self._flush()
        self._bloom_bits = max(8 * 1024 * 1024, expected * self.BITS_PER_ITEM)
        self._bloom = bytearray(self._bloom_bits // 8 + 1)
        for key in self.spill.keys(self.namespace):
            self._bloom_add(key)

    def _flush(self) -> None:
        if self._pending:
            self.spill.write(self.namespace, [(key, None) for key in self._pending])
            self._pending = set()

    def add(self, value: str) -> None:
        self.add_key(self.spill.key(self.namespace, value))

    def add_key(self, key: bytes) -> None:
        if self._bloom is None:
            if key in self._memory:
                return
            self._memory.add(key)
            self.size += 1
            if len(self._memory) > self.capacity:
                self.spill.write(
                    self.namespace, [(item, None) for item in self._memory]
                )
                self._memory = set()
                self._rebuild(self.size * 4)
            return
        if self.contains_key(key):
            return
        self.size += 1
        if self.size * self.BITS_PER_ITEM > self._bloom_bits * 2:
            self._pending.add(key)
            self._rebuild(self.size * 4)
            return
        self._bloom_add(key)
        self._pending.add(key)
        if len(self._pending) >= 5_000:
            self._flush()

    def contains_key(self, key: bytes) -> bool:
        if self._bloom is None:
            return key in self._memory
        if not self._bloom_has(key):
            return False
        if key in self._pending:
            return True
        found, _value = self.spill.read(self.namespace, key)
        return found

    def __contains__(self, value: str) -> bool:
        return self.contains_key(self.spill.key(self.namespace, value))

    @property
    def memory_bytes(self) -> int:
        return len(self._bloom or b"") + len(self._memory) * 100


class SpillQueue:
    """Distinct values in first-seen order, each with a small mutable record.

    Values are sealed when spilled. Iteration is in insertion order, in
    bounded slices, so a model stage reads its work a unit at a time.
    """

    def __init__(self, spill: JobSpill, namespace: str, capacity: Optional[int] = None):
        self.spill = spill
        self.namespace = namespace
        self.capacity = capacity or spill.capacity
        self._index = SpillMap(
            spill, namespace + ".index", sealed=False, capacity=self.capacity
        )
        self._memory: Dict[int, Tuple[bytes, str]] = {}
        self.length = 0

    def add(self, value: str) -> Tuple[int, bool]:
        """(position, new) for a value; a value already queued keeps its place."""
        key = self.spill.key(self.namespace, value)
        found = self._index.get_key(key)
        if found is not None:
            return int(found), False
        position = self.length
        self._index.put_key(key, position)
        self._memory[position] = (key, value)
        self.length += 1
        if len(self._memory) >= self.capacity:
            self._flush()
        return position, True

    def position(self, value: str) -> Optional[int]:
        found = self._index.get_key(self.spill.key(self.namespace, value))
        return None if found is None else int(found)

    def _flush(self) -> None:
        if self._memory:
            items = [
                (
                    position,
                    key,
                    self.spill.seal(
                        self.namespace, key, value.encode("utf-8", "surrogatepass")
                    ),
                )
                for position, (key, value) in self._memory.items()
            ]
            self.spill.append_seq(self.namespace, items)
            self._memory = {}

    def slices(self, size: int) -> Iterator[List[str]]:
        """Every queued value, in order, ``size`` at a time."""
        for start in range(0, self.length, size):
            stop = min(self.length, start + size)
            values: Dict[int, str] = {}
            for position, key, sealed in self.spill.read_seq(
                self.namespace, start, stop
            ):
                values[position] = self.spill.unseal(
                    self.namespace, key, sealed
                ).decode("utf-8", "surrogatepass")
            for position in range(start, stop):
                if position in self._memory:
                    values[position] = self._memory[position][1]
            yield [values[position] for position in range(start, stop)]


class StreamingAllocator:
    """The dataset-local surrogate allocator, with its mapping in a spill.

    Same numbering as ``surrogates.SurrogateAllocator``: first-seen order per
    family, skipping numbers already present in the input. The mapping key is
    a keyed digest of (family, normalized value); the stored value is the
    generated token, so nothing spilled here is PHI.
    """

    MAX_RESERVED = 200_000

    def __init__(self, spill: JobSpill):
        self._assigned = SpillMap(spill, "allocator", sealed=False)
        self._next: Dict[str, int] = {}
        self._reserved: Dict[str, set] = {}
        self._counts: Dict[str, int] = {}
        self.reserved_overflow = False
        self._lock = threading.Lock()

    def reserve_existing(self, text: str) -> None:
        from services.surrogates import SURROGATE_PATTERN

        if not text or "<" not in text:
            return
        with self._lock:
            for match in SURROGATE_PATTERN.finditer(text):
                numbers = self._reserved.setdefault(match.group(1), set())
                if (
                    sum(len(item) for item in self._reserved.values())
                    >= self.MAX_RESERVED
                ):
                    self.reserved_overflow = True
                    return
                numbers.add(int(match.group(2)))

    def surrogate_for(self, entity_type: str, value: str) -> Optional[str]:
        from services.surrogates import SURROGATE_FAMILIES, normalize_value

        family = SURROGATE_FAMILIES.get(entity_type)
        if family is None:
            return None
        normalized = normalize_value(entity_type, value)
        if not normalized:
            return None
        name = family + "\0" + normalized
        with self._lock:
            existing = self._assigned.get(name)
            if existing is not None:
                return existing
            reserved = self._reserved.get(family, ())
            number = self._next.get(family, 1)
            while number in reserved:
                number += 1
            self._next[family] = number + 1
            token = f"<{family}_{number}>"
            self._assigned.put(name, token)
            self._counts[family] = self._counts.get(family, 0) + 1
            return token

    def is_emitted(self, token: str) -> bool:
        from services.surrogates import SURROGATE_PATTERN

        match = SURROGATE_PATTERN.fullmatch(token)
        if not match:
            return False
        family, number = match.group(1), int(match.group(2))
        return number < self._next.get(family, 1) and number not in self._reserved.get(
            family, ()
        )

    def counts(self) -> Dict[str, int]:
        return dict(sorted(self._counts.items()))

    def distinct_entities(self) -> int:
        return sum(self._counts.values())


class EmittedLedger:
    """What this transformation wrote: allocator tokens, and every other replacement.

    A membership test only, which is all ``generated_token_provenance`` asks.
    """

    def __init__(self, spill: JobSpill, allocator: StreamingAllocator):
        self._allocator = allocator
        self._other = SpillSet(spill, "emitted")

    def add(self, token: str) -> None:
        if not self._allocator.is_emitted(token):
            self._other.add(token)

    def __contains__(self, token: object) -> bool:
        return isinstance(token, str) and (
            self._allocator.is_emitted(token) or token in self._other
        )


def cleanup_orphans(active: Callable[[str], bool]) -> int:
    """Remove spill directories of jobs that are no longer running. Returns how many."""
    removed = 0
    try:
        entries = list(spill_root().iterdir())
    except OSError:
        return 0
    for entry in entries:
        if entry.is_dir() and not active(entry.name):
            shutil.rmtree(entry, ignore_errors=True)
            removed += 1
    return removed
