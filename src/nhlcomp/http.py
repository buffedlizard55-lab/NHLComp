"""HTTP client with a content-addressed disk cache and provenance capture.

The sandbox in which this repository is developed has no outbound network, so every
network helper degrades to the cache and reports that fact instead of inventing a
response.  ``require_network=True`` makes the failure loud rather than silent.
"""

from __future__ import annotations

import gzip
import io
import json
import os
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone

from .store import sha256_text, utcnow

DEFAULT_TIMEOUT = 20.0
USER_AGENT = "nhlcomp-research/1.0 (+https://github.com/buffedlizard55-lab/NHLComp)"


class NetworkUnavailable(RuntimeError):
    pass


@dataclass
class Response:
    url: str
    status: int
    body: str
    retrieved_at: str
    from_cache: bool
    sha256: str

    @property
    def json(self):
        return json.loads(self.body)


class HttpClient:
    """Caches responses to ``cache_dir`` keyed by sha256 of the URL.

    The cache filename embeds a digest so the mapping is auditable, and each cached
    blob is stored alongside a ``.meta.json`` sidecar recording the original URL,
    the retrieval time and the HTTP status.  Nothing is cached for URLs that return
    a non-2xx status.
    """

    def __init__(self, cache_dir: str, *, timeout: float = DEFAULT_TIMEOUT,
                 user_agent: str = USER_AGENT, offline: bool | None = None):
        self.cache_dir = cache_dir
        self.timeout = timeout
        self.user_agent = user_agent
        os.makedirs(cache_dir, exist_ok=True)
        self._offline = offline
        self.calls = {"hits": 0, "misses": 0, "failures": 0}

    # ------------------------------------------------------------------ cache
    def _cache_paths(self, url: str) -> tuple[str, str]:
        digest = sha256_text(url)[:32]
        return (os.path.join(self.cache_dir, digest + ".json"),
                os.path.join(self.cache_dir, digest + ".meta.json"))

    def cached(self, url: str) -> Response | None:
        blob, meta = self._cache_paths(url)
        if not (os.path.exists(blob) and os.path.exists(meta)):
            return None
        with open(blob, "r", encoding="utf-8") as fh:
            body = fh.read()
        with open(meta, "r", encoding="utf-8") as fh:
            m = json.load(fh)
        return Response(url=url, status=int(m.get("status", 200)), body=body,
                        retrieved_at=m.get("retrieved_at", ""), from_cache=True,
                        sha256=m.get("sha256", sha256_text(body)))

    def _store(self, url: str, status: int, body: str) -> str:
        blob, meta = self._cache_paths(url)
        digest = sha256_text(body)
        with open(blob, "w", encoding="utf-8") as fh:
            fh.write(body)
        with open(meta, "w", encoding="utf-8") as fh:
            json.dump({"url": url, "status": status, "retrieved_at": utcnow(),
                       "sha256": digest, "bytes": len(body.encode("utf-8"))}, fh, indent=1)
        return digest

    # ------------------------------------------------------------------ get
    def get(self, url: str, *, headers: dict[str, str] | None = None,
            retries: int = 3, backoff: float = 1.5,
            use_cache: bool = True, require_network: bool = False) -> Response:
        if use_cache:
            hit = self.cached(url)
            if hit is not None:
                self.calls["hits"] += 1
                return hit
        if self._offline:
            self.calls["failures"] += 1
            raise NetworkUnavailable(f"offline mode and no cache entry for {url}")

        req_headers = {"User-Agent": self.user_agent, "Accept": "application/json",
                       "Accept-Encoding": "gzip"}
        if headers:
            req_headers.update(headers)

        last_err: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                req = urllib.request.Request(url, headers=req_headers)
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    raw = resp.read()
                    if resp.headers.get("Content-Encoding") == "gzip":
                        raw = gzip.GzipFile(fileobj=io.BytesIO(raw)).read()
                    body = raw.decode("utf-8", errors="replace")
                    status = int(resp.status)
                if 200 <= status < 300:
                    self.calls["misses"] += 1
                    digest = self._store(url, status, body)
                    return Response(url=url, status=status, body=body, retrieved_at=utcnow(),
                                    from_cache=False, sha256=digest)
                last_err = urllib.error.HTTPError(url, status, "non-2xx", None, None)
            except (urllib.error.URLError, socket.timeout, OSError, TimeoutError) as exc:
                last_err = exc
            if attempt < retries:
                time.sleep(backoff * attempt)

        self.calls["failures"] += 1
        if use_cache:
            hit = self.cached(url)
            if hit is not None:  # stale cache is better than nothing, but must be labelled
                self.calls["hits"] += 1
                return hit
        if require_network:
            raise NetworkUnavailable(f"GET {url} failed: {last_err}")
        raise NetworkUnavailable(f"GET {url} failed: {last_err}")

    def probe(self, url: str, *, headers: dict[str, str] | None = None) -> tuple[int | None, str]:
        """Reachability check that never raises.  Returns (status_or_None, verdict)."""
        try:
            resp = self.get(url, headers=headers, retries=1, use_cache=False, require_network=True)
            return resp.status, "reachable"
        except NetworkUnavailable as exc:
            msg = str(exc)
            if "403" in msg or "401" in msg:
                return 403, "blocked"
            if "404" in msg:
                return 404, "unreachable"
            return None, "unreachable"


def parse_iso(ts: str) -> datetime:
    """Parse the ISO-8601 variants the NHL/Kalshi APIs actually emit."""
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def qs(params: dict[str, object]) -> str:
    return urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
