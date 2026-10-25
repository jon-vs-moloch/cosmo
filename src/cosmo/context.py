"""Repository-aware inspection and role-specific context compilation.

Read-only access is always relative to an explicitly approved ``repo_root``.
Retrievals are bounded, refuse path/symlink escapes, and can be recorded into
immutable context manifests under the existing AtomicStore. Prompt transcripts
are never persisted here.
"""

from __future__ import annotations

import ast
import hashlib
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

from .atomic import (
    AtomicHarnessError,
    AtomicStore,
    _assert_state_root_disjoint,
    _safe_record_identifier,
    _safe_relative_path,
    _sha256_bytes,
    _sha256_text,
    new_id,
    utc_now,
)
from .reporting import canonical_hash


class ContextError(AtomicHarnessError):
    """Context inspection or compilation failed an invariant."""


# Directories and path-name noise that must never appear in repository maps.
DEFAULT_EXCLUDE_DIR_NAMES = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".tox",
        ".nox",
        ".venv",
        "venv",
        "node_modules",
        ".eggs",
        "dist",
        "build",
        ".idea",
        ".vscode",
        ".DS_Store",
    }
)

DEFAULT_EXCLUDE_FILE_SUFFIXES = (
    ".pyc",
    ".pyo",
    ".so",
    ".dylib",
    ".o",
    ".a",
    ".class",
    ".whl",
    ".egg",
)

LANGUAGE_BY_SUFFIX: dict[str, str] = {
    ".py": "python",
    ".pyi": "python",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".json": "json",
    ".md": "markdown",
    ".rst": "rst",
    ".toml": "toml",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".html": "html",
    ".css": "css",
    ".rs": "rust",
    ".go": "go",
    ".java": "java",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".sh": "shell",
    ".bash": "shell",
    ".zsh": "shell",
    ".txt": "text",
}

DEFAULT_MAX_MAP_ENTRIES = 2000
DEFAULT_MAX_FILE_BYTES = 256 * 1024
DEFAULT_MAX_READ_BYTES = 64 * 1024
DEFAULT_MAX_SEARCH_MATCHES = 50
DEFAULT_MAX_SEARCH_FILE_BYTES = 128 * 1024
DEFAULT_MAX_SYMBOL_FILE_BYTES = 128 * 1024
DEFAULT_MAX_EXCERPT_CHARS = 4000
_HASH_CHUNK_BYTES = 1024 * 1024


def language_hint(relative_path: str) -> str | None:
    suffix = Path(relative_path).suffix.lower()
    return LANGUAGE_BY_SUFFIX.get(suffix)


@dataclass(frozen=True)
class FileMapEntry:
    path: str
    sha256: str
    size_bytes: int
    language: str | None = None
    symbols: list[str] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "language": self.language,
            "symbols": list(self.symbols),
            "imports": list(self.imports),
        }


@dataclass(frozen=True)
class RepositoryMap:
    """Deterministic, JSON-serializable repository inventory."""

    schema_version: str
    repo_root_hash: str
    files: tuple[FileMapEntry, ...]
    excluded_prefixes: tuple[str, ...]
    truncated: bool
    entry_limit: int
    file_byte_limit: int
    map_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "repo_root_hash": self.repo_root_hash,
            "files": [entry.to_dict() for entry in self.files],
            "excluded_prefixes": list(self.excluded_prefixes),
            "truncated": self.truncated,
            "entry_limit": self.entry_limit,
            "file_byte_limit": self.file_byte_limit,
            "map_sha256": self.map_sha256,
            "file_count": len(self.files),
        }


class RepositoryInspector:
    """Bounded, escape-safe read-only view over an approved repository root."""

    def __init__(
        self,
        repo_root: Path,
        *,
        state_root: Path | None = None,
        exclude_dir_names: Iterable[str] = DEFAULT_EXCLUDE_DIR_NAMES,
        max_map_entries: int = DEFAULT_MAX_MAP_ENTRIES,
        max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
        max_read_bytes: int = DEFAULT_MAX_READ_BYTES,
        max_search_matches: int = DEFAULT_MAX_SEARCH_MATCHES,
        max_search_file_bytes: int = DEFAULT_MAX_SEARCH_FILE_BYTES,
        max_symbol_file_bytes: int = DEFAULT_MAX_SYMBOL_FILE_BYTES,
    ):
        self.repo_root = Path(repo_root).resolve()
        if not self.repo_root.is_dir():
            raise ContextError(f"repo_root is not a directory: {self.repo_root}")
        self.state_root = Path(state_root).resolve() if state_root is not None else None
        if self.state_root is not None:
            try:
                _assert_state_root_disjoint(self.repo_root, self.state_root)
            except AtomicHarnessError as exc:
                raise ContextError(str(exc)) from exc
        self.exclude_dir_names = frozenset(str(name) for name in exclude_dir_names)
        self.max_map_entries = int(max_map_entries)
        self.max_file_bytes = int(max_file_bytes)
        self.max_read_bytes = int(max_read_bytes)
        self.max_search_matches = int(max_search_matches)
        self.max_search_file_bytes = int(max_search_file_bytes)
        self.max_symbol_file_bytes = int(max_symbol_file_bytes)
        if self.max_map_entries <= 0:
            raise ContextError("max_map_entries must be positive")
        if self.max_file_bytes <= 0 or self.max_read_bytes <= 0:
            raise ContextError("byte limits must be positive")
        if self.max_search_matches <= 0:
            raise ContextError("max_search_matches must be positive")
        if self.max_search_file_bytes <= 0 or self.max_symbol_file_bytes <= 0:
            raise ContextError("byte limits must be positive")

    # ------------------------------------------------------------------
    # Path safety
    # ------------------------------------------------------------------

    def safe_relative(self, relative_path: str) -> str:
        """Normalize and validate a relative path string (no FS resolution)."""
        try:
            return _safe_relative_path(relative_path)
        except AtomicHarnessError as exc:
            raise ContextError(str(exc)) from exc

    def resolve_relative(self, relative_path: str) -> Path:
        """Resolve a relative path under repo_root; refuse escapes and symlinks."""
        safe = self.safe_relative(relative_path)
        candidate = self.repo_root / safe
        # Refuse any symlink in the relative chain before resolution.
        cursor = self.repo_root
        for part in PurePosixPath(safe).parts:
            cursor = cursor / part
            if cursor.is_symlink():
                raise ContextError(
                    f"path contains a symlink which is not permitted: {safe}"
                )
        try:
            resolved = candidate.resolve(strict=False)
        except OSError as exc:
            raise ContextError(f"cannot resolve path: {safe}") from exc
        try:
            resolved.relative_to(self.repo_root)
        except ValueError as exc:
            raise ContextError(
                f"path escapes repo_root: {relative_path!r}"
            ) from exc
        return resolved

    def _is_under_state_root(self, path: Path) -> bool:
        if self.state_root is None:
            return False
        try:
            path.resolve().relative_to(self.state_root)
        except ValueError:
            return False
        return True

    def _should_skip_relative(self, relative: PurePosixPath) -> bool:
        parts = relative.parts
        if any(part in self.exclude_dir_names for part in parts):
            return True
        name = relative.name
        if name in self.exclude_dir_names:
            return True
        if name.startswith(".") and name not in {".env.example"}:
            # Keep ordinary dotfiles out of default inventory noise, but still
            # allow explicit reads when operators select them.
            if any(part.startswith(".") for part in parts[:-1]):
                return True
            if name in {".gitignore", ".gitattributes", ".editorconfig"}:
                return False
            if name.startswith("."):
                # Root-level env/local config noise.
                if name in {".DS_Store", ".coverage", ".env", ".env.local"}:
                    return True
        lower = name.lower()
        if any(lower.endswith(suffix) for suffix in DEFAULT_EXCLUDE_FILE_SUFFIXES):
            return True
        return False

    def _iter_candidate_files(self) -> list[Path]:
        results: list[Path] = []
        for path in sorted(self.repo_root.rglob("*"), key=lambda p: p.as_posix()):
            if not path.is_file():
                continue
            if path.is_symlink():
                continue
            try:
                relative = path.relative_to(self.repo_root)
            except ValueError:
                continue
            if self._should_skip_relative(PurePosixPath(relative.as_posix())):
                continue
            if self._is_under_state_root(path):
                continue
            # Skip any path whose resolved form escapes (hardlink/mount edge).
            try:
                resolved = path.resolve()
                resolved.relative_to(self.repo_root)
            except (OSError, ValueError):
                continue
            if self._is_under_state_root(resolved):
                continue
            results.append(path)
        return results

    @staticmethod
    def _hash_file_streaming(path: Path) -> str:
        """True content SHA-256 without loading the whole file into memory."""
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while True:
                chunk = stream.read(_HASH_CHUNK_BYTES)
                if not chunk:
                    break
                digest.update(chunk)
        return digest.hexdigest()

    # ------------------------------------------------------------------
    # Python symbol extraction
    # ------------------------------------------------------------------

    def inspect_python_symbols(
        self, relative_path: str, *, content: str | None = None
    ) -> dict[str, Any]:
        """Extract top-level symbols and imports from a Python file when safe."""
        safe = self.safe_relative(relative_path)
        if content is None:
            target = self.resolve_relative(safe)
            if not target.is_file():
                raise ContextError(f"file not found: {safe}")
            size = target.stat().st_size
            if size > self.max_symbol_file_bytes:
                return {
                    "path": safe,
                    "symbols": [],
                    "imports": [],
                    "skipped": True,
                    "reason": "file_too_large",
                    "size_bytes": size,
                }
            try:
                content = target.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                return {
                    "path": safe,
                    "symbols": [],
                    "imports": [],
                    "skipped": True,
                    "reason": "not_utf8",
                    "size_bytes": size,
                }
        if len(content.encode("utf-8")) > self.max_symbol_file_bytes:
            return {
                "path": safe,
                "symbols": [],
                "imports": [],
                "skipped": True,
                "reason": "content_too_large",
            }
        try:
            tree = ast.parse(content, filename=safe)
        except SyntaxError as exc:
            return {
                "path": safe,
                "symbols": [],
                "imports": [],
                "skipped": True,
                "reason": f"syntax_error:{exc.msg}",
            }
        symbols: list[str] = []
        imports: list[str] = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                symbols.append(node.name)
            elif isinstance(node, ast.ClassDef):
                symbols.append(node.name)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        symbols.append(target.id)
            elif isinstance(node, ast.AnnAssign) and isinstance(
                node.target, ast.Name
            ):
                symbols.append(node.target.id)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if node.names and any(alias.name == "*" for alias in node.names):
                    imports.append(f"{module}.*" if module else "*")
                else:
                    for alias in node.names:
                        imports.append(
                            f"{module}.{alias.name}" if module else alias.name
                        )
        return {
            "path": safe,
            "symbols": sorted(set(symbols)),
            "imports": sorted(set(imports)),
            "skipped": False,
        }

    # ------------------------------------------------------------------
    # Map / tree / read / search
    # ------------------------------------------------------------------

    def build_map(self) -> RepositoryMap:
        """Build a deterministic repository map with optional Python metadata."""
        files: list[FileMapEntry] = []
        truncated = False
        for path in self._iter_candidate_files():
            if len(files) >= self.max_map_entries:
                truncated = True
                break
            relative = path.relative_to(self.repo_root).as_posix()
            try:
                size = path.stat().st_size
            except OSError:
                continue
            # Always record a true content hash. Large files are hashed with a
            # bounded stream so metadata limits never invent a sentinel digest.
            if size > self.max_file_bytes:
                try:
                    digest = self._hash_file_streaming(path)
                except OSError:
                    continue
                files.append(
                    FileMapEntry(
                        path=relative,
                        sha256=digest,
                        size_bytes=size,
                        language=language_hint(relative),
                        symbols=[],
                        imports=[],
                    )
                )
                continue
            try:
                data = path.read_bytes()
            except OSError:
                continue
            digest = _sha256_bytes(data)
            lang = language_hint(relative)
            symbols: list[str] = []
            imports: list[str] = []
            if lang == "python":
                try:
                    text = data.decode("utf-8")
                except UnicodeDecodeError:
                    text = None
                if text is not None:
                    info = self.inspect_python_symbols(relative, content=text)
                    if not info.get("skipped"):
                        symbols = list(info["symbols"])
                        imports = list(info["imports"])
            files.append(
                FileMapEntry(
                    path=relative,
                    sha256=digest,
                    size_bytes=size,
                    language=lang,
                    symbols=symbols,
                    imports=imports,
                )
            )
        # Deterministic order by path.
        files.sort(key=lambda entry: entry.path)
        excluded = sorted(self.exclude_dir_names)
        if self.state_root is not None:
            try:
                state_rel = self.state_root.relative_to(self.repo_root).as_posix()
            except ValueError:
                state_rel = str(self.state_root)
            if state_rel not in excluded:
                excluded.append(state_rel)
                excluded.sort()
        body = {
            "files": [entry.to_dict() for entry in files],
            "excluded_prefixes": excluded,
            "truncated": truncated,
            "entry_limit": self.max_map_entries,
            "file_byte_limit": self.max_file_bytes,
        }
        map_sha = canonical_hash(body)
        root_hash = _sha256_text(str(self.repo_root))
        return RepositoryMap(
            schema_version="cosmo.repository-map.v1",
            repo_root_hash=root_hash,
            files=tuple(files),
            excluded_prefixes=tuple(excluded),
            truncated=truncated,
            entry_limit=self.max_map_entries,
            file_byte_limit=self.max_file_bytes,
            map_sha256=map_sha,
        )

    def list_tree(self, *, prefix: str = "", max_entries: int | None = None) -> dict[str, Any]:
        """List relative file paths under an optional prefix (read-only)."""
        # ``0`` must not fall through to the default (``or`` would treat it as missing).
        limit = self.max_map_entries if max_entries is None else int(max_entries)
        if limit <= 0:
            raise ContextError("max_entries must be positive")
        if prefix:
            safe_prefix = self.safe_relative(prefix)
            # Fail closed on unsafe prefix resolution; nonexistent prefixes
            # remain allowed and simply yield an empty listing.
            prefix_path = self.repo_root / safe_prefix
            if prefix_path.exists() or prefix_path.is_symlink():
                self.resolve_relative(safe_prefix)
        else:
            safe_prefix = ""
        entries: list[dict[str, Any]] = []
        truncated = False
        for path in self._iter_candidate_files():
            relative = path.relative_to(self.repo_root).as_posix()
            if safe_prefix and not (
                relative == safe_prefix or relative.startswith(safe_prefix + "/")
            ):
                continue
            if len(entries) >= limit:
                truncated = True
                break
            try:
                size = path.stat().st_size
            except OSError:
                size = -1
            entries.append(
                {
                    "path": relative,
                    "size_bytes": size,
                    "language": language_hint(relative),
                }
            )
        return {
            "prefix": safe_prefix,
            "entries": entries,
            "truncated": truncated,
            "entry_limit": limit,
            "result_sha256": canonical_hash(entries),
        }

    def read_artifact(
        self,
        relative_path: str,
        *,
        max_bytes: int | None = None,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> dict[str, Any]:
        """Read a bounded UTF-8 excerpt of a relative artifact."""
        safe = self.safe_relative(relative_path)
        target = self.resolve_relative(safe)
        if not target.is_file():
            raise ContextError(f"file not found: {safe}")
        if self._is_under_state_root(target):
            raise ContextError("refusing to read AtomicStore state root contents")
        # ``0`` must not fall through to the default.
        byte_limit = self.max_read_bytes if max_bytes is None else int(max_bytes)
        if byte_limit <= 0:
            raise ContextError("max_bytes must be positive")
        try:
            full_sha = self._hash_file_streaming(target)
            size_bytes = target.stat().st_size
        except OSError as exc:
            raise ContextError(f"cannot read artifact: {safe}") from exc
        # Bound the bytes loaded for the excerpt without affecting full_sha.
        with target.open("rb") as stream:
            data = stream.read(byte_limit + 1)
        truncated = False
        if len(data) > byte_limit:
            data = data[:byte_limit]
            truncated = True
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ContextError(f"artifact is not valid UTF-8: {safe}") from exc
        lines = text.splitlines(keepends=True)
        range_start = 1
        range_end = len(lines)
        if start_line is not None or end_line is not None:
            if start_line is None:
                start_line = 1
            if end_line is None:
                end_line = len(lines)
            if start_line < 1 or end_line < start_line:
                raise ContextError("invalid line range")
            # 1-based inclusive.
            excerpt_lines = lines[start_line - 1 : end_line]
            text = "".join(excerpt_lines)
            range_start = start_line
            range_end = min(end_line, start_line - 1 + len(excerpt_lines))
            if len(text.encode("utf-8")) > byte_limit:
                # Hard bound on excerpt bytes.
                encoded = text.encode("utf-8")[:byte_limit]
                text = encoded.decode("utf-8", errors="ignore")
                truncated = True
        return {
            "path": safe,
            "sha256": full_sha,
            "content_sha256": _sha256_text(text),
            "size_bytes": size_bytes,
            "excerpt": text,
            "range": {"start_line": range_start, "end_line": range_end},
            "truncated": truncated,
            "byte_limit": byte_limit,
            "language": language_hint(safe),
            "selection_method": "explicit",
        }

    def search_text(
        self,
        query: str,
        *,
        prefix: str = "",
        max_matches: int | None = None,
        case_sensitive: bool = True,
    ) -> dict[str, Any]:
        """Search file contents under the approved root for a literal query."""
        if not query:
            raise ContextError("search query must not be empty")
        # ``0`` must not fall through to the default.
        limit = (
            self.max_search_matches if max_matches is None else int(max_matches)
        )
        if limit <= 0:
            raise ContextError("max_matches must be positive")
        if prefix:
            safe_prefix = self.safe_relative(prefix)
            prefix_path = self.repo_root / safe_prefix
            if prefix_path.exists() or prefix_path.is_symlink():
                self.resolve_relative(safe_prefix)
        else:
            safe_prefix = ""
        flags = 0 if case_sensitive else re.IGNORECASE
        pattern = re.compile(re.escape(query), flags)
        matches: list[dict[str, Any]] = []
        truncated = False
        for path in self._iter_candidate_files():
            relative = path.relative_to(self.repo_root).as_posix()
            if safe_prefix and not (
                relative == safe_prefix or relative.startswith(safe_prefix + "/")
            ):
                continue
            try:
                size = path.stat().st_size
            except OSError:
                continue
            if size > self.max_search_file_bytes:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            file_digest: str | None = None
            for line_no, line in enumerate(text.splitlines(), start=1):
                if pattern.search(line):
                    if file_digest is None:
                        try:
                            file_digest = self._hash_file_streaming(path)
                        except OSError:
                            file_digest = _sha256_text(text)
                    matches.append(
                        {
                            "path": relative,
                            "line": line_no,
                            "text": line[:500],
                            "sha256": file_digest,
                        }
                    )
                    if len(matches) >= limit:
                        truncated = True
                        break
            if truncated:
                break
        return {
            "query": query,
            "prefix": safe_prefix,
            "matches": matches,
            "truncated": truncated,
            "match_limit": limit,
            "result_sha256": canonical_hash(
                [{"path": m["path"], "line": m["line"]} for m in matches]
            ),
            "selection_method": "retrieval",
        }

    def retrieval_manifest_item(
        self, method: str, payload: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Normalize a retrieval result into a context-manifest item."""
        path = payload.get("path")
        content_hash = (
            payload.get("content_sha256")
            or payload.get("sha256")
            or payload.get("result_sha256")
            or payload.get("map_sha256")
        )
        excerpt = payload.get("excerpt")
        if excerpt is None and "matches" in payload:
            excerpt = json_safe_excerpt(payload["matches"])
        range_info = payload.get("range")
        return {
            "source_path": path,
            "source_id": payload.get("source_id") or path,
            "content_hash": content_hash,
            "selection_method": method,
            "included_range_or_excerpt": (
                range_info
                if range_info is not None
                else (excerpt if excerpt is not None else None)
            ),
            "metadata": {
                key: value
                for key, value in payload.items()
                if key
                not in {
                    "excerpt",
                    "matches",
                    "content",
                    "files",
                }
            },
        }


def json_safe_excerpt(value: Any, *, limit: int = DEFAULT_MAX_EXCERPT_CHARS) -> str:
    import json

    text = json.dumps(value, sort_keys=True, ensure_ascii=False)
    if len(text) > limit:
        return text[:limit] + "…"
    return text


class ContextCompiler:
    """Compile role-specific immutable context manifests into AtomicStore."""

    ROLES = frozenset({"planning", "worker", "review", "synthesis"})

    def __init__(
        self,
        store: AtomicStore,
        *,
        max_excerpt_chars: int = DEFAULT_MAX_EXCERPT_CHARS,
    ):
        self.store = store
        self.max_excerpt_chars = int(max_excerpt_chars)

    def _clip_excerpt(self, value: Any) -> Any:
        if isinstance(value, str) and len(value) > self.max_excerpt_chars:
            return value[: self.max_excerpt_chars] + "…"
        return value

    def _normalize_item(self, item: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(item, Mapping):
            raise ContextError("context item must be a mapping")
        source_path = item.get("source_path") or item.get("logical_path")
        source_id = item.get("source_id") or item.get("id") or source_path
        content_hash = item.get("content_hash") or item.get("sha256")
        selection = item.get("selection_method") or "explicit"
        if selection not in {
            "explicit",
            "dependency",
            "retrieval",
            "ledger_projection",
            "repository_map",
            "task_spec",
            "request",
            "plan_node",
            "strategy",
            "validation",
            "candidate",
            "operator",
        }:
            raise ContextError(f"unsupported selection_method: {selection!r}")
        if content_hash is None:
            # Derive a stable hash from the durable fields when none is given.
            content_hash = canonical_hash(
                {
                    "source_path": source_path,
                    "source_id": source_id,
                    "selection_method": selection,
                    "included_range_or_excerpt": item.get(
                        "included_range_or_excerpt"
                    ),
                }
            )
        excerpt = item.get("included_range_or_excerpt")
        if isinstance(excerpt, str):
            excerpt = self._clip_excerpt(excerpt)
        return {
            "source_path": source_path,
            "source_id": source_id,
            "content_hash": content_hash,
            "selection_method": selection,
            "included_range_or_excerpt": excerpt,
            "kind": item.get("kind"),
            "metadata": dict(item.get("metadata") or {}),
        }

    def _write_manifest(
        self,
        *,
        role: str,
        items: Sequence[Mapping[str, Any]],
        excluded: Sequence[str],
        subject: Mapping[str, Any] | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if role not in self.ROLES:
            raise ContextError(f"unsupported context role: {role!r}")
        normalized = [self._normalize_item(item) for item in items]
        # Stable order: kind/source_path/source_id/content_hash
        normalized.sort(
            key=lambda item: (
                str(item.get("kind") or ""),
                str(item.get("source_path") or ""),
                str(item.get("source_id") or ""),
                str(item.get("content_hash") or ""),
            )
        )
        default_excluded = [
            "prior conversation transcripts",
            "prior attempt transcripts",
            "unrelated repository files",
            "raw logs when structured results exist",
            "stale hypotheses",
        ]
        excluded_list = list(dict.fromkeys([*default_excluded, *excluded]))
        manifest_id = new_id("CTX")
        body = {
            "schema_version": "cosmo.context-manifest.v1",
            "context_manifest_id": manifest_id,
            "role": role,
            "items": normalized,
            "excluded": excluded_list,
            "subject": dict(subject or {}),
            "created_at": utc_now(),
        }
        if extra:
            body["extra"] = dict(extra)
        body["manifest_sha256"] = canonical_hash(
            {
                "role": role,
                "items": normalized,
                "excluded": excluded_list,
                "subject": body["subject"],
            }
        )
        self.store.write("context_manifests", manifest_id, body)
        self.store.append_event(
            "context_manifest_compiled",
            {
                "context_manifest_id": manifest_id,
                "role": role,
                "item_count": len(normalized),
                "manifest_sha256": body["manifest_sha256"],
            },
        )
        return body

    def record_retrieval(
        self,
        *,
        context_manifest_id: str,
        item: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Append an on-demand retrieval event (does not rewrite the manifest).

        Manifests remain immutable. Operators compile a new manifest when a
        larger pack is required; each retrieval is still auditable via events.
        """
        context_manifest_id = _safe_record_identifier(context_manifest_id)
        self.store.read("context_manifests", context_manifest_id)
        normalized = self._normalize_item(item)
        self.store.append_event(
            "context_retrieval",
            {
                "context_manifest_id": context_manifest_id,
                "item": normalized,
            },
        )
        return normalized

    def compile_for_planning(
        self,
        *,
        request: Mapping[str, Any] | None = None,
        task: Mapping[str, Any] | None = None,
        task_spec: Mapping[str, Any] | None = None,
        plan_nodes: Sequence[Mapping[str, Any]] = (),
        ledger_projection: Mapping[str, Any] | None = None,
        repository_map: Mapping[str, Any] | RepositoryMap | None = None,
        selected_files: Sequence[Mapping[str, Any]] = (),
    ) -> dict[str, Any]:
        """Planning pack: contracts, ledger summary, map, explicit reads only."""
        items: list[dict[str, Any]] = []
        if request is not None:
            items.append(
                {
                    "kind": "request",
                    "source_id": request.get("request_id"),
                    "content_hash": request.get("immutable_sha256")
                    or canonical_hash(dict(request)),
                    "selection_method": "explicit",
                    "included_range_or_excerpt": {
                        "original_text": request.get("original_text"),
                        "desired_outcome": request.get("desired_outcome"),
                        "explicit_constraints": request.get("explicit_constraints"),
                    },
                }
            )
        if task is not None:
            items.append(
                {
                    "kind": "task",
                    "source_id": task.get("task_id"),
                    "content_hash": canonical_hash(
                        {
                            "task_id": task.get("task_id"),
                            "status": task.get("status"),
                            "current_spec_id": task.get("current_spec_id"),
                            "plan_node_id": task.get("plan_node_id"),
                        }
                    ),
                    "selection_method": "explicit",
                    "included_range_or_excerpt": {
                        "task_id": task.get("task_id"),
                        "status": task.get("status"),
                        "current_spec_id": task.get("current_spec_id"),
                        "plan_node_id": task.get("plan_node_id"),
                    },
                }
            )
        if task_spec is not None:
            items.append(
                {
                    "kind": "task_spec",
                    "source_id": task_spec.get("spec_id"),
                    "content_hash": canonical_hash(dict(task_spec)),
                    "selection_method": "task_spec",
                    "included_range_or_excerpt": {
                        "objective": task_spec.get("objective"),
                        "requirements": task_spec.get("requirements"),
                        "acceptance_criteria": task_spec.get("acceptance_criteria"),
                        "constraints": task_spec.get("constraints"),
                        "artifact_path": task_spec.get("artifact_path"),
                        "validators": task_spec.get("validators"),
                    },
                }
            )
        for node in plan_nodes:
            items.append(
                {
                    "kind": "plan_node",
                    "source_id": node.get("node_id"),
                    "content_hash": canonical_hash(dict(node)),
                    "selection_method": "plan_node",
                    "included_range_or_excerpt": {
                        "node_id": node.get("node_id"),
                        "kind": node.get("kind"),
                        "objective": node.get("objective"),
                        "writable_artifact": node.get("writable_artifact"),
                        "dependencies": node.get("dependencies"),
                        "parent_node_id": node.get("parent_node_id"),
                        "status": node.get("status"),
                    },
                }
            )
        if ledger_projection is not None:
            items.append(
                {
                    "kind": "ledger_projection",
                    "source_id": ledger_projection.get("projection_id"),
                    "content_hash": canonical_hash(dict(ledger_projection)),
                    "selection_method": "ledger_projection",
                    "included_range_or_excerpt": {
                        "observations": ledger_projection.get("observations"),
                        "hypotheses": ledger_projection.get("hypotheses"),
                        "retained_successes": ledger_projection.get(
                            "retained_successes"
                        ),
                        "rejected_strategies": ledger_projection.get(
                            "rejected_strategies"
                        ),
                        "invalidated_assumptions": ledger_projection.get(
                            "invalidated_assumptions"
                        ),
                    },
                }
            )
        if repository_map is not None:
            map_dict = (
                repository_map.to_dict()
                if isinstance(repository_map, RepositoryMap)
                else dict(repository_map)
            )
            items.append(
                {
                    "kind": "repository_map",
                    "source_id": map_dict.get("map_sha256"),
                    "content_hash": map_dict.get("map_sha256")
                    or canonical_hash(map_dict),
                    "selection_method": "repository_map",
                    "included_range_or_excerpt": {
                        "file_count": map_dict.get("file_count")
                        or len(map_dict.get("files") or []),
                        "truncated": map_dict.get("truncated"),
                        "excluded_prefixes": map_dict.get("excluded_prefixes"),
                        "files": [
                            {
                                "path": entry.get("path")
                                if isinstance(entry, dict)
                                else entry.path,
                                "sha256": entry.get("sha256")
                                if isinstance(entry, dict)
                                else entry.sha256,
                                "size_bytes": entry.get("size_bytes")
                                if isinstance(entry, dict)
                                else entry.size_bytes,
                                "language": entry.get("language")
                                if isinstance(entry, dict)
                                else entry.language,
                            }
                            for entry in (map_dict.get("files") or [])
                        ],
                    },
                }
            )
        for selected in selected_files:
            items.append(
                {
                    "kind": selected.get("kind") or "read_only_file",
                    "source_path": selected.get("path")
                    or selected.get("source_path"),
                    "source_id": selected.get("source_id")
                    or selected.get("path")
                    or selected.get("source_path"),
                    "content_hash": selected.get("content_hash")
                    or selected.get("sha256")
                    or selected.get("content_sha256"),
                    "selection_method": selected.get("selection_method")
                    or "explicit",
                    "included_range_or_excerpt": selected.get(
                        "included_range_or_excerpt",
                        selected.get("excerpt") or selected.get("range"),
                    ),
                    "metadata": dict(selected.get("metadata") or {}),
                }
            )
        subject = {
            "request_id": (request or {}).get("request_id"),
            "task_id": (task or {}).get("task_id"),
            "spec_id": (task_spec or {}).get("spec_id"),
        }
        return self._write_manifest(
            role="planning",
            items=items,
            excluded=[],
            subject=subject,
        )

    def compile_for_worker(
        self,
        *,
        task_spec: Mapping[str, Any],
        writable_artifact: Mapping[str, Any],
        read_only_dependencies: Sequence[Mapping[str, Any]] = (),
        contracts: Sequence[Mapping[str, Any]] = (),
        tests: Sequence[Mapping[str, Any]] = (),
        strategy_summary: str | None = None,
        ledger_projection: Mapping[str, Any] | None = None,
        request: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Worker pack: declared artifact + explicit deps/contracts/tests only."""
        if not writable_artifact:
            raise ContextError("worker context requires a writable_artifact item")
        path = writable_artifact.get("path") or writable_artifact.get("source_path")
        if not path:
            raise ContextError("writable_artifact must declare a path")
        # Refuse whole-repo dumps.
        for label, bundle in (
            ("dependency", read_only_dependencies),
            ("contract", contracts),
            ("test", tests),
        ):
            for item in bundle:
                if item.get("selection_method") == "repository_map":
                    raise ContextError(
                        f"worker context must not include whole-repo maps as {label}"
                    )

        items: list[dict[str, Any]] = []
        items.append(
            {
                "kind": "task_spec",
                "source_id": task_spec.get("spec_id"),
                "content_hash": canonical_hash(dict(task_spec)),
                "selection_method": "task_spec",
                "included_range_or_excerpt": {
                    "objective": task_spec.get("objective"),
                    "requirements": task_spec.get("requirements"),
                    "acceptance_criteria": task_spec.get("acceptance_criteria"),
                    "constraints": task_spec.get("constraints"),
                    "artifact_path": task_spec.get("artifact_path"),
                    "validators": task_spec.get("validators"),
                },
            }
        )
        if request is not None:
            items.append(
                {
                    "kind": "request",
                    "source_id": request.get("request_id"),
                    "content_hash": request.get("immutable_sha256")
                    or canonical_hash(dict(request)),
                    "selection_method": "explicit",
                    "included_range_or_excerpt": {
                        "original_text": request.get("original_text"),
                        "desired_outcome": request.get("desired_outcome"),
                    },
                }
            )
        items.append(
            {
                "kind": "writable_artifact",
                "source_path": path,
                "source_id": path,
                "content_hash": writable_artifact.get("content_hash")
                or writable_artifact.get("sha256")
                or writable_artifact.get("content_sha256"),
                "selection_method": "explicit",
                "included_range_or_excerpt": writable_artifact.get(
                    "included_range_or_excerpt",
                    writable_artifact.get("excerpt")
                    or writable_artifact.get("range"),
                ),
                "metadata": {
                    "role": "writable",
                    **dict(writable_artifact.get("metadata") or {}),
                },
            }
        )
        for group, method_default, kind in (
            (read_only_dependencies, "dependency", "dependency"),
            (contracts, "explicit", "contract"),
            (tests, "explicit", "test"),
        ):
            for item in group:
                items.append(
                    {
                        "kind": item.get("kind") or kind,
                        "source_path": item.get("path") or item.get("source_path"),
                        "source_id": item.get("source_id")
                        or item.get("path")
                        or item.get("source_id"),
                        "content_hash": item.get("content_hash")
                        or item.get("sha256")
                        or item.get("content_sha256"),
                        "selection_method": item.get("selection_method")
                        or method_default,
                        "included_range_or_excerpt": item.get(
                            "included_range_or_excerpt",
                            item.get("excerpt") or item.get("range"),
                        ),
                        "metadata": dict(item.get("metadata") or {}),
                    }
                )
        if strategy_summary is not None:
            items.append(
                {
                    "kind": "strategy_summary",
                    "source_id": "strategy",
                    "content_hash": _sha256_text(strategy_summary),
                    "selection_method": "strategy",
                    "included_range_or_excerpt": strategy_summary,
                }
            )
        if ledger_projection is not None:
            items.append(
                {
                    "kind": "ledger_projection",
                    "source_id": ledger_projection.get("projection_id"),
                    "content_hash": canonical_hash(dict(ledger_projection)),
                    "selection_method": "ledger_projection",
                    "included_range_or_excerpt": {
                        "retained_successes": ledger_projection.get(
                            "retained_successes"
                        ),
                        "rejected_strategies": ledger_projection.get(
                            "rejected_strategies"
                        ),
                        "invalidated_assumptions": ledger_projection.get(
                            "invalidated_assumptions"
                        ),
                    },
                }
            )
        return self._write_manifest(
            role="worker",
            items=items,
            excluded=["implicit whole-repository contents", "repository_map"],
            subject={
                "spec_id": task_spec.get("spec_id"),
                "artifact_path": path,
            },
        )

    def compile_for_review(
        self,
        *,
        task_spec: Mapping[str, Any],
        worker_context_manifest: Mapping[str, Any],
        candidate: Mapping[str, Any],
        validation: Mapping[str, Any] | None = None,
        ledger_projection: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        items: list[dict[str, Any]] = [
            {
                "kind": "task_spec",
                "source_id": task_spec.get("spec_id"),
                "content_hash": canonical_hash(dict(task_spec)),
                "selection_method": "task_spec",
                "included_range_or_excerpt": {
                    "objective": task_spec.get("objective"),
                    "requirements": task_spec.get("requirements"),
                    "acceptance_criteria": task_spec.get("acceptance_criteria"),
                },
            },
            {
                "kind": "prior_context_manifest",
                "source_id": worker_context_manifest.get("context_manifest_id"),
                "content_hash": worker_context_manifest.get("manifest_sha256")
                or canonical_hash(dict(worker_context_manifest)),
                "selection_method": "explicit",
                "included_range_or_excerpt": {
                    "role": worker_context_manifest.get("role"),
                    "item_count": len(worker_context_manifest.get("items") or []),
                    "manifest_sha256": worker_context_manifest.get(
                        "manifest_sha256"
                    ),
                },
            },
            {
                "kind": "candidate",
                "source_path": candidate.get("path") or candidate.get("source_path"),
                "source_id": candidate.get("source_id")
                or candidate.get("artifact_id")
                or candidate.get("path"),
                "content_hash": candidate.get("content_hash")
                or candidate.get("sha256")
                or candidate.get("content_sha256"),
                "selection_method": "candidate",
                "included_range_or_excerpt": candidate.get(
                    "included_range_or_excerpt",
                    candidate.get("excerpt") or candidate.get("content"),
                ),
            },
        ]
        if validation is not None:
            items.append(
                {
                    "kind": "validation",
                    "source_id": validation.get("validation_id"),
                    "content_hash": canonical_hash(dict(validation)),
                    "selection_method": "validation",
                    "included_range_or_excerpt": {
                        "passed": validation.get("passed"),
                        "checks": validation.get("checks"),
                        "candidate_sha256": validation.get("candidate_sha256"),
                    },
                }
            )
        if ledger_projection is not None:
            items.append(
                {
                    "kind": "ledger_projection",
                    "source_id": ledger_projection.get("projection_id"),
                    "content_hash": canonical_hash(dict(ledger_projection)),
                    "selection_method": "ledger_projection",
                    "included_range_or_excerpt": {
                        "observations": ledger_projection.get("observations"),
                        "rejected_strategies": ledger_projection.get(
                            "rejected_strategies"
                        ),
                    },
                }
            )
        return self._write_manifest(
            role="review",
            items=items,
            excluded=[],
            subject={
                "spec_id": task_spec.get("spec_id"),
                "worker_context_manifest_id": worker_context_manifest.get(
                    "context_manifest_id"
                ),
            },
        )

    def compile_for_synthesis(
        self,
        *,
        task_spec: Mapping[str, Any],
        candidates: Sequence[Mapping[str, Any]],
        validations: Sequence[Mapping[str, Any]] = (),
        strengths_and_failures: Mapping[str, Any] | None = None,
        rejected_strategies: Sequence[Any] = (),
    ) -> dict[str, Any]:
        if not candidates:
            raise ContextError("synthesis context requires at least one candidate")
        items: list[dict[str, Any]] = [
            {
                "kind": "task_spec",
                "source_id": task_spec.get("spec_id"),
                "content_hash": canonical_hash(dict(task_spec)),
                "selection_method": "task_spec",
                "included_range_or_excerpt": {
                    "objective": task_spec.get("objective"),
                    "artifact_path": task_spec.get("artifact_path"),
                    "requirements": task_spec.get("requirements"),
                },
            }
        ]
        for index, candidate in enumerate(candidates):
            items.append(
                {
                    "kind": "candidate",
                    "source_path": candidate.get("path")
                    or candidate.get("source_path"),
                    "source_id": candidate.get("source_id")
                    or candidate.get("artifact_id")
                    or f"candidate-{index}",
                    "content_hash": candidate.get("content_hash")
                    or candidate.get("sha256")
                    or candidate.get("content_sha256"),
                    "selection_method": "candidate",
                    "included_range_or_excerpt": candidate.get(
                        "included_range_or_excerpt",
                        candidate.get("excerpt") or candidate.get("content"),
                    ),
                }
            )
        for validation in validations:
            items.append(
                {
                    "kind": "validation",
                    "source_id": validation.get("validation_id"),
                    "content_hash": canonical_hash(dict(validation)),
                    "selection_method": "validation",
                    "included_range_or_excerpt": {
                        "passed": validation.get("passed"),
                        "checks": validation.get("checks"),
                    },
                }
            )
        if strengths_and_failures is not None:
            items.append(
                {
                    "kind": "strengths_and_failures",
                    "source_id": "strengths_and_failures",
                    "content_hash": canonical_hash(dict(strengths_and_failures)),
                    "selection_method": "explicit",
                    "included_range_or_excerpt": dict(strengths_and_failures),
                }
            )
        if rejected_strategies:
            items.append(
                {
                    "kind": "rejected_strategies",
                    "source_id": "rejected_strategies",
                    "content_hash": canonical_hash(list(rejected_strategies)),
                    "selection_method": "ledger_projection",
                    "included_range_or_excerpt": list(rejected_strategies),
                }
            )
        return self._write_manifest(
            role="synthesis",
            items=items,
            excluded=[],
            subject={"spec_id": task_spec.get("spec_id")},
        )

    # Spec-aligned aliases
    def compileForAttempt(
        self,
        *,
        task_spec: Mapping[str, Any],
        writable_artifact: Mapping[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        return self.compile_for_worker(
            task_spec=task_spec,
            writable_artifact=writable_artifact,
            **kwargs,
        )

    def compileForReview(self, **kwargs: Any) -> dict[str, Any]:
        return self.compile_for_review(**kwargs)

    def compileForSynthesis(self, **kwargs: Any) -> dict[str, Any]:
        return self.compile_for_synthesis(**kwargs)
