"""In-process authority objects for Cosmo's model/operator boundary.

These objects are deliberately not JSON contracts.  A model can name a tool
and provide JSON arguments, but it cannot construct an authenticated caller or
an operator-approved repository capability.  Trusted entry points create them
before inference begins and the dispatcher carries them out-of-band.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from stat import S_ISDIR
from typing import Any, Literal
from uuid import uuid4


class AuthorityError(RuntimeError):
    """Caller or capability authentication failed."""


_CALLER_SEAL = object()
_CAPABILITY_SEAL = object()


@dataclass(frozen=True)
class CallerProvenance:
    """Authenticated, immutable identity attached outside model-controlled JSON."""

    principal_id: str
    principal_type: Literal["model", "operator"]
    authenticated_by: str
    authentication_id: str
    _seal: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._seal is not _CALLER_SEAL:
            raise AuthorityError("caller provenance was not authenticated")
        if not self.principal_id.strip() or not self.authenticated_by.strip():
            raise AuthorityError("caller provenance fields must not be empty")

    def to_record(self) -> dict[str, str]:
        return {
            "principal_id": self.principal_id,
            "principal_type": self.principal_type,
            "authenticated_by": self.authenticated_by,
            "authentication_id": self.authentication_id,
        }


def authenticate_operator(
    principal_id: str = "local-operator",
    *,
    authenticated_by: str = "cosmo.operator-entrypoint",
) -> CallerProvenance:
    """Authenticate the local human/operator entry point.

    This is a process authority constructor, not a tool and not part of any
    model-visible schema.  Deployments may replace the entry-point check with
    stronger OS/session authentication without changing service contracts.
    """

    return CallerProvenance(
        principal_id=principal_id,
        principal_type="operator",
        authenticated_by=authenticated_by,
        authentication_id=f"AUTH-{uuid4().hex}",
        _seal=_CALLER_SEAL,
    )


def authenticate_model(
    principal_id: str,
    *,
    authenticated_by: str = "cosmo.chat-session",
) -> CallerProvenance:
    """Bind a model principal to the trusted chat session that invokes it."""

    return CallerProvenance(
        principal_id=principal_id,
        principal_type="model",
        authenticated_by=authenticated_by,
        authentication_id=f"AUTH-{uuid4().hex}",
        _seal=_CALLER_SEAL,
    )


def require_operator(caller: CallerProvenance | None) -> CallerProvenance:
    """Fail closed unless *caller* is an authenticated operator principal."""

    if not isinstance(caller, CallerProvenance) or caller._seal is not _CALLER_SEAL:
        raise AuthorityError("authenticated operator provenance is required")
    if caller.principal_type != "operator":
        raise AuthorityError("operator-only action denied to non-operator caller")
    return caller


def require_model(caller: CallerProvenance | None) -> CallerProvenance:
    """Fail closed unless *caller* is an authenticated model principal."""

    if not isinstance(caller, CallerProvenance) or caller._seal is not _CALLER_SEAL:
        raise AuthorityError("authenticated model provenance is required")
    if caller.principal_type != "model":
        raise AuthorityError("model review provenance requires a model caller")
    return caller


@dataclass(frozen=True)
class RepoRootCapability:
    """Immutable operator grant for one resolved repository root and grantee."""

    capability_id: str
    repo_root: Path
    grantee_principal_id: str
    grantee_authentication_id: str
    granted_by: CallerProvenance
    allowed_tools: frozenset[str]
    root_device: int
    root_inode: int
    _seal: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._seal is not _CAPABILITY_SEAL:
            raise AuthorityError("repository capability was not operator-approved")
        if not self.repo_root.is_absolute() or not self.repo_root.is_dir():
            raise AuthorityError(
                f"approved repo root is not an existing absolute directory: {self.repo_root}"
            )

    def authorize(self, caller: CallerProvenance, tool_name: str) -> Path:
        if self._seal is not _CAPABILITY_SEAL:
            raise AuthorityError("repository capability authentication failed")
        if (
            caller.principal_id != self.grantee_principal_id
            or caller.authentication_id != self.grantee_authentication_id
        ):
            raise AuthorityError("repository capability belongs to another caller")
        if tool_name not in self.allowed_tools:
            raise AuthorityError(
                f"repository capability does not authorize tool: {tool_name}"
            )
        try:
            root_stat = self.repo_root.lstat()
            resolved = self.repo_root.resolve(strict=True)
        except OSError as exc:
            raise AuthorityError(
                "approved repository root is no longer available"
            ) from exc
        if (
            self.repo_root.is_symlink()
            or resolved != self.repo_root
            or not S_ISDIR(root_stat.st_mode)
            or root_stat.st_dev != self.root_device
            or root_stat.st_ino != self.root_inode
        ):
            raise AuthorityError(
                "approved repository root identity changed after capability grant"
            )
        return self.repo_root

    def to_record(self) -> dict[str, Any]:
        return {
            "capability_id": self.capability_id,
            "repo_root": str(self.repo_root),
            "grantee_principal_id": self.grantee_principal_id,
            "grantee_authentication_id": self.grantee_authentication_id,
            "granted_by": self.granted_by.to_record(),
            "allowed_tools": sorted(self.allowed_tools),
            "root_device": self.root_device,
            "root_inode": self.root_inode,
        }


def approve_repo_root(
    operator: CallerProvenance,
    *,
    repo_root: Path,
    grantee: CallerProvenance,
    allowed_tools: frozenset[str],
) -> RepoRootCapability:
    """Issue an immutable root capability from an authenticated operator."""

    authority = require_operator(operator)
    root = Path(repo_root).resolve(strict=True)
    if not root.is_dir():
        raise AuthorityError(f"repo root is not a directory: {root}")
    root_stat = root.lstat()
    return RepoRootCapability(
        capability_id=f"REPOCAP-{uuid4().hex}",
        repo_root=root,
        grantee_principal_id=grantee.principal_id,
        grantee_authentication_id=grantee.authentication_id,
        granted_by=authority,
        allowed_tools=frozenset(allowed_tools),
        root_device=root_stat.st_dev,
        root_inode=root_stat.st_ino,
        _seal=_CAPABILITY_SEAL,
    )
