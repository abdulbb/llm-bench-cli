"""Git utilities for LLM-Bench."""

import os
import subprocess
from dataclasses import dataclass


@dataclass
class GitInfo:
    """Git repository information."""

    commit_hash: str | None
    commit_short: str | None
    branch: str | None
    is_dirty: bool
    tag: str | None

    def to_dict(self) -> dict[str, str | bool | None]:
        """Convert to dictionary for serialization."""
        return {
            "commit_hash": self.commit_hash,
            "commit_short": self.commit_short,
            "branch": self.branch,
            "is_dirty": self.is_dirty,
            "tag": self.tag,
        }

    def summary(self) -> str:
        """Get a short summary string for display."""
        if not self.commit_short:
            return "not a git repo"
        parts = [self.commit_short]
        if self.branch:
            parts.insert(0, self.branch)
        if self.tag:
            parts.append(f"({self.tag})")
        if self.is_dirty:
            parts.append("*")
        return " ".join(parts)


def get_git_info() -> GitInfo:
    """Get current git repository information.

    Returns:
        GitInfo with commit hash, branch, dirty status, and tag.
        Returns empty GitInfo if not in a git repository.
    """
    try:
        # Get full commit hash
        commit_hash = _run_git_command(["git", "rev-parse", "HEAD"])

        # Get short commit hash
        commit_short = _run_git_command(["git", "rev-parse", "--short", "HEAD"])

        # Get current branch name
        branch = _run_git_command(["git", "rev-parse", "--abbrev-ref", "HEAD"])

        # Check if working directory is dirty
        status = _run_git_command(["git", "status", "--porcelain"])
        is_dirty = bool(status)

        # Get tag if exists
        tag = _run_git_command(["git", "describe", "--tags", "--exact-match", "HEAD"])

        return GitInfo(
            commit_hash=commit_hash,
            commit_short=commit_short,
            branch=branch,
            is_dirty=is_dirty,
            tag=tag,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        # Not a git repo or git not installed
        return GitInfo(
            commit_hash=None,
            commit_short=None,
            branch=None,
            is_dirty=False,
            tag=None,
        )


def _get_safe_git_env() -> dict[str, str]:
    """Create a sanitized environment for git commands.

    This prevents git hooks and unsafe configurations from executing
    arbitrary code when running git commands.

    Returns:
        Sanitized environment dictionary.
    """
    env = os.environ.copy()
    # Disable system and global git config
    env["GIT_CONFIG_NOSYSTEM"] = "1"
    env["GIT_CONFIG_GLOBAL"] = ""
    # Disable credential helpers and prompts
    env["GIT_ASKPASS"] = ""
    env["GIT_TERMINAL_PROMPT"] = "0"
    # Disable SSH prompts
    env["GIT_SSH_COMMAND"] = "ssh -oBatchMode=yes"
    return env


def _run_git_command(cmd: list[str]) -> str | None:
    """Run a git command with sanitized environment.

    Security: Uses sanitized environment to prevent git hooks and
    unsafe configurations from executing arbitrary code.

    Args:
        cmd: Git command as list of strings.

    Returns:
        Command output or None if it fails.
    """
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
            shell=False,  # Explicitly disable shell
            env=_get_safe_git_env(),
        )
        return result.stdout.strip() or None
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
