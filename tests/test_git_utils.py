"""Tests for git utilities."""

import subprocess
from unittest.mock import MagicMock, patch

from llm_bench.git_utils import GitInfo, _run_git_command, get_git_info


class TestRunGitCommand:
    """Tests for _run_git_command."""

    def test_successful_command(self) -> None:
        """Test that successful git commands return their output."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout="test output\n",
                returncode=0,
            )
            result = _run_git_command(["git", "rev-parse", "HEAD"])
            assert result == "test output"

    def test_empty_output_returns_none(self) -> None:
        """Test that empty output returns None."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout="",
                returncode=0,
            )
            result = _run_git_command(["git", "describe", "--tags"])
            assert result is None

    def test_failed_command_returns_none(self) -> None:
        """Test that failed commands return None."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(1, "git")
            result = _run_git_command(["git", "describe", "--tags"])
            assert result is None

    def test_timeout_returns_none(self) -> None:
        """Test that timeouts return None."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired("git", 5)
            result = _run_git_command(["git", "status"])
            assert result is None


class TestGetGitInfo:
    """Tests for get_git_info."""

    def test_successful_git_info(self) -> None:
        """Test successful retrieval of git info."""
        with patch("llm_bench.git_utils._run_git_command") as mock_cmd:
            mock_cmd.side_effect = [
                "abc123def456",  # commit_hash
                "abc123d",  # commit_short
                "main",  # branch
                "",  # status (clean)
                "v1.0.0",  # tag
            ]
            info = get_git_info()
            assert info.commit_hash == "abc123def456"
            assert info.commit_short == "abc123d"
            assert info.branch == "main"
            assert info.is_dirty is False
            assert info.tag == "v1.0.0"

    def test_dirty_repo(self) -> None:
        """Test detection of dirty working directory."""
        with patch("llm_bench.git_utils._run_git_command") as mock_cmd:
            mock_cmd.side_effect = [
                "abc123def456",
                "abc123d",
                "feature-branch",
                " M src/file.py\n",  # dirty
                None,  # no tag
            ]
            info = get_git_info()
            assert info.is_dirty is True
            assert info.tag is None

    def test_not_git_repo(self) -> None:
        """Test handling of non-git directories."""
        with patch("llm_bench.git_utils._run_git_command") as mock_cmd:
            mock_cmd.side_effect = subprocess.CalledProcessError(1, "git")
            info = get_git_info()
            assert info.commit_hash is None
            assert info.commit_short is None
            assert info.branch is None
            assert info.is_dirty is False
            assert info.tag is None


class TestGitInfoSummary:
    """Tests for GitInfo.summary method."""

    def test_summary_with_all_info(self) -> None:
        """Test summary with full git info."""
        info = GitInfo(
            commit_hash="abc123def456",
            commit_short="abc123d",
            branch="main",
            is_dirty=False,
            tag="v1.0.0",
        )
        assert info.summary() == "main abc123d (v1.0.0)"

    def test_summary_dirty_repo(self) -> None:
        """Test summary shows dirty indicator."""
        info = GitInfo(
            commit_hash="abc123def456",
            commit_short="abc123d",
            branch="feature",
            is_dirty=True,
            tag=None,
        )
        assert info.summary() == "feature abc123d *"

    def test_summary_no_git(self) -> None:
        """Test summary when not in git repo."""
        info = GitInfo(
            commit_hash=None,
            commit_short=None,
            branch=None,
            is_dirty=False,
            tag=None,
        )
        assert info.summary() == "not a git repo"

    def test_summary_no_branch(self) -> None:
        """Test summary when branch is None (detached HEAD)."""
        info = GitInfo(
            commit_hash="abc123def456",
            commit_short="abc123d",
            branch=None,
            is_dirty=False,
            tag=None,
        )
        assert info.summary() == "abc123d"
