"""fcntl-based lock preventing concurrent processes from downloading the same HF checkpoint."""

import contextlib
import fcntl
import os
import time
from pathlib import Path
from typing import Optional

from transformers.utils import cached_file

from ..core.logging import get_logger


class ModelDownloadLock:
    """Lock preventing concurrent downloads of the same model."""

    def __init__(self, model_repo_id: str, revision: str = "main", cache_dir: Optional[str] = None):
        self.model_repo_id = model_repo_id
        self.revision = revision
        self.cache_dir = cache_dir

        cache_root = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "huggingface"
        lock_dir = cache_root / "download_locks"
        lock_dir.mkdir(parents=True, exist_ok=True)

        safe_repo_id = model_repo_id.replace("/", "_").replace(":", "_")
        safe_revision = revision.replace("/", "_").replace(":", "_")
        self.lock_file = lock_dir / f"{safe_repo_id}_{safe_revision}.lock"

        logger = get_logger(__name__)
        logger.debug(f"ModelDownloadLock initialized for {model_repo_id}@{revision}")
        logger.debug(f"Lock file: {self.lock_file}")

    def is_model_downloaded(self) -> bool:
        """Return True iff the model's config.json is already cached locally."""
        try:
            cached_file(
                path_or_repo_id=self.model_repo_id,
                filename="config.json",
                revision=self.revision,
                cache_dir=self.cache_dir,
                local_files_only=True,
            )
            return True
        except Exception:
            return False

    @contextlib.contextmanager
    def acquire_download_lock(self, timeout: int = 300):
        """Acquire an exclusive download lock; raise TimeoutError on contention."""
        logger = get_logger(__name__)

        if os.environ.get("RATE_SKIP_DOWNLOAD_LOCK", "0") == "1":
            logger.debug(
                "Skipping download lock for %s@%s due to RATE_SKIP_DOWNLOAD_LOCK",
                self.model_repo_id,
                self.revision,
            )
            yield
            return

        if self.is_model_downloaded():
            logger.debug(
                f"Model {self.model_repo_id}@{self.revision} already downloaded, skipping lock"
            )
            yield
            return

        logger.info(f"Acquiring download lock for {self.model_repo_id}@{self.revision}")

        start_time = time.time()
        lock_acquired = False
        lock_file_handle = None

        try:
            while time.time() - start_time < timeout:
                try:
                    lock_file_handle = open(self.lock_file, "w")
                    fcntl.flock(lock_file_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    lock_acquired = True

                    lock_file_handle.write(f"pid:{os.getpid()}\n")
                    lock_file_handle.write(f"model:{self.model_repo_id}@{self.revision}\n")
                    lock_file_handle.write(f"timestamp:{time.time()}\n")
                    lock_file_handle.flush()

                    logger.info(f"Download lock acquired for {self.model_repo_id}@{self.revision}")
                    break

                except (OSError, IOError):
                    if lock_file_handle:
                        lock_file_handle.close()
                        lock_file_handle = None

                    time.sleep(1)

                    if self.is_model_downloaded():
                        logger.debug(
                            f"Model {self.model_repo_id}@{self.revision} downloaded by another process"
                        )
                        yield
                        return

            if not lock_acquired:
                raise TimeoutError(
                    f"Could not acquire download lock for {self.model_repo_id}@{self.revision} within {timeout} seconds"
                )

            # re-check: another process may have finished while blocked
            if self.is_model_downloaded():
                logger.debug(
                    f"Model {self.model_repo_id}@{self.revision} already downloaded, releasing lock"
                )
                yield
                return

            logger.debug(f"Proceeding with download for {self.model_repo_id}@{self.revision}")
            yield

        finally:
            if lock_acquired and lock_file_handle:
                try:
                    fcntl.flock(lock_file_handle.fileno(), fcntl.LOCK_UN)
                    lock_file_handle.close()
                    logger.debug(f"Download lock released for {self.model_repo_id}@{self.revision}")
                except Exception as e:
                    logger.warning(f"Error releasing download lock: {e}")

            try:
                if self.lock_file.exists():
                    self.lock_file.unlink()
            except Exception as e:
                logger.warning(f"Error removing lock file: {e}")
