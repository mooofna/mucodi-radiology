"""Contract: import rate_eval does not pull torch."""

import subprocess
import sys
import textwrap


def test_import_rate_eval_does_not_pull_torch():
    script = textwrap.dedent(
        """
        import sys
        import rate_eval  # noqa: F401

        torch_modules = sorted(k for k in sys.modules if k == "torch" or k.startswith("torch."))
        if torch_modules:
            print("FAIL: torch loaded by `import rate_eval`")
            for name in torch_modules[:20]:
                print(f"  - {name}")
            sys.exit(1)
        print("OK: import rate_eval is torch-free")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"`import rate_eval` pulled torch into sys.modules.\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
