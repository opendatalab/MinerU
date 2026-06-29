import pytest
import subprocess
import sys
import os


@pytest.mark.parametrize("payload", [
    # Exact exploit case: command injection via shell metacharacters
    "--model-path /tmp/; echo 'EXPLOIT' > /tmp/hacked.txt; #",
    # Boundary case: argument with spaces and quotes that could break parsing
    "--model-path '$(echo vulnerable)'",
    # Valid input: normal argument that should work correctly
    "--model-path /safe/path/to/model",
])
def test_lmdeploy_server_shell_injection_invariant(payload):
    """Invariant: Arguments passed to lmdeploy_server.py must not allow arbitrary command execution."""
    # Construct the command to run the actual production script
    script_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "mineru", "model", "vlm", "lmdeploy_server.py"
    )
    
    # Use subprocess to run the script with the payload as argument
    # This tests the actual code path with real os.system() call
    result = subprocess.run(
        [sys.executable, script_path, payload],
        capture_output=True,
        text=True,
        timeout=2  # Prevent hanging if injection succeeds
    )
    
    # Security property: The script must not execute arbitrary commands
    # We check that no unexpected output appears (like EXPLOIT from our test)
    # and that the script either succeeds normally or fails safely
    assert "EXPLOIT" not in result.stdout, f"Command injection detected in stdout: {result.stdout}"
    assert "EXPLOIT" not in result.stderr, f"Command injection detected in stderr: {result.stderr}"
    
    # Additional safety check: ensure no unexpected files were created
    # This catches the file creation attempt in our exploit payload
    exploit_file = "/tmp/hacked.txt"
    if os.path.exists(exploit_file):
        os.remove(exploit_file)
        pytest.fail(f"Security invariant violated: exploit file {exploit_file} was created")