import pytest
import hashlib
import sys
import os

# Add the module path to sys.path to import the actual production function
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from mineru.utils.hash_utils import bytes_md5


@pytest.mark.parametrize("payload", [
    # Exact exploit case: MD5 collision payloads (first 64 bytes of two colliding messages)
    b"d131dd02c5e6eec4693d9a0698aff95c2fcab58712467eab4004583eb8fb7f8955ad340609f4b30283e488832571415a085125e8f7cdc99fd91dbdf280373c5bd8823e3156348f5bae6dacd436c919c6dd53e2b487da03fd02396306d248cda0e99f33420f577ee8ce54b67080a80d1ec69821bcb6a8839396f9652b6ff72a70",
    # Boundary case: Empty bytes (edge case for hashing)
    b"",
    # Valid input: Normal file content
    b"Hello, World! This is a test file content.",
])
def test_md5_hash_maintains_cryptographic_integrity(payload):
    """Invariant: MD5 hash output must be deterministic and consistent even with adversarial inputs."""
    # Call the actual production function
    result = bytes_md5(payload)
    
    # Property 1: Output must be a string
    assert isinstance(result, str), "MD5 hash output must be a string"
    
    # Property 2: Output must be uppercase hex (as per function spec)
    assert result.isupper(), "MD5 hash output must be uppercase"
    
    # Property 3: Output must be exactly 32 characters (MD5 hex digest length)
    assert len(result) == 32, "MD5 hex digest must be 32 characters"
    
    # Property 4: Output must be valid hexadecimal
    assert all(c in "0123456789ABCDEF" for c in result), "MD5 output must be valid hex"
    
    # Property 5: Must match Python's hashlib.md5 result (consistency check)
    expected = hashlib.md5(payload).hexdigest().upper()
    assert result == expected, "MD5 implementation must match standard library"