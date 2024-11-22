import pytest
import json
from magic_pdf.libs.json_compressor import JsonCompressor

# Test data fixtures
@pytest.fixture
def test_cases():
    return [
        # Simple dictionary
        {"name": "John", "age": 30},
        
        # Nested dictionary
        {
            "person": {
                "name": "Alice",
                "address": {
                    "street": "123 Main St",
                    "city": "New York"
                }
            }
        },
        
        # List of dictionaries
        [
            {"id": 1, "value": "first"},
            {"id": 2, "value": "second"}
        ],
        
        # Dictionary with various data types
        {
            "string": "hello",
            "integer": 42,
            "float": 3.14,
            "boolean": True,
            "null": None,
            "array": [1, 2, 3],
            "nested": {"key": "value"}
        },
        
        # Empty structures
        {},
        [],
        {"empty_list": [], "empty_dict": {}}
    ]

@pytest.fixture
def large_data():
    return {
        "data": ["test" * 100] * 100  # Create a large repeated string
    }

def test_compression_decompression_cycle(test_cases):
    """Test that data remains intact after compression and decompression"""
    for test_data in test_cases:
        # Compress the data
        compressed = JsonCompressor.compress_json(test_data)
        
        # Verify compressed string is not empty and is a string
        assert isinstance(compressed, str)
        assert len(compressed) > 0
        
        # Decompress the data
        decompressed = JsonCompressor.decompress_json(compressed)
        
        # Verify the decompressed data matches original
        assert test_data == decompressed

def test_compression_reduces_size(large_data):
    """Test that compression actually reduces data size for large enough input"""
    original_size = len(json.dumps(large_data))
    compressed = JsonCompressor.compress_json(large_data)
    compressed_size = len(compressed)
    
    # Verify compression actually saved space
    assert compressed_size < original_size

def test_invalid_json_serializable():
    """Test handling of non-JSON serializable input"""
    with pytest.raises(TypeError):
        JsonCompressor.compress_json(set([1, 2, 3]))  # sets are not JSON serializable

def test_invalid_compressed_string():
    """Test handling of invalid compressed string"""
    with pytest.raises(Exception):
        JsonCompressor.decompress_json("invalid_base64_string")

def test_empty_string_input():
    """Test handling of empty string input"""
    with pytest.raises(Exception):
        JsonCompressor.decompress_json("")

def test_special_characters():
    """Test handling of special characters"""
    test_data = {
        "special": "!@#$%^&*()_+-=[]{}|;:,.<>?",
        "unicode": "Hello 世界 🌍"
    }
    
    compressed = JsonCompressor.compress_json(test_data)
    decompressed = JsonCompressor.decompress_json(compressed)
    assert test_data == decompressed

# Parametrized test for different types of input
@pytest.mark.parametrize("test_input", [
    {"simple": "value"},
    [1, 2, 3],
    {"nested": {"key": "value"}},
    ["mixed", 1, True, None],
    {"unicode": "🌍"}
])
def test_various_input_types(test_input):
    """Test compression and decompression with various input types"""
    compressed = JsonCompressor.compress_json(test_input)
    decompressed = JsonCompressor.decompress_json(compressed)
    assert test_input == decompressed