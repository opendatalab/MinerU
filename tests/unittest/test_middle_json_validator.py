from mineru.parser import MIDDLE_JSON_SCHEMA_VERSION


def test_middle_json_schema_version_is_public_constant() -> None:
    assert MIDDLE_JSON_SCHEMA_VERSION == "2.0"
