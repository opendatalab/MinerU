import pytest

from mineru.doclib.utils.route_utils import RouteInfo, get_route_info, has_route_info, route


def test_route_attaches_metadata_to_function() -> None:
    @route("POST", "/parses", tags=("parse",))
    def parse() -> None:
        return None

    route_info = get_route_info(parse)
    assert route_info.method == "POST"
    assert route_info.path == "/parses"
    assert route_info.tags == ("parse",)
    assert route_info.index > 0
    assert has_route_info(parse)


def test_route_info_can_be_read_from_bound_method() -> None:
    class Client:
        @route("GET", "/server/status", tags=("server",))
        def server_status(self) -> None:
            return None

    route_info = get_route_info(Client().server_status)
    assert route_info.method == "GET"
    assert route_info.path == "/server/status"
    assert route_info.tags == ("server",)
    assert route_info.index > 0


def test_route_index_increases_by_definition_order() -> None:
    @route("GET", "/first")
    def first() -> None:
        return None

    @route("GET", "/second")
    def second() -> None:
        return None

    assert get_route_info(first).index < get_route_info(second).index


def test_get_route_info_rejects_unannotated_callable() -> None:
    def unannotated() -> None:
        return None

    assert not has_route_info(unannotated)
    with pytest.raises(ValueError, match="does not have doclib route metadata"):
        get_route_info(unannotated)


def test_route_info_validates_method_path_and_tags() -> None:
    with pytest.raises(ValueError, match="unsupported route method"):
        RouteInfo(method="TRACE", path="/x")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="must start"):
        RouteInfo(method="GET", path="x")

    with pytest.raises(ValueError, match="must not end"):
        RouteInfo(method="GET", path="/x/")

    with pytest.raises(ValueError, match="empty"):
        RouteInfo(method="GET", path="/x", tags=("valid", ""))

    with pytest.raises(ValueError, match="non-negative"):
        RouteInfo(method="GET", path="/x", index=-1)
