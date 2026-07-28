# Copyright (c) Opendatalab. All rights reserved.
import base64

from mineru.backend.office.mkcontent.output_builders import mk_blocks_to_markdown
from mineru.backend.utils.office_chart import (
    parse_chart_spec_from_ooxml,
    render_chart_svg_from_ooxml,
)
from mineru.utils.enum_class import BlockType, ContentType, MakeMode


_SCATTER_CHART_XML = b"""<c:chartSpace xmlns:c="http://schemas.openxmlformats.org/drawingml/2006/chart"
              xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
  <c:chart>
    <c:title><c:tx><c:rich><a:p><a:r><a:t>Discharge Curve</a:t></a:r></a:p></c:rich></c:tx></c:title>
    <c:plotArea>
      <c:scatterChart>
        <c:ser>
          <c:tx><c:v>Cell 1</c:v></c:tx>
          <c:spPr><a:ln w="25400"><a:solidFill><a:srgbClr val="FF0000"/></a:solidFill><a:prstDash val="dash"/></a:ln></c:spPr>
          <c:xVal><c:numRef><c:numCache>
            <c:pt idx="0"><c:v>0</c:v></c:pt>
            <c:pt idx="1"><c:v>1</c:v></c:pt>
            <c:pt idx="2"><c:v>2</c:v></c:pt>
          </c:numCache></c:numRef></c:xVal>
          <c:yVal><c:numRef><c:numCache>
            <c:pt idx="0"><c:v>4.1</c:v></c:pt>
            <c:pt idx="1"><c:v>3.7</c:v></c:pt>
            <c:pt idx="2"><c:v>2.8</c:v></c:pt>
          </c:numCache></c:numRef></c:yVal>
        </c:ser>
      </c:scatterChart>
      <c:valAx>
        <c:scaling><c:min val="0"/><c:max val="2"/></c:scaling>
        <c:axPos val="b"/>
        <c:title><c:tx><c:rich><a:p><a:r><a:t>Capacity</a:t></a:r></a:p></c:rich></c:tx></c:title>
        <c:majorUnit val="1"/>
      </c:valAx>
      <c:valAx>
        <c:scaling><c:min val="2.5"/><c:max val="4.5"/></c:scaling>
        <c:axPos val="l"/>
        <c:title><c:tx><c:rich><a:p><a:r><a:t>Voltage</a:t></a:r></a:p></c:rich></c:tx></c:title>
        <c:majorUnit val="0.5"/>
      </c:valAx>
    </c:plotArea>
  </c:chart>
</c:chartSpace>
"""


def test_render_scatter_chart_svg_preserves_visual_properties() -> None:
    spec = parse_chart_spec_from_ooxml(_SCATTER_CHART_XML)

    assert spec is not None
    assert spec.title == "Discharge Curve"
    assert spec.x_axis_title == "Capacity"
    assert spec.value_axis_title == "Voltage"
    assert (spec.x_axis_min, spec.x_axis_max, spec.x_axis_major_unit) == (0, 2, 1)
    assert (spec.y_axis_min, spec.y_axis_max, spec.y_axis_major_unit) == (2.5, 4.5, 0.5)
    assert spec.series[0].line_color == "FF0000"
    assert spec.series[0].line_dash == "dash"
    assert spec.series[0].line_width == 2

    image_data_uri = render_chart_svg_from_ooxml(_SCATTER_CHART_XML)
    prefix, encoded_svg = image_data_uri.split(",", 1)
    svg = base64.b64decode(encoded_svg).decode("utf-8")

    assert prefix == "data:image/svg+xml;base64"
    assert "<svg" in svg
    assert "Discharge Curve" in svg
    assert "Capacity" in svg
    assert "Voltage" in svg
    assert "Cell 1" in svg
    assert "rgb(100%,0%,0%)" in svg.replace(" ", "").lower()
    assert "stroke-dasharray" in svg

    assert "matrix(0,1,-1,0" in svg.replace(" ", "")

def test_chart_markdown_prefers_image_and_folds_structured_content() -> None:
    chart_html = "<table><tbody><tr><td>4.1</td></tr></tbody></table>"
    chart_block = {
        "type": BlockType.CHART,
        "blocks": [
            {
                "type": BlockType.CHART_BODY,
                "lines": [
                    {
                        "spans": [
                            {
                                "type": ContentType.CHART,
                                "image_path": "chart.svg",
                                "content": chart_html,
                            }
                        ]
                    }
                ],
            }
        ],
    }

    markdown = mk_blocks_to_markdown(
        [chart_block],
        MakeMode.MM_MD,
        "images",
    )

    assert len(markdown) == 1
    assert markdown[0].startswith("![](images/chart.svg)")
    assert "<details>" in markdown[0]
    assert "<summary>chart content</summary>" in markdown[0]
    assert chart_html in markdown[0]
