"""HTML head/footer templates -- mirrors the Java HTMLExporter output."""

HEAD = (
    "<!DOCTYPE html>\n"
    '<html lang="zh-CN">\n'
    "<head>\n"
    '  <meta charset="UTF-8">\n'
    '  <meta http-equiv="X-UA-Compatible" content="IE=edge">\n'
    '  <meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
    "  <title>OFD Preview</title>\n"
    "</head>\n"
    '<body style="margin:0;background:#E6E8EB;">\n'
    '  <div style="display:flex;flex-direction:column;align-items:center;">'
    '<div style="height:10px;"></div>'
)

PAGE_GAP = '<div style="height:10px;"></div>'

FOOT = "  </div>\n</body>\n</html>"
