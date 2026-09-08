"""Global constants shared across modules."""

# OFD XML namespace (GB/T 33190-2016).
OFD_NS = "http://www.ofdspec.org/2016"
NSMAP = {"ofd": OFD_NS}

# OFD page coordinates are in millimeters; HTML SVG output uses 1 mm = 1 user unit
# (we put `viewBox` in mm, then scale via CSS so 1 mm == ~3.78 px at 96 dpi).
MM_TO_PX = 96.0 / 25.4
