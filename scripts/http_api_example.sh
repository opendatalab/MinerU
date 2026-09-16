#!/usr/bin/env bash
# mineru-http-example — one complete V1 API cycle with plain HTTP calls:
# create upload -> upload bytes -> complete -> submit parse job -> bounded poll -> download artifacts.
#
# The documentation pages (docs/{en,zh}/usage/http_api.md) show excerpts of this
# script; keep them in sync by copying from here, never by editing both copies.
#
# Usage:
#   MINERU_API_URL=http://127.0.0.1:8000 [MINERU_API_KEY=...] \
#     ./scripts/http_api_example.sh document.html
#
# Environment:
#   MINERU_API_URL     (required)  Service root before /v1.
#   MINERU_API_KEY     (optional)  Bearer key; attached only to same-origin requests.
#   MINERU_TIER        (optional)  Requested tier (default: flash, for native documents).
#   OUTPUT_FORMATS     (optional)  Comma-separated (default: markdown,zip).
#   OUTPUT_DIR         (optional)  Download directory (default: current directory).
#   PAGE_RANGE         (optional)  PDF page range such as "1-10"; omit for whole/native documents.
#   CONNECT_TIMEOUT    (optional)  Connect timeout per request in seconds (default: 10).
#   REQUEST_TIMEOUT    (optional)  Max seconds per regular request (default: 60).
#   TRANSFER_TIMEOUT   (optional)  Max seconds per upload/download (default: 300).
#   MAX_POLLS          (optional)  Polling attempts before giving up (default: 100).
#   POLL_INTERVAL      (optional)  Seconds between polls (default: 3).
#
# Exit codes:
#   0    Job completed and every requested artifact was saved.
#   1    Script, transport, protocol, download, or local write failure.
#   2    Job partial: artifacts of the completed files were saved first.
#   3    Job failed.
#   4    Job canceled.
#   124  Polling budget exhausted; the job keeps running. job_id is printed
#        so the same polling request can be resumed later.
#
# Notes:
#   - Requires bash, curl (>= 7.83.0 for cross-port credential isolation), jq,
#     python3, and sha256sum or shasum.
#   - MAX_POLLS bounds the number of polling requests, not the total wall time:
#     each request also spends up to REQUEST_TIMEOUT plus POLL_INTERVAL.
#   - The byte upload uses the method/URL/headers returned by the service. The
#     MinerU API key is attached only when the upload URL is same-origin with
#     the API base (scheme + host + effective port, relative URLs resolved
#     against the base). Service-provided upload headers are always kept
#     as-is. --location-trusted is never used.
set -euo pipefail

die() {
  echo "error: $*" >&2
  exit 1
}

command -v curl >/dev/null 2>&1 || die "curl is required"
command -v jq >/dev/null 2>&1 || die "jq is required"
command -v python3 >/dev/null 2>&1 || die "python3 is required"

BASE_URL="${MINERU_API_URL:-}"
API_KEY="${MINERU_API_KEY:-}"
TIER="${MINERU_TIER:-flash}"
OUTPUT_FORMATS_CSV="${OUTPUT_FORMATS:-markdown,zip}"
OUTPUT_DIR="${OUTPUT_DIR:-.}"
PAGE_RANGE="${PAGE_RANGE:-}"
CONNECT_TIMEOUT="${CONNECT_TIMEOUT:-10}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-60}"
TRANSFER_TIMEOUT="${TRANSFER_TIMEOUT:-300}"
MAX_POLLS="${MAX_POLLS:-100}"
POLL_INTERVAL="${POLL_INTERVAL:-3}"

[ -n "$BASE_URL" ] || die "MINERU_API_URL is required"
BASE_URL="${BASE_URL%/}"
[ $# -eq 1 ] || die "usage: $0 <input-file>"
INPUT_FILE=$1
[ -f "$INPUT_FILE" ] || die "file not found: $INPUT_FILE"
mkdir -p -- "$OUTPUT_DIR" || die "cannot create output dir: $OUTPUT_DIR"

REQUEST_BODY=$(mktemp)
RESPONSE=$(mktemp)
JOB_RESPONSE=$(mktemp)
trap 'rm -f "$REQUEST_BODY" "$RESPONSE" "$JOB_RESPONSE"' EXIT

auth_args=()
if [ -n "$API_KEY" ]; then
  auth_args=(-H "Authorization: Bearer $API_KEY")
fi

# Request expecting a JSON body saved to $RESPONSE. Tool exit codes are
# converted to script exit code 1; curl/jq codes are never passed through.
curl_json() { # curl_json <stage> <max-time> <curl-args...>
  local stage=$1 max_time=$2
  shift 2
  local rc=0
  curl --silent --show-error --fail \
    --connect-timeout "$CONNECT_TIMEOUT" --max-time "$max_time" \
    -o "$RESPONSE" "$@" || rc=$?
  if [ "$rc" -ne 0 ]; then
    die "$stage failed (curl exit $rc)"
  fi
}

# Extract a non-empty string field; jq -e makes null/false/missing exit non-zero.
json_string() { # json_string <file> <jq-expr> <label>
  local value
  value=$(jq -er "$2 | select(type == \"string\" and length > 0)" "$1") || die "invalid response at $3: field $2 is not a non-empty string"
  printf '%s\n' "$value"
}

job_status_ok() {
  case "$1" in
    queued | running | completed | partial | failed | canceled) return 0 ;;
    *) return 1 ;;
  esac
}

is_terminal() {
  case "$1" in
    completed | partial | failed | canceled) return 0 ;;
    *) return 1 ;;
  esac
}

file_size() {
  stat -c%s "$1" 2>/dev/null || stat -f%z "$1"
}

file_sha256() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | cut -d' ' -f1
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | cut -d' ' -f1
  else
    die "sha256sum or shasum is required"
  fi
}

# Same-origin check aligned with the SDK rule (api_client._same_origin_upload_headers):
# compare scheme + host + effective port; resolve relative upload URLs against
# the API base. Prints the resolved absolute URL on stdout. Exit 0 = same
# origin, 1 = different, 2 = invalid URL.
resolve_upload_url() { # resolve_upload_url <api-base> <target-url>
  python3 - "$1" "$2" <<'PY'
import sys
from urllib.parse import urljoin, urlsplit


def origin(url: str):
    parts = urlsplit(url)
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        raise ValueError(f"not an http(s) URL: {url!r}")
    return parts.scheme.lower(), parts.hostname.lower(), parts.port or (443 if parts.scheme == "https" else 80)


try:
    base = origin(sys.argv[1])
    resolved = urljoin(sys.argv[1], sys.argv[2])
    target = origin(resolved)
except ValueError as exc:
    print(exc, file=sys.stderr)
    raise SystemExit(2)
print(resolved)
raise SystemExit(0 if base == target else 1)
PY
}

# Build UPLOAD_HEADER_ARGS from the service-provided upload_headers mapping.
# Headers are passed as a bash argument array; values containing newlines are
# rejected; no eval and no unquoted string expansion.
UPLOAD_HEADER_ARGS=()
build_upload_header_args() { # build_upload_header_args <response-file>
  local response=$1 key value
  jq -e '(.upload_headers // {}) | type == "object" and ([to_entries[].value | type] | all(. == "string"))' \
    "$response" >/dev/null || die "upload_headers is not a string mapping"
  while IFS= read -r -d '' key && IFS= read -r -d '' value; do
    case "$key$value" in
      *$'\n'* | *$'\r'*) die "upload_headers contains a newline in '$key'" ;;
    esac
    UPLOAD_HEADER_ARGS+=(-H "$key: $value")
  done < <(jq -j '(.upload_headers // {}) | to_entries[] | (.key, "\u0000", (.value | tostring), "\u0000")' "$response")
}

ext_for_format() {
  case "$1" in
    markdown) printf '.md' ;;
    middle_json | structured_content) printf '.json' ;;
    zip) printf '.zip' ;;
    html) printf '.html' ;;
    latex) printf '.tex' ;;
    docx) printf '.docx' ;;
    *) printf '.%s' "$1" ;;
  esac
}

sanitize_name() {
  printf '%s' "$1" | tr -c 'A-Za-z0-9._-' '_'
}

print_file_results() {
  jq -r '.files[] | "    \(.name // "?"): \(.status // "?")\(if .error then " (\(.error.code // "error"): \(.error.message // ""))" else "" end)"' \
    "$JOB_RESPONSE" || true
}

# Download the requested artifacts of every completed file. Writes to *.part
# first and moves into place on success, so a half file never looks complete.
# Sets MISSING_ARTIFACTS=1 when a completed file lacks a requested format.
MISSING_ARTIFACTS=0
download_artifacts() {
  local name fmt fid safe dest part
  while IFS=$'\t' read -r name fmt fid; do
    if [ -z "$fid" ]; then
      echo "error: completed file '$name' has no '$fmt' output reference" >&2
      MISSING_ARTIFACTS=1
      continue
    fi
    safe=$(sanitize_name "$name")
    dest="$OUTPUT_DIR/${safe}$(ext_for_format "$fmt")"
    part="$dest.part"
    echo "==> Downloading $fmt -> $dest"
    local download_rc=0
    curl --silent --show-error --fail --location \
      --connect-timeout "$CONNECT_TIMEOUT" --max-time "$TRANSFER_TIMEOUT" \
      ${auth_args[@]+"${auth_args[@]}"} -o "$part" "$BASE_URL/v1/files/$fid/content" || download_rc=$?
    if [ "$download_rc" -ne 0 ]; then
      rm -f -- "$part"
      die "download of $fmt failed (file $fid, curl exit $download_rc)"
    fi
    mv -f -- "$part" "$dest"
  done < <(
    for fmt in ${OUTPUT_FORMATS_CSV//,/ }; do
      jq -r --arg fmt "$fmt" \
        '.files[] | select(.status == "completed") | [(.name // "?"), $fmt, (.output_files[$fmt].file_id // "")] | @tsv' \
        "$JOB_RESPONSE"
    done
  )
}

# ── 1. Create the upload session ────────────────────────────────────────────
INPUT_NAME=$(basename -- "$INPUT_FILE")
INPUT_BYTES=$(file_size "$INPUT_FILE") || die "cannot stat $INPUT_FILE"
INPUT_SHA256=$(file_sha256 "$INPUT_FILE")
case "$INPUT_NAME" in
  *.pdf) INPUT_MIME="application/pdf" ;;
  *.html | *.htm) INPUT_MIME="text/html" ;;
  *.docx) INPUT_MIME="application/vnd.openxmlformats-officedocument.wordprocessingml.document" ;;
  *.doc) INPUT_MIME="application/msword" ;;
  *.pptx) INPUT_MIME="application/vnd.openxmlformats-officedocument.presentationml.presentation" ;;
  *.xlsx) INPUT_MIME="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" ;;
  *.epub) INPUT_MIME="application/epub+zip" ;;
  *) INPUT_MIME="application/octet-stream" ;;
esac

echo "==> Creating upload for $INPUT_NAME ($INPUT_BYTES bytes)"
jq -n \
  --arg filename "$INPUT_NAME" --argjson bytes "$INPUT_BYTES" \
  --arg mime_type "$INPUT_MIME" --arg sha256sum "$INPUT_SHA256" \
  '{filename: $filename, bytes: $bytes, mime_type: $mime_type, purpose: "parse", sha256sum: $sha256sum}' \
  >"$REQUEST_BODY"
curl_json "create upload" "$REQUEST_TIMEOUT" -X POST "$BASE_URL/v1/uploads" \
  ${auth_args[@]+"${auth_args[@]}"} -H "Content-Type: application/json" --data-binary "@$REQUEST_BODY"

UPLOAD_STATUS=$(json_string "$RESPONSE" '.status' 'create upload')
case "$UPLOAD_STATUS" in
  pending | completed) ;;
  *) die "create upload: unsupported status '$UPLOAD_STATUS'" ;;
esac
UPLOAD_ID=$(json_string "$RESPONSE" '.id' 'create upload')

# ── 2. Upload the bytes (pending only) ─────────────────────────────────────
if [ "$UPLOAD_STATUS" = "pending" ]; then
  UPLOAD_URL=$(json_string "$RESPONSE" '.upload_url' 'create upload')
  UPLOAD_METHOD=$(json_string "$RESPONSE" '.upload_method // "PUT"' 'create upload')
  case "$UPLOAD_METHOD" in
    PUT) ;;
    *) die "unsupported upload_method '$UPLOAD_METHOD'" ;;
  esac
  build_upload_header_args "$RESPONSE"
  origin_rc=0
  UPLOAD_URL_ABSOLUTE=$(resolve_upload_url "$BASE_URL" "$UPLOAD_URL") || origin_rc=$?
  [ "$origin_rc" -ne 2 ] || die "service returned an invalid upload_url"
  upload_args=(${UPLOAD_HEADER_ARGS[@]+"${UPLOAD_HEADER_ARGS[@]}"})
  if [ "$origin_rc" -eq 0 ]; then
    upload_args+=(${auth_args[@]+"${auth_args[@]}"})
  fi
  echo "==> Uploading bytes via $UPLOAD_METHOD (same-origin: $([ "$origin_rc" -eq 0 ] && echo yes || echo no))"
  upload_rc=0
  curl --silent --show-error --fail \
    --connect-timeout "$CONNECT_TIMEOUT" --max-time "$TRANSFER_TIMEOUT" \
    -X "$UPLOAD_METHOD" "$UPLOAD_URL_ABSOLUTE" ${upload_args[@]+"${upload_args[@]}"} \
    --data-binary "@$INPUT_FILE" -o /dev/null || upload_rc=$?
  if [ "$upload_rc" -ne 0 ]; then
    die "byte upload failed (curl exit $upload_rc)"
  fi

  # ── 3. Complete the upload to obtain the file id ──────────────────────────
  echo "==> Completing upload $UPLOAD_ID"
  curl_json "complete upload" "$REQUEST_TIMEOUT" -X POST "$BASE_URL/v1/uploads/$UPLOAD_ID/complete" \
    ${auth_args[@]+"${auth_args[@]}"}
  COMPLETE_STATUS=$(json_string "$RESPONSE" '.status' 'complete upload')
  [ "$COMPLETE_STATUS" = "completed" ] || die "complete upload: unexpected status '$COMPLETE_STATUS'"
else
  echo "==> Upload deduplicated; reusing the existing file"
fi
FILE_ID=$(json_string "$RESPONSE" '.file.id' 'upload')

# ── 4. Submit the parse job ────────────────────────────────────────────────
echo "==> Submitting parse job (tier=$TIER, formats=$OUTPUT_FORMATS_CSV)"
jq -n \
  --arg file_id "$FILE_ID" --arg tier "$TIER" \
  --arg formats_csv "$OUTPUT_FORMATS_CSV" --arg page_range "$PAGE_RANGE" \
  '{
    files: [{source: {type: "file_id", file_id: $file_id}}
            + (if $page_range == "" then {} else {page_range: $page_range} end)],
    tier: $tier,
    output_formats: ($formats_csv | split(",") | map(gsub("^ +| +$"; "")))
  }' >"$REQUEST_BODY"
curl_json "create job" "$REQUEST_TIMEOUT" -X POST "$BASE_URL/v1/parse/jobs" \
  ${auth_args[@]+"${auth_args[@]}"} -H "Content-Type: application/json" --data-binary "@$REQUEST_BODY"
JOB_ID=$(json_string "$RESPONSE" '.job_id' 'create job')
JOB_STATUS=$(json_string "$RESPONSE" '.status' 'create job')
job_status_ok "$JOB_STATUS" || die "create job: unsupported status '$JOB_STATUS'"
cp -- "$RESPONSE" "$JOB_RESPONSE"
echo "==> Job $JOB_ID ($JOB_STATUS)"

# ── 5. Poll until a terminal state (bounded by request count) ───────────────
polls=0
while ! is_terminal "$JOB_STATUS"; do
  polls=$((polls + 1))
  if [ "$polls" -gt "$MAX_POLLS" ]; then
    echo "error: polling budget exhausted after $MAX_POLLS attempts; job is still '$JOB_STATUS' and was NOT canceled" >&2
    echo "job_id=$JOB_ID"
    echo "resume with: curl -fsS ${BASE_URL}/v1/parse/jobs/${JOB_ID} -H 'Authorization: Bearer <key>'" >&2
    exit 124
  fi
  sleep "$POLL_INTERVAL"
  curl_json "poll job" "$REQUEST_TIMEOUT" "$BASE_URL/v1/parse/jobs/$JOB_ID" ${auth_args[@]+"${auth_args[@]}"}
  JOB_STATUS=$(json_string "$RESPONSE" '.status' 'poll job')
  job_status_ok "$JOB_STATUS" || die "poll job: unsupported status '$JOB_STATUS'"
  cp -- "$RESPONSE" "$JOB_RESPONSE"
done

echo "==> Job $JOB_ID finished: $JOB_STATUS"
print_file_results

# ── 6. Download the artifacts of completed files ────────────────────────────
case "$JOB_STATUS" in
  completed)
    download_artifacts
    if [ "$MISSING_ARTIFACTS" -ne 0 ]; then
      die "job completed but some requested artifacts were missing"
    fi
    echo "==> Done."
    exit 0
    ;;
  partial)
    # Save the successful artifacts before reporting partial completion.
    download_artifacts
    echo "==> Partial success: some files failed; saved artifacts of the completed files." >&2
    exit 2
    ;;
  failed)
    exit 3
    ;;
  canceled)
    exit 4
    ;;
  *)
    die "unreachable terminal status '$JOB_STATUS'"
    ;;
esac
