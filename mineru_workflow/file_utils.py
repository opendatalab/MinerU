import json
import fcntl
import shutil
import re
from pathlib import Path
from datetime import datetime

class AtomicJsonFile:
    """Provides atomic read/write behavior for JSON array and JSONL files using fcntl file locks."""

    @staticmethod
    def append_jsonl(file_path: Path, record: dict):
        """Atomically appends a JSON record as a newline (JSONL) to the file."""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "a", encoding="utf-8") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            fcntl.flock(f, fcntl.LOCK_UN)

    @staticmethod
    def read_json_dict(file_path: Path) -> dict:
        """Reads a JSON file storing a dict (like PROCESSING.json). Atomically reads it."""
        if not file_path.exists():
            return {}
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                fcntl.flock(f, fcntl.LOCK_SH)
                data = json.load(f)
                fcntl.flock(f, fcntl.LOCK_UN)
            return data
        except json.JSONDecodeError:
            return {}

    @staticmethod
    def write_json_dict(file_path: Path, data: dict):
        """Atomically overwrite a JSON file storing a dict."""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.flush()
            fcntl.flock(f, fcntl.LOCK_UN)


def safe_move_file(src_path: Path, dest_dir: Path) -> Path:
    """Moves a file safely to the destination directory, handling filename collisions."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / src_path.name
    
    if dest_path.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dest_path = dest_dir / f"{src_path.stem}_{timestamp}{src_path.suffix}"
    
    shutil.move(str(src_path), str(dest_path))
    return dest_path


def get_current_iso_time() -> str:
    """Returns the current time in ISO 8601 format."""
    return datetime.now().astimezone().isoformat()


def sanitize_filename(filename: str) -> str:
    """
    Remove unsafe filesystem characters.
    Normalize repeated spaces, punctuation, and separators.
    """
    # Remove chars unsafe for windows/linux
    unsafe_chars_regex = r'[\<\>\:\"\/\\\|\?\*]'
    clean = re.sub(unsafe_chars_regex, '', filename)
    # Replace multiple spaces/newlines/tabs with single space
    clean = re.sub(r'\s+', ' ', clean)
    # Replace weird double dashes/underscores if desired, optional
    clean = re.sub(r' +', ' ', clean).strip()
    return clean

def extract_title_case(title: str) -> str:
    """
    Converts to Title Case while preserving ALL CAPS acronyms.
    """
    words = title.split()
    title_words = []
    for word in words:
        if word.isupper() and bool(re.search(f"[A-Z]", word)):
            title_words.append(word)
        else:
            title_words.append(word.capitalize())
    return " ".join(title_words)


def extract_date(text: str) -> str:
    """
    Attempt to find a sensible YYYY-MM-DD from the text.
    Fallback: returns empty if not confident.
    """
    # Matches YYYY-MM-DD, or YYYY/MM/DD, or DD-MM-YYYY
    # Simplest approach is strictly enforcing YYYY-MM-DD logic for output.
    # We will search for 4 digits (19xx or 20xx), a dash, 2 digits, a dash, 2 digits.
    match = re.search(r'(20\d{2}|19\d{2})[-/](0[1-9]|1[0-2])[-/](0[1-9]|[12]\d|3[01])', text)
    if match:
        return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
    return ""

def format_human_readable_title(raw_title: str, text_content: str = "", original_filename: str = "") -> str:
    """
    Generates a human-readable document title based on rules:
    - Title Case
    - ALL CAPS preserved
    - Sanitize chars
    - Append date
    """
    title_to_use = raw_title
    
    # Try to extract a title from the markdown content
    if not title_to_use and text_content:
        # Boilerplate/Generic headers to skip (normalized lowercase)
        generic_headers = {
            "contents", "abstract", "introduction", "background", 
            "electronic document", "summary", "title", "warning",
            "document citation", "executive summary", "author", 
            "source", "regulatory compliance", "cost", "sovereignty",
            "ai factories", "deviance", "the blame game",
            "strategies for learning from failure", "amy c. edmondson",
            "productizing ai", "enterprise guide", "table of contents",
            "what is ai infrastructure"
        }
        
        # Split text into lines to process
        lines = [line.strip() for line in text_content.split('\n') if line.strip()]
        
        # Strategy 1: Look for labels like "Title:" followed by a value (on same or next line)
        for i, line in enumerate(lines[:30]):
            lower_line = line.lower()
            if lower_line == "title:":
                if i + 1 < len(lines):
                    potential = lines[i+1].strip()
                    if potential and potential.lower() not in generic_headers:
                        title_to_use = potential
                        break
            elif lower_line.startswith("title:"):
                potential = line[6:].strip()
                if potential and potential.lower() not in generic_headers:
                    title_to_use = potential
                    break
        
        # Strategy 2: Look for multiple consecutive non-generic lines at the very START
        # Many PDF extractors put the title at the top but don't mark it as an H1.
        if not title_to_use:
            candidate_lines = []
            for i, line in enumerate(lines[:10]): # Only first 10 meaningful lines
                clean_line = line.strip()
                lower_line = clean_line.lower()
                
                # If we hit an H1, treat it as a stop or a candidate if nothing else
                if clean_line.startswith('#'):
                    # If it's a generic H1, skip it and continue
                    if clean_line[1:].strip().lower() in generic_headers:
                        if candidate_lines: break
                        continue
                    # If it's a real H1 and we have no candidate yet, use it
                    if not candidate_lines:
                        title_to_use = clean_line[1:].strip()
                    break
                
                # Skip if it's a label or generic
                if clean_line.endswith(':') or lower_line in generic_headers:
                    if candidate_lines: break
                    continue
                
                # Skip small noise/images
                if len(clean_line) < 3 or clean_line.startswith('!['):
                    continue
                
                # If it looks like a title part (not a long paragraph)
                if len(clean_line.split()) < 40:
                    candidate_lines.append(clean_line)
                else:
                    break # it's a paragraph
            
            if not title_to_use and candidate_lines:
                temp_title = " ".join(candidate_lines)
                if len(temp_title) > 5:
                    title_to_use = temp_title

        # Strategy 3: Find the first non-generic H1 (# Heading) anywhere in the first 50 lines
        if not title_to_use:
            for line in lines[:50]:
                if line.startswith('# '):
                    potential = line[2:].strip()
                    if potential.lower() not in generic_headers and len(potential) > 5:
                        title_to_use = potential
                        break

    if not title_to_use:
        title_to_use = original_filename or "Document"
    
    # Clean up trailing punctuation and whitespace
    if title_to_use:
        title_to_use = title_to_use.strip().rstrip('.,:;-_')
        
    title_to_use = sanitize_filename(title_to_use)
    # Strip extension if present
    if title_to_use.lower().endswith(".pdf"):
        title_to_use = title_to_use[:-4]
        
    proper_title = extract_title_case(title_to_use)
    
    # Try finding date in textual content first, fallback to title search.
    date_str = extract_date(text_content)
    if not date_str:
        date_str = extract_date(original_filename)
        
    # Reasonable length truncation (100 chars max)
    if len(proper_title) > 100:
        proper_title = proper_title[:100].strip()
        
    if date_str:
        return f"{proper_title} {date_str}"
    else:
        return proper_title
