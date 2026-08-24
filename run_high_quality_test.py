import os
import subprocess
import argparse
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description="Run MinerU (magic-pdf) with maximum quality settings for high-end GPU machines."
    )
    parser.add_argument(
        "-p", "--path", 
        required=True, 
        help="Path to a PDF, image, DOCX, PPTX, or XLSX file, or a directory."
    )
    parser.add_argument(
        "-o", "--output", 
        required=True, 
        help="Directory to save the extraction results."
    )
    parser.add_argument(
        "-l", "--lang", 
        default="ch", 
        help="Languages in the document (e.g., ch, english, korean, etc.). Default: ch."
    )
    args = parser.parse_args()

    input_path = os.path.abspath(args.path)
    output_dir = os.path.abspath(args.output)
    
    # Ensure output dir exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # MinerU High Quality Configuration for GPU
    # -m auto             : Automatically select the best method (txt/ocr) based on the file.
    # -b hybrid-engine    : Use the next-generation hybrid engine which leverages local compute for maximum accuracy.
    # --effort high       : Enables high-effort parsing, which includes detailed image/chart analysis.
    # -f True             : Enable formula parsing.
    # -t True             : Enable table parsing.
    # --image-analysis True : Enable image and chart analysis.
    # --client-side-output-generation True : Generate detailed markdown and viewer outputs from JSON.
    
    cmd = [
        sys.executable, "-m", "mineru.cli.client",
        "-p", input_path,
        "-o", output_dir,
        "-m", "auto",
        "-b", "hybrid-engine",
        "--effort", "high",
        "-f", "True",
        "-t", "True",
        "--image-analysis", "True",
        "--client-side-output-generation", "True",
        "-l", args.lang
    ]

    print("======================================================")
    print("🚀 Starting MinerU High-Quality Extraction Test")
    print("======================================================")
    print(f"Input : {input_path}")
    print(f"Output: {output_dir}")
    print("Configuration:")
    print(" - Method         : auto (Best method based on document type)")
    print(" - Backend        : hybrid-engine (Max accuracy via local models)")
    print(" - Effort         : high (Detailed image/chart analysis)")
    print(" - Formula Parsing: Enabled")
    print(" - Table Parsing  : Enabled")
    print(" - Image Analysis : Enabled")
    print(" - Output Gen     : Enabled (Markdown, Content lists, including color styles)")
    print(" - Language       : " + args.lang)
    print("======================================================")
    print("Executing command:")
    print(" ".join(cmd))
    print("======================================================\n")

    try:
        # Run the extraction command
        subprocess.run(cmd, check=True)
        print("\n✅ Extraction completed successfully!")
        print(f"Check the results in: {output_dir}")
    except subprocess.CalledProcessError as e:
        print("\n❌ Error occurred during extraction!")
        print(f"Command failed with exit code: {e.returncode}")
        sys.exit(e.returncode)
    except FileNotFoundError:
        print("\n❌ Error: MinerU module not found. Make sure the virtual environment is active and mineru is installed.")
        sys.exit(1)

if __name__ == "__main__":
    main()
