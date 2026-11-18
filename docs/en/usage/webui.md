# Parse and review documents in the WebUI

Open the [online demo](https://huggingface.co/spaces/opendatalab/MinerU), or install MinerU and start the local interface from a terminal.

```bash
mineru-kit webui
```

Open the address printed in the terminal. By default the WebUI starts a local Router and one parsing worker. To submit files to an existing V1 service, run `mineru-kit webui --api-url http://127.0.0.1:8000`.

## Upload and select pages {#upload}

1. Choose or drop a file in the upload area, or use **Paste** for a file or image on your clipboard. You can start reading a local PDF as soon as you select it.
2. For a PDF or image, choose one of the service's available tiers. Native office documents, web pages, and other native formats use Flash and are parsed in full.
3. To parse consecutive PDF pages, drag the two ends of the page range. Page numbers start at 1 and include both endpoints. All pages are selected initially, up to any `--max-pages` limit set by the deployment. Turning a preview page only changes what you see.
4. Click **Convert**. MinerU automatically decides whether a PDF needs OCR. To ignore the text layer and recognize the pages again, turn on **Force OCR** before conversion.

## Check the result against the source {#source-tracing}

Open the **Markdown** tab to review the formatted result. Use **Contents** to jump to a heading. For PDFs and images, click a region in the source to highlight its parsed content. Click a paragraph, heading, formula, or figure with a source location to return to that region. For a paragraph or table merged across pages, use **Source page** to switch between its source pages.

Region boxes are drawn over the source preview. Even after parsing a selection, you can still browse the full PDF using its original page numbers.

## Download the result {#downloads}

Open **Download** in the results pane and choose a format. Markdown and JSON each download as a separate ZIP containing the content and its `images/` assets. The **JSON** tab and its download contain Structured Content. For Middle JSON, use the Tasks `default` bundle or `mineru-kit parse --format middle_json`.

The menu also offers HTML, DOCX, LaTeX, EPUB, and PDF. LaTeX includes the `.tex` file and images. The PDF is rendered from the parsed result for sharing. Standalone HTML references images served by the running WebUI. For an offline copy with its images, choose the Markdown ZIP. [Output formats by interface](../reference/output_files.md)
