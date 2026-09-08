// 查看器独立于宿主框架运行；PDF、字体和解码器只从同源的固定资源目录读取。
const params = new URLSearchParams(window.location.hash.slice(1));
const messages = JSON.parse(params.get("messages") || "{}");
const status = document.getElementById("status");
const container = document.getElementById("viewer-container");
const pageInput = document.getElementById("page");
const controls = Object.fromEntries(["previous", "next", "zoom-out", "zoom-in", "fit"].map((id) => [id, document.getElementById(id)]));
// 文案由 MinerU 的唯一词典传入，只写入文本节点与无副作用的无障碍属性。
const text = (name) => messages[name] || name;
document.title = text("preview");
document.getElementById("toolbar").setAttribute("aria-label", text("preview"));
document.getElementById("page-label").textContent = text("pdf_page");
for (const [id, name] of Object.entries({ previous: "pdf_previous", next: "pdf_next", "zoom-out": "pdf_zoom_out", "zoom-in": "pdf_zoom_in", fit: "pdf_fit_width" })) {
    controls[id].title = text(name);
    controls[id].setAttribute("aria-label", text(name));
}
controls.fit.textContent = text("pdf_fit_width");
status.textContent = text("pdf_loading");

let disposed = false;
let loadingTask;
let viewer;
let links;
let resizeObserver;
let resizeFrame;
let fitWidth = true;
let ready = false;

// 清除或换文件时释放渲染队列、worker 和观察器，旧页面不会向新预览写入结果。
const dispose = () => {
    disposed = true;
    cancelAnimationFrame(resizeFrame);
    resizeObserver?.disconnect();
    viewer?.setDocument(null);
    links?.setDocument(null);
    loadingTask?.destroy().catch(() => {});
};
window.addEventListener("pagehide", dispose, { once: true });

// 控件始终反映当前实际阅读页，缩放与翻页都只操作浏览器内的 PDF.js。
const updateControls = () => {
    if (disposed || !viewer || !ready) return;
    pageInput.disabled = false;
    pageInput.value = String(viewer.currentPageNumber);
    pageInput.max = String(viewer.pagesCount);
    document.getElementById("pages").textContent = String(viewer.pagesCount);
    document.getElementById("scale").textContent = `${Math.round(viewer.currentScale * 100)}%`;
    controls.previous.disabled = viewer.currentPageNumber <= 1;
    controls.next.disabled = viewer.currentPageNumber >= viewer.pagesCount;
    controls["zoom-out"].disabled = viewer.currentScale <= .25;
    controls["zoom-in"].disabled = viewer.currentScale >= 4;
    controls.fit.disabled = false;
};

// 首次加载和容器宽度变化时按可用宽度布局，保留 PDF.js 的可见页懒渲染与缓存回收。
const resize = () => {
    if (disposed || !viewer || !ready) return;
    if (fitWidth) viewer.currentScaleValue = "page-width";
    viewer.update();
    updateControls();
};

// 用显式边界处理跳页，空值或非整数输入恢复当前页，避免产生无效渲染任务。
const goToPage = (value) => {
    if (!ready || disposed) return;
    const page = Number(value);
    if (Number.isInteger(page) && page >= 1 && page <= viewer.pagesCount) viewer.currentPageNumber = page;
    updateControls();
};

// 在支持区间内调整比例，手动缩放后不被容器尺寸观察器覆盖。
const zoom = (delta) => {
    if (!ready || disposed) return;
    fitWidth = false;
    viewer.currentScaleValue = String(Math.min(4, Math.max(.25, viewer.currentScale + delta)));
    updateControls();
};

// 模块和 worker 使用同一份固定发行包，不启用 PDF 内嵌脚本、编辑或交互表单。
try {
    const source = new URL(params.get("file") || "", window.location.href);
    if (!params.get("file") || source.origin !== window.location.origin || !source.pathname.includes("/gradio_api/file=")) {
        throw new Error("Invalid PDF source");
    }
    const pdfjs = await import("./vendor/pdfjs/legacy/build/pdf.min.mjs");
    globalThis.pdfjsLib = pdfjs;
    const { EventBus, PDFViewer, PDFLinkService } = await import("./vendor/pdfjs/legacy/web/pdf_viewer.mjs");
    if (!disposed) {
        pdfjs.GlobalWorkerOptions.workerSrc = new URL("./vendor/pdfjs/legacy/build/pdf.worker.min.mjs", import.meta.url).href;
        const eventBus = new EventBus();
        links = new PDFLinkService({ eventBus, externalLinkEnabled: false });
        // 同步移除页面边框和适应宽度时的额外预留，避免左右再次产生固定空隙。
        viewer = new PDFViewer({ container, eventBus, linkService: links, textLayerMode: 0, annotationMode: 1, annotationEditorMode: -1, removePageBorders: true });
        links.setViewer(viewer);
        eventBus.on("pagesinit", () => {
            ready = true;
            document.getElementById("toolbar").setAttribute("aria-busy", "false");
            resize();
        });
        eventBus.on("pagechanging", updateControls);
        eventBus.on("scalechanging", updateControls);
        eventBus.on("pagerendered", (event) => {
            if (!disposed) {
                status.textContent = event.error ? text("pdf_load_failed") : "";
                document.body.dataset.rendered = event.error ? "error" : "true";
            }
        });
        controls.previous.addEventListener("click", () => goToPage(viewer.currentPageNumber - 1));
        controls.next.addEventListener("click", () => goToPage(viewer.currentPageNumber + 1));
        pageInput.addEventListener("change", () => goToPage(pageInput.value));
        pageInput.addEventListener("keydown", (event) => { if (event.key === "Enter") goToPage(pageInput.value); });
        controls["zoom-out"].addEventListener("click", () => zoom(-.25));
        controls["zoom-in"].addEventListener("click", () => zoom(.25));
        controls.fit.addEventListener("click", () => { fitWidth = true; resize(); });
        resizeObserver = new ResizeObserver(() => {
            cancelAnimationFrame(resizeFrame);
            resizeFrame = requestAnimationFrame(resize);
        });
        resizeObserver.observe(container);
        loadingTask = pdfjs.getDocument({
            url: source.href,
            withCredentials: true,
            cMapUrl: new URL("./vendor/pdfjs/cmaps/", import.meta.url).href,
            cMapPacked: true,
            standardFontDataUrl: new URL("./vendor/pdfjs/standard_fonts/", import.meta.url).href,
            wasmUrl: new URL("./vendor/pdfjs/wasm/", import.meta.url).href,
            isEvalSupported: false,
            enableXfa: false,
        });
        const pdf = await loadingTask.promise;
        if (!disposed) {
            viewer.setDocument(pdf);
            links.setDocument(pdf);
        }
    }
} catch (error) {
    if (!disposed) {
        const name = error?.name;
        status.textContent = text(name === "PasswordException" ? "pdf_password" : name === "InvalidPDFException" ? "pdf_invalid" : "pdf_load_failed");
        status.setAttribute("role", "alert");
        document.body.dataset.rendered = "error";
        document.getElementById("toolbar").setAttribute("aria-busy", "false");
    }
}
