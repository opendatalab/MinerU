// 父页面协调两个预览，产物身份与转换 ticket 分别管理。
() => {
    if (window.__mineruTrace) return;
    let documentId = "";
    let entries = [];
    let pageIndices = [];
    let layoutRegions = [];
    // 合并正文中的多个同页来源仍用完整物理身份区分。
    const sourceKey = (source) => JSON.stringify([source.page_idx, source.block_index, source.child_path]);
    // 只向当前自有 iframe 发消息，隔离的 HTML 使用明确窗口引用。
    const send = (selector, message) => {
        const frame = document.querySelector(selector);
        frame?.contentWindow?.postMessage(
            { ...message, channel: "mineru-trace", documentId },
            selector.includes("rendered") ? "*" : location.origin
        );
    };
    // 初始化可重复执行，覆盖两侧加载次序和 iframe 重挂载。
    const initialize = () => {
        send("iframe.mineru-pdf-frame", { type: "init", entries, pageIndices, layoutRegions });
    };
    window.__mineruTrace = {
        // 切换或重提立即使旧来源和待定位请求失效。
        reset() {
            send("iframe.mineru-pdf-frame", { type: "clear" });
            documentId = "";
            entries = [];
            pageIndices = [];
            layoutRegions = [];
        },
        // 只在已接受的结果回执进入预览时绑定产物标识。
        activate(id) {
            if (documentId === id) return;
            this.reset();
            documentId = id;
        }
    };
    window.addEventListener("message", (event) => {
        const data = event.data;
        if (!documentId || data?.channel !== "mineru-trace" || data.documentId !== documentId) return;
        const htmlFrame = document.querySelector("iframe.mineru-rendered-html-frame");
        const pdfFrame = document.querySelector("iframe.mineru-pdf-frame");
        const fromHtml = event.source === htmlFrame?.contentWindow;
        const fromPdf = event.source === pdfFrame?.contentWindow && event.origin === location.origin;
        if (!fromHtml && !fromPdf) return;
        if (data.type === "ready") {
            if (fromHtml) {
                if (!Array.isArray(data.entries) || !Array.isArray(data.pageIndices)) return;
                entries = data.entries;
                pageIndices = data.pageIndices;
                layoutRegions = Array.isArray(data.layoutRegions) ? data.layoutRegions : [];
                send("iframe.mineru-rendered-html-frame", {
                    type: "labels",
                    pageLabel: window.__mineruI18n.text("trace_source")
                });
            }
            initialize();
            return;
        }
        if (data.type !== "select") return;
        const entry = entries.find((item) => item.id === data.id);
        if (!entry) return;
        const source = entry.sources.find((item) => sourceKey(item) === data.sourceKey)
            || entry.sources.find((item) => item.page_idx === data.pageIdx) || entry.sources[0];
        if (!source) return;
        const selection = { type: "select", id: entry.id, pageIdx: source.page_idx, sourceKey: sourceKey(source) };
        send("iframe.mineru-rendered-html-frame", { ...selection, scroll: fromPdf });
        send("iframe.mineru-pdf-frame", { ...selection, scroll: fromHtml });
    });
}
