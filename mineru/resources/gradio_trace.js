// 父页面协调两个预览，产物身份与转换 ticket 分别管理。
() => {
    if (window.__mineruTrace) return;
    let documentId = "";
    let entries = [];
    let initialPageIdx = null;
    let sourceId = "";
    let htmlReady = false;
    let layoutRegions = [];
    let pageSyncEnabled = true;
    let hasPageSync = false;
    let entriesById = new Map();
    // 合并正文中的多个同页来源仍用完整物理身份区分。
    const sourceKey = (source) => JSON.stringify([source.page_idx, source.block_index, source.child_path]);
    // 只向当前自有 iframe 发消息，隔离的 HTML 使用明确窗口引用。
    const send = (selector, message) => {
        const frame = document.querySelector(selector);
        frame?.contentWindow?.postMessage(
            { ...message, channel: "mineru-trace", documentId, sourceId },
            selector.includes("rendered") ? "*" : location.origin
        );
    };
    // 初始化可重复执行，覆盖两侧加载次序和 iframe 重挂载。
    const initialize = () => {
        send("iframe.mineru-pdf-frame", { type: "bind" });
        if (htmlReady) send("iframe.mineru-pdf-frame", {
            type: "init", entries, initialPageIdx, layoutRegions, pageSync: hasPageSync ? pageSyncEnabled : null
        });
    };
    window.__mineruTrace = {
        // 切换或重提立即使旧来源和待定位请求失效。
        reset() {
            send("iframe.mineru-pdf-frame", { type: "clear" });
            documentId = "";
            entries = [];
            entriesById.clear();
            hasPageSync = false;
            initialPageIdx = null;
            htmlReady = false;
            layoutRegions = [];
        },
        // 只在已接受的结果回执进入预览时绑定产物标识。
        activate(id, currentSourceId) {
            if (documentId === id && sourceId === currentSourceId) return;
            this.reset();
            documentId = id;
            sourceId = currentSourceId;
            initialize();
            // 正文可能先于结果回执完成加载，主动握手补回已错过的 ready。
            send("iframe.mineru-rendered-html-frame", { type: "request-ready" });
        }
    };
    window.addEventListener("message", (event) => {
        const data = event.data;
        if (!documentId || data?.channel !== "mineru-trace") return;
        const htmlFrame = document.querySelector("iframe.mineru-rendered-html-frame");
        const pdfFrame = document.querySelector("iframe.mineru-pdf-frame");
        const fromHtml = event.source === htmlFrame?.contentWindow;
        const fromPdf = event.source === pdfFrame?.contentWindow && event.origin === location.origin;
        if (!fromHtml && !fromPdf) return;
        if (fromPdf && (data.sourceId !== sourceId || (data.type !== "ready" && data.documentId !== documentId))) return;
        if (fromHtml && data.documentId !== documentId) return;
        if (data.type === "ready") {
            if (fromHtml) {
                if (!Array.isArray(data.entries)) return;
                entries = data.entries;
                entriesById = new Map(entries.map(entry => [entry.id, entry]));
                hasPageSync = data.pdfPageSync === true && entries.length > 0;
                initialPageIdx = data.initialPageIdx;
                htmlReady = true;
                layoutRegions = Array.isArray(data.layoutRegions) ? data.layoutRegions : [];
                send("iframe.mineru-rendered-html-frame", {
                    type: "labels",
                    pageSync: hasPageSync && pageSyncEnabled,
                    pageLabel: window.__mineruI18n.text("trace_source")
                });
            }
            initialize();
            return;
        }
        if (fromPdf && hasPageSync && data.type === "toggle-page-sync" && typeof data.enabled === "boolean") {
            pageSyncEnabled = data.enabled;
            const mode = { type: "page-sync-mode", pageSync: pageSyncEnabled };
            send("iframe.mineru-pdf-frame", mode);
            send("iframe.mineru-rendered-html-frame", mode);
            return;
        }
        if (data.type !== "select") {
            if (!hasPageSync || !pageSyncEnabled) return;
            const visible = Boolean(htmlFrame?.getClientRects().length);
            if (fromPdf && data.type === "pdf-page" && Number.isInteger(data.pageIdx)) {
                // 隐藏正文时只释放旧阅读页，返回 Markdown 后的第一次滚动即可重新接管。
                send("iframe.mineru-rendered-html-frame", { type: "pdf-page", pageIdx: data.pageIdx, scroll: visible });
                return;
            }
            if (!fromHtml || !visible) return;
            if (data.type === "html-page") {
                const entry = entriesById.get(data.id);
                if (entry) send("iframe.mineru-pdf-frame", { type: "html-page", pages: [...new Set(entry.sources.map(s => s.page_idx))] });
                return;
            }
            // 显式目录跳转沿用点击溯源，两栏共同选中标题，目录自行对齐正文。
            if (data.type !== "outline") return;
        }
        const entry = entriesById.get(data.id);
        if (!entry) return;
        const source = entry.sources.find((item) => sourceKey(item) === data.sourceKey)
            || entry.sources.find((item) => item.page_idx === data.pageIdx) || entry.sources[0];
        if (!source) return;
        const selection = { type: "select", id: entry.id, pageIdx: source.page_idx, sourceKey: sourceKey(source) };
        send("iframe.mineru-rendered-html-frame", { ...selection, scroll: fromPdf });
        send("iframe.mineru-pdf-frame", { ...selection, scroll: fromHtml });
    });
}
