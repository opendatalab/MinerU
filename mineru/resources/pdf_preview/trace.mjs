// PDF.js 的溯源层只依赖宿主查看器及公开页视图，状态按文档隔离。
export function installTrace({ viewer, eventBus, container, documentId, messages }) {
    const layers = new Set();
    let entries = [];
    let pageIndices = [];
    let selectedId = "";
    let pending = null;
    let showAll = true;
    const abort = new AbortController();
    const options = { signal: abort.signal };
    const toggle = document.createElement("button");
    toggle.type = "button";
    toggle.id = "trace-toggle";
    toggle.textContent = messages.trace_regions;
    toggle.setAttribute("aria-pressed", "true");
    toggle.hidden = true;
    document.getElementById("toolbar").append(toggle);
    // 以当前父窗口的同源地址发送状态，来源页号始终采用原文索引。
    const send = (message) => parent.postMessage({ ...message, channel: "mineru-trace", documentId }, location.origin);
    // 清除已挂载区域，同时保留 PDF.js 自己管理的画布及批注。
    const clearLayers = () => {
        for (const layer of layers) layer.remove();
        layers.clear();
    };
    // 重绘时从文档状态恢复选中样式，避免依赖已销毁的 DOM 引用。
    const updateSelection = () => {
        for (const layer of layers) {
            layer.classList.toggle("show-all", showAll);
            for (const box of layer.children) box.classList.toggle("selected", box.traceIds.includes(selectedId));
        }
    };
    // 跳页后的页面可能尚未获得真实尺寸，等待对应页完成渲染再定位。
    const finishNavigation = () => {
        if (!pending) return;
        const pageNumber = pageIndices.indexOf(pending.pageIdx) + 1;
        const page = viewer.getPageView(pageNumber - 1);
        const layer = page?.div.querySelector(".mineru-trace-layer");
        const box = [...(layer?.children || [])].find((item) => item.traceIds.includes(pending.id));
        if (!box) return;
        const rect = box.getBoundingClientRect();
        const host = container.getBoundingClientRect();
        container.scrollTo({
            top:
                container.scrollTop +
                rect.top -
                host.top -
                Math.max(16, (container.clientHeight - Math.min(rect.height, container.clientHeight - 32)) / 2)
        });
        pending = null;
    };
    // 同框父子优先具体节点，剩余重叠按面积从小到大排列命中层级。
    const mount = (pageNumber) => {
        const page = viewer.getPageView(pageNumber - 1);
        if (!page?.div.querySelector("canvas")) return;
        for (const layer of layers) if (!layer.isConnected) layers.delete(layer);
        const old = page.div.querySelector(".mineru-trace-layer");
        if (old) {
            layers.delete(old);
            old.remove();
        }
        const candidates = entries.flatMap((entry) =>
            entry.sources
                .filter((source) => source.page_idx === pageIndices[pageNumber - 1])
                .map((source) => ({ entry, bbox: source.bbox }))
        );
        candidates.sort((a, b) => b.entry.depth - a.entry.depth);
        const unique = new Map();
        for (const item of candidates) {
            const key = item.bbox.join(",");
            if (!unique.has(key)) unique.set(key, { ...item, ids: [] });
            unique.get(key).ids.push(item.entry.id);
        }
        const regions = [...unique.values()];
        regions.sort(
            (a, b) => (b.bbox[2] - b.bbox[0]) * (b.bbox[3] - b.bbox[1]) - (a.bbox[2] - a.bbox[0]) * (a.bbox[3] - a.bbox[1])
        );
        const layer = document.createElement("div");
        layer.className = "mineru-trace-layer";
        for (const { entry, bbox, ids } of regions) {
            const box = document.createElement("button");
            box.type = "button";
            box.dataset.traceId = entry.id;
            box.traceIds = ids;
            box.setAttribute("aria-label", `${messages.trace_source} ${pageIndices[pageNumber - 1] + 1}`);
            box.style.left = `${bbox[0] * 100}%`;
            box.style.top = `${bbox[1] * 100}%`;
            box.style.width = `${(bbox[2] - bbox[0]) * 100}%`;
            box.style.height = `${(bbox[3] - bbox[1]) * 100}%`;
            box.addEventListener("click", () => send({ type: "select", id: entry.id, pageIdx: pageIndices[pageNumber - 1] }));
            layer.append(box);
        }
        page.div.append(layer);
        layers.add(layer);
        updateSelection();
        finishNavigation();
    };
    // 仅接受当前宿主、当前产物的结构化消息。
    window.addEventListener(
        "message",
        (event) => {
            const data = event.data;
            if (
                event.source !== parent ||
                event.origin !== location.origin ||
                data?.channel !== "mineru-trace" ||
                data.documentId !== documentId
            )
                return;
            if (data.type === "clear") {
                entries = [];
                selectedId = "";
                pending = null;
                clearLayers();
                toggle.hidden = true;
            } else if (data.type === "init") {
                if (!Array.isArray(data.entries) || !Array.isArray(data.pageIndices)) return;
                entries = data.entries.filter(
                    (entry) =>
                        typeof entry.id === "string" &&
                        Array.isArray(entry.sources) &&
                        entry.sources.every(
                            (source) =>
                                Number.isInteger(source.page_idx) &&
                                Array.isArray(source.bbox) &&
                                source.bbox.length === 4 &&
                                source.bbox.every((value) => Number.isFinite(value) && value >= 0 && value <= 1)
                        )
                );
                pageIndices = data.pageIndices;
                toggle.hidden = !entries.length;
                for (const page of viewer.getCachedPageViews()) mount(page.id);
            } else if (data.type === "select" && entries.some((entry) => entry.id === data.id)) {
                pending = null;
                selectedId = data.id;
                updateSelection();
                if (data.scroll) {
                    const pageNumber = pageIndices.indexOf(data.pageIdx) + 1;
                    if (pageNumber < 1) return;
                    pending = data;
                    viewer.scrollPageIntoView({ pageNumber });
                    finishNavigation();
                }
            }
        },
        options
    );
    toggle.addEventListener(
        "click",
        () => {
            showAll = !showAll;
            toggle.setAttribute("aria-pressed", String(showAll));
            updateSelection();
        },
        options
    );
    // 查看器事件在缩放和缓存重建后重新挂载区域。
    const onRendered = (event) => {
        if (!event.error) mount(event.pageNumber);
    };
    eventBus.on("pagerendered", onRendered);
    send({ type: "ready" });
    return () => {
        // 查看器卸载时释放本次文档拥有的监听器和浮层。
        abort.abort();
        eventBus.off("pagerendered", onRendered);
        clearLayers();
        toggle.remove();
    };
}
