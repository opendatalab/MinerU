// PDF.js 的溯源层只依赖宿主查看器及公开页视图，状态按文档隔离。
export function installTrace({ viewer, eventBus, container, documentId, messages }) {
    const layers = new Map();
    let entries = [];
    let sourcesByPage = new Map();
    let layoutByPage = new Map();
    let layoutBySource = new Map();
    let layoutByBox = new Map();
    let traceableSources = new Set();
    let mergedEntries = new Map();
    let pageIndices = [];
    let pageNumbers = new Map();
    let selectedId = "";
    let selectedSourceKey = "";
    let hoveredBox = null;
    let focusedBox = null;
    let pending = null;
    let showAll = true;
    const abort = new AbortController();
    const options = { signal: abort.signal };
    let connectionFrame = 0;
    // SVG 独立覆盖滚动视口，跨页线段可穿过页面边界，且不会随 PDF 缩放改变笔画粗细。
    const svgNode = (name, attributes = {}) => {
        const node = document.createElementNS("http://www.w3.org/2000/svg", name);
        for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, String(value));
        return node;
    };
    const connections = svgNode("svg", { class: "mineru-merge-layer", "aria-hidden": "true" });
    document.body.append(connections);
    const toggle = document.createElement("button");
    toggle.type = "button";
    toggle.id = "trace-toggle";
    toggle.textContent = messages.trace_regions;
    toggle.setAttribute("aria-pressed", "true");
    toggle.hidden = true;
    document.getElementById("toolbar").append(toggle);
    // 普通调用语句保留当前调用帧，避免 WebKit 在严格模式尾调用中把消息来源归到父窗口。
    const send = (message) => {
        parent.postMessage({ ...message, channel: "mineru-trace", documentId }, location.origin);
    };
    // 物理身份包含原始页、顶层块号和完整子路径，跨页合并后仍能精确关联。
    const sourceKey = (source) => JSON.stringify([source.page_idx, source.block_index, source.child_path]);
    // 视觉父块的物理区域由主体子块表示，同框查找限定在同一个原始顶层块。
    const boxKey = (source) => JSON.stringify([source.page_idx, source.block_index, source.bbox]);
    // 优先取精确来源，再解析视觉父块与主体子块共享的几何位置。
    const layoutRegion = (source) => layoutBySource.get(sourceKey(source)) || layoutByBox.get(boxKey(source));
    // 两侧预览都由共享 RGB 数值构造类型色，浅色类型的深色阶由 CSS 统一计算。
    const themeColor = (region) => {
        const color = region?.color || [0.35, 0.4, 0.45];
        return `rgb(${color.map(value => Math.round(value * 255)).join(" ")})`;
    };
    // 同框父子按现有命中顺序优先取最具体的合并节点，单来源块暂时隐藏已选中的连接。
    const activeMerge = () => {
        const inspected = [hoveredBox, focusedBox].find(box => box?.isConnected);
        if (inspected) return inspected.traceIds.map(id => mergedEntries.get(id)).find(Boolean);
        return mergedEntries.get(selectedId);
    };
    // 读取真实页框而非缓存的命中元素，页面画布被回收后仍能使用已加载的页面尺寸。
    const sourceRect = (source, host) => {
        const pageNumber = pageNumbers.get(source.page_idx);
        const page = pageNumber === undefined ? null : viewer.getPageView(pageNumber - 1);
        const bbox = source.bbox;
        if (!page?.pdfPage || !page.div.isConnected || bbox[2] <= bbox[0] || bbox[3] <= bbox[1]) return null;
        const rect = page.div.getBoundingClientRect();
        if (!rect.width || !rect.height) return null;
        return {
            left: rect.left - host.left + bbox[0] * rect.width,
            top: rect.top - host.top + bbox[1] * rect.height,
            right: rect.left - host.left + bbox[2] * rect.width,
            bottom: rect.top - host.top + bbox[3] * rect.height
        };
    };
    // 每段固定使用前块右下角与后块左上角，来源次序直接来自 renderer 的实际合并结果。
    const drawConnections = () => {
        connectionFrame = 0;
        connections.replaceChildren();
        const entry = activeMerge();
        connections.dataset.traceId = entry?.id || "";
        if (!entry) return;
        const bounds = container.getBoundingClientRect();
        const host = { left: bounds.left + container.clientLeft, top: bounds.top + container.clientTop };
        const width = container.clientWidth;
        const height = container.clientHeight;
        Object.assign(connections.style, {
            left: `${host.left}px`, top: `${host.top}px`, width: `${width}px`, height: `${height}px`
        });
        const rects = entry.sources.map(source => sourceRect(source, host));
        for (let index = 1; index < rects.length; index++) {
            const previous = rects[index - 1];
            const current = rects[index];
            if (!previous || !current) continue;
            const x1 = previous.right;
            const y1 = previous.bottom;
            const x2 = current.left;
            const y2 = current.top;
            // 整段在视口外时省略 DOM，跨越视口的线段由 SVG 边界自然裁剪。
            if (Math.max(x1, x2) < 0 || Math.min(x1, x2) > width ||
                Math.max(y1, y2) < 0 || Math.min(y1, y2) > height) continue;
            const group = svgNode("g");
            group.dataset.fromSource = sourceKey(entry.sources[index - 1]);
            group.dataset.toSource = sourceKey(entry.sources[index]);
            group.style.setProperty("--trace-color", themeColor(layoutRegion(entry.sources[index - 1])));
            group.append(
                svgNode("line", { x1, y1, x2, y2 }),
                svgNode("circle", { cx: x1, cy: y1, r: 2.5 }),
                svgNode("circle", { cx: x2, cy: y2, r: 2.5 })
            );
            connections.append(group);
            const x = (x1 + x2) / 2;
            const y = (y1 + y2) / 2;
            const label = svgNode("text", { x, y, "text-anchor": "middle", "dominant-baseline": "central" });
            label.textContent = messages.trace_merged;
            group.append(label);
            const labelWidth = label.getComputedTextLength() + 8;
            // 短线保留端点和无障碍说明，标签只在能完整容纳的中点显示。
            if (Math.hypot(x2 - x1, y2 - y1) < labelWidth + 12 ||
                x - labelWidth / 2 < 0 || x + labelWidth / 2 > width || y < 9 || y > height - 9) {
                label.remove();
                continue;
            }
            group.insertBefore(svgNode("rect", {
                x: x - labelWidth / 2, y: y - 9, width: labelWidth, height: 18, rx: 3
            }), label);
        }
    };
    // 一帧内的悬停、滚动、缩放和 PDF.js 重绘共用一次几何测量。
    const scheduleConnections = () => {
        if (!connectionFrame) connectionFrame = requestAnimationFrame(drawConnections);
    };
    // 失效立即清空旧文档的连接，并撤销尚未执行的绘制。
    const clearConnections = () => {
        cancelAnimationFrame(connectionFrame);
        connectionFrame = 0;
        connections.replaceChildren();
        connections.dataset.traceId = "";
    };
    // 两种浮层统一使用显示方向下的归一化坐标。
    const placeRegion = (node, bbox) => {
        node.style.left = `${bbox[0] * 100}%`;
        node.style.top = `${bbox[1] * 100}%`;
        node.style.width = `${(bbox[2] - bbox[0]) * 100}%`;
        node.style.height = `${(bbox[3] - bbox[1]) * 100}%`;
    };
    // 初始化时按页建索引，翻页和重绘只扫描当前页的来源与布局。
    const indexRegions = (regions) => {
        sourcesByPage = new Map();
        layoutByPage = new Map();
        layoutBySource = new Map();
        layoutByBox = new Map();
        traceableSources = new Set();
        mergedEntries = new Map();
        pageNumbers = new Map(pageIndices.map((pageIdx, index) => [pageIdx, index + 1]));
        for (const entry of entries) {
            const seen = new Set();
            // 按首次出现的位置保留物理区域，父子共框及重复来源只参与一次连接。
            const sources = entry.sources.filter(source => {
                const key = JSON.stringify([source.page_idx, source.bbox]);
                if (seen.has(key)) return false;
                seen.add(key);
                return true;
            });
            if (sources.length > 1) mergedEntries.set(entry.id, { ...entry, sources });
            for (const source of entry.sources) {
                if (!sourcesByPage.has(source.page_idx)) sourcesByPage.set(source.page_idx, []);
                sourcesByPage.get(source.page_idx).push({ entry, bbox: source.bbox, source });
                traceableSources.add(sourceKey(source));
            }
        }
        for (const region of regions) {
            layoutBySource.set(sourceKey(region), region);
            const previous = layoutByBox.get(boxKey(region));
            if (!previous || region.child_path.length > previous.child_path.length) layoutByBox.set(boxKey(region), region);
            if (!layoutByPage.has(region.page_idx)) layoutByPage.set(region.page_idx, []);
            layoutByPage.get(region.page_idx).push(region);
        }
    };
    // 默认仅排列紧凑序号，当前交互来源的完整标签优先贴着所属框显示。
    const positionLabels = ({ layout, annotations }) => {
        const width = layout.clientWidth;
        const height = layout.clientHeight;
        if (!width || !height) return;
        const occupied = [];
        const ordered = [...annotations].sort((a, b) => Number(b.expanded) - Number(a.expanded));
        for (const item of ordered) {
            const { label, region, expanded } = item;
            label.textContent = expanded ? item.fullLabel : item.shortLabel;
            label.style.maxWidth = `${width}px`;
            const labelHeight = label.offsetHeight;
            const labelWidth = label.offsetWidth;
            let left = Math.max(0, Math.min(region.bbox[0] * width, width - labelWidth));
            const top = Math.max(0, region.bbox[1] * height - labelHeight - 1);
            // 只比较同一行附近的标签，沿框的上沿寻找空位。
            const nearby = occupied.filter((rect) => top < rect.bottom && top + labelHeight > rect.top);
            if (nearby.some((rect) => left < rect.right && left + labelWidth > rect.left)) {
                for (const rect of nearby.sort((a, b) => a.left - b.left)) {
                    if (left < rect.right && left + labelWidth > rect.left) left = rect.right + 2;
                }
                left = Math.max(0, Math.min(left, width - labelWidth));
            }
            label.style.left = `${left - region.bbox[0] * width}px`;
            label.style.top = `${top - region.bbox[1] * height}px`;
            occupied.push({ left, right: left + labelWidth + 2, top, bottom: top + labelHeight });
        }
    };
    // 清除已挂载区域，同时保留 PDF.js 自己管理的画布及批注。
    const clearLayers = () => {
        for (const { hit, layout } of layers.values()) {
            hit.remove();
            layout.remove();
        }
        layers.clear();
        clearConnections();
    };
    // 重绘时从文档状态恢复选中样式，避免依赖已销毁的 DOM 引用。
    const updateSelection = () => {
        const inspected = [hoveredBox, focusedBox].find(box => box?.isConnected);
        const inspectedKey = inspected?.dataset.sourceKey || "";
        const expandedKey = inspectedKey || selectedSourceKey;
        const selectedSources = entries.find(entry => entry.id === selectedId)?.sources || [];
        const selectedKeys = new Set(selectedSources.map(source => sourceKey(layoutRegion(source) || source)));
        for (const layer of layers.values()) {
            layer.layout.classList.toggle("show-all", showAll);
            for (const box of layer.hit.children) {
                const selected = box.traceIds.includes(selectedId);
                box.classList.toggle("selected", selected);
                if (!box.source) continue;
                // 同框父子使用当前选中来源的类型色，鼠标命中仍由具体子块负责。
                const source = selected && selectedSources.find(item => item.page_idx === box.source.page_idx
                    && item.bbox.every((value, index) => value === box.source.bbox[index]));
                box.style.setProperty("--trace-color", themeColor(layoutRegion(source || box.source)));
            }
            let labelsChanged = false;
            for (const item of layer.annotations) {
                const key = item.node.dataset.sourceKey;
                const expanded = key === expandedKey;
                labelsChanged ||= expanded !== item.expanded;
                item.expanded = expanded;
                item.node.classList.toggle("selected", selectedKeys.has(key));
                item.node.classList.toggle("hovered", key === inspectedKey);
                item.node.classList.toggle("expanded", expanded);
            }
            if (labelsChanged) positionLabels(layer);
        }
        scheduleConnections();
    };
    // 可跳转块和辅助块共用悬停、键盘检查入口，已销毁元素的状态自动失效。
    const inspectOnFocus = (node) => {
        // 键盘定位或重绘后，指针仍可能停在原框内。首次移动恢复检查，其余移动跳过排版。
        for (const name of ["pointerenter", "pointermove"]) {
            node.addEventListener(name, () => {
                if (hoveredBox === node) return;
                hoveredBox = node;
                updateSelection();
            });
        }
        node.addEventListener("pointerleave", () => {
            if (hoveredBox === node) hoveredBox = null;
            updateSelection();
        });
        node.addEventListener("focus", () => {
            focusedBox = node;
            updateSelection();
        });
        node.addEventListener("blur", () => {
            if (focusedBox === node) focusedBox = null;
            updateSelection();
        });
    };
    // 跳页后的页面可能尚未获得真实尺寸，等待对应页完成渲染再定位。
    const finishNavigation = () => {
        if (!pending) return;
        const pageNumber = pageIndices.indexOf(pending.pageIdx) + 1;
        const page = viewer.getPageView(pageNumber - 1);
        const layer = page?.div.querySelector(".mineru-trace-layer");
        const boxes = [...(layer?.children || [])].filter(item => item.traceIds.includes(pending.id));
        const target = layoutBySource.get(selectedSourceKey);
        const box = boxes.find(item => item.dataset?.sourceKey === selectedSourceKey
            || (target && item.source && boxKey(item.source) === boxKey(target))) || boxes[0];
        if (!box) return;
        const rect = box.getBoundingClientRect();
        const host = container.getBoundingClientRect();
        // 放大后同时沿横纵方向对齐，超宽区域保留左侧起始部分可见。
        container.scrollTo({
            left:
                container.scrollLeft +
                rect.left -
                host.left -
                Math.max(16, (container.clientWidth - Math.min(rect.width, container.clientWidth - 32)) / 2),
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
        for (const [number, layer] of layers) if (!layer.hit.isConnected) layers.delete(number);
        const old = layers.get(pageNumber);
        if (old) {
            old.hit.remove();
            old.layout.remove();
        }
        const pageIdx = pageIndices[pageNumber - 1];
        const candidates = [...(sourcesByPage.get(pageIdx) || [])];
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
        const layout = document.createElement("div");
        layout.className = "mineru-layout-layer";
        layout.setAttribute("aria-hidden", "true");
        const annotations = [];
        const labelsBySource = new Map();
        for (const region of layoutByPage.get(pageIdx) || []) {
            const node = document.createElement("div");
            node.className = "mineru-layout-region";
            node.dataset.blockType = region.type;
            node.dataset.blockIndex = String(region.index ?? "");
            node.dataset.sourceKey = sourceKey(region);
            node.style.setProperty("--trace-color", themeColor(region));
            placeRegion(node, region.bbox);
            // 阅读序号统一转换为一基显示，缺失块号的子块使用同样的一基层级路径。
            const shortLabel = region.index == null
                ? `${region.block_index + 1}/${region.child_path.map(index => index + 1).join(".")}`
                : String(region.index + 1);
            const fullLabel = `${region.type} · ${shortLabel}`;
            labelsBySource.set(sourceKey(region), fullLabel);
            const label = document.createElement("span");
            label.className = "mineru-layout-label";
            label.textContent = shortLabel;
            node.append(label);
            layout.append(node);
            annotations.push({
                node, label, region, shortLabel, fullLabel, expanded: false
            });
            // 正文过滤掉的区域仍可检查完整标签，使用描述元素表达其只读性质。
            if (!traceableSources.has(sourceKey(region))) {
                const inspector = document.createElement("div");
                inspector.className = "mineru-layout-inspector";
                inspector.dataset.sourceKey = sourceKey(region);
                inspector.traceIds = [];
                inspector.tabIndex = 0;
                inspector.setAttribute("role", "img");
                inspector.setAttribute("aria-label", `${messages.trace_source} ${pageIdx + 1} · ${fullLabel}`);
                placeRegion(inspector, region.bbox);
                inspectOnFocus(inspector);
                layer.append(inspector);
            }
        }
        for (const { entry, bbox, ids, source } of regions) {
            const box = document.createElement("button");
            box.type = "button";
            box.dataset.traceId = entry.id;
            box.source = source;
            const region = layoutRegion(source);
            box.dataset.sourceKey = sourceKey(region || source);
            box.style.setProperty("--trace-color", themeColor(region));
            box.traceIds = ids;
            const description = [messages.trace_source, pageIdx + 1, labelsBySource.get(sourceKey(source))]
                .filter(Boolean).join(" · ");
            box.setAttribute("aria-label", description);
            const merged = ids.map(id => mergedEntries.get(id)).find(Boolean);
            if (merged) box.setAttribute("aria-description", messages.trace_merged_description.replace("{count}", merged.sources.length));
            placeRegion(box, bbox);
            box.addEventListener("click", () => send({ type: "select", id: entry.id, pageIdx, sourceKey: sourceKey(source) }));
            inspectOnFocus(box);
            layer.append(box);
        }
        page.div.append(layer, layout);
        const mounted = { hit: layer, layout, annotations };
        layers.set(pageNumber, mounted);
        updateSelection();
        positionLabels(mounted);
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
                selectedSourceKey = "";
                hoveredBox = null;
                focusedBox = null;
                indexRegions([]);
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
                const layoutRegions = Array.isArray(data.layoutRegions) ? data.layoutRegions.filter((region) =>
                    Number.isInteger(region.page_idx) && Number.isInteger(region.block_index) &&
                    Array.isArray(region.child_path) && region.child_path.every((value) => Number.isInteger(value) && value >= 0) &&
                    typeof region.type === "string" && (region.index === null || Number.isInteger(region.index)) &&
                    Array.isArray(region.bbox) && region.bbox.length === 4 &&
                    region.bbox.every((value) => Number.isFinite(value) && value >= 0 && value <= 1) &&
                    Array.isArray(region.color) && region.color.length === 3 &&
                    region.color.every((value) => Number.isFinite(value) && value >= 0 && value <= 1)
                ) : [];
                indexRegions(layoutRegions);
                clearLayers();
                hoveredBox = null;
                focusedBox = null;
                toggle.hidden = !layoutRegions.length && !entries.length;
                for (const page of viewer.getCachedPageViews()) mount(page.id);
                scheduleConnections();
            } else if (data.type === "select" && entries.some((entry) => entry.id === data.id)) {
                pending = null;
                selectedId = data.id;
                const entry = entries.find(item => item.id === data.id);
                const source = entry.sources.find(item => sourceKey(item) === data.sourceKey)
                    || entry.sources.find(item => item.page_idx === data.pageIdx) || entry.sources[0];
                selectedSourceKey = source ? sourceKey(layoutRegion(source) || source) : "";
                hoveredBox = null;
                focusedBox = null;
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
    const connectionEvents = ["updateviewarea", "scalechanging", "pagesloaded"];
    for (const name of connectionEvents) eventBus.on(name, scheduleConnections);
    container.addEventListener("scroll", scheduleConnections, { ...options, passive: true });
    const connectionObserver = new ResizeObserver(scheduleConnections);
    connectionObserver.observe(container);
    send({ type: "ready" });
    return () => {
        // 查看器卸载时释放本次文档拥有的监听器和浮层。
        abort.abort();
        eventBus.off("pagerendered", onRendered);
        for (const name of connectionEvents) eventBus.off(name, scheduleConnections);
        connectionObserver.disconnect();
        clearLayers();
        connections.remove();
        toggle.remove();
    };
}
