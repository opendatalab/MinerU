// 此脚本运行于隔离的结果 iframe，全部内容定位在本窗口完成。
(config) => {
    // 与 MessageEvent.origin 使用同一 URL 规范化规则，统一默认端口和域名大小写。
    const parentOrigin = new URL(config.parentOrigin).origin;
    let pageLabel = config.pageLabels[navigator.language.startsWith("zh") ? 1 : 0];
    const nodes = new Map();
    const entries = [];
    const readingBlocks = [];
    const firstByPage = new Map();
    let positions = null;
    let pageSync = false;
    let htmlDrives = false;
    let lastPages = "";
    let pageFrame = 0;
    let selected = null;
    let selectedPage = null;
    let alignOnResize = false;
    let resizeFrame = 0;
    const motion = createMineruPreviewScroll(document.scrollingElement);
    // PDF 结果目录和溯源共用正文滚动器，后来的导航可以直接接管前一次动画。
    if (config.pdfPageSync) window.__mineruPreviewScroll = motion;
    // 使用完整物理身份索引原始布局，避免合并正文丢失各来源的类型色。
    const sourceKey = (source) => JSON.stringify([source.page_idx, source.block_index, source.child_path]);
    // 视觉父块通过同一顶层块的主体区域取得主题色。
    const boxKey = (source) => JSON.stringify([source.page_idx, source.block_index, source.bbox]);
    const regionsBySource = new Map();
    const regionsByBox = new Map();
    for (const region of config.layoutRegions || []) {
        regionsBySource.set(sourceKey(region), region);
        const previous = regionsByBox.get(boxKey(region));
        if (!previous || region.child_path.length > previous.child_path.length) regionsByBox.set(boxKey(region), region);
    }
    // 合并目标切换来源时同步正文与来源页控件的颜色，颜色只写入自有 CSS 变量。
    const setTheme = (node, source) => {
        const region = regionsBySource.get(sourceKey(source)) || regionsByBox.get(boxKey(source));
        const color = region?.color || [0.35, 0.4, 0.45];
        node.style.setProperty("--trace-color", `rgb(${color.map(value => Math.round(value * 255)).join(" ")})`);
    };
    // 来源只从 renderer 创建的属性读取，不从正文字符串推断。
    for (const node of document.querySelectorAll("[data-docvortex-trace-id]")) {
        const id = node.dataset.docvortexTraceId;
        const sources = JSON.parse(node.dataset.docvortexSources);
        if (!sources.length) continue;
        nodes.set(id, node);
        entries.push({ id, sources, depth: sources[0].child_path.length });
        if (config.pdfPageSync) {
            const pages = [...new Set(sources.map(source => source.page_idx))];
            readingBlocks.push({ id, node, pages, key: pages.join(",") });
        }
        setTheme(node, sources[0]);
        node.tabIndex = 0;
    }
    const navigation = document.createElement("nav");
    navigation.className = "mineru-trace-pages";
    navigation.setAttribute("aria-label", pageLabel);
    navigation.hidden = true;
    document.body.append(navigation);
    // 所有事件带产物身份，父窗口仍需核对实际发送窗口。
    const send = (message) =>
        parent.postMessage({ ...message, channel: "mineru-trace", documentId: config.documentId }, parentOrigin);
    // 缓存可见正文块的文档坐标和各原始页的首个块；隐藏的折叠内容不参与定位。
    const indexPositions = () => {
        if (positions) return;
        firstByPage.clear();
        positions = [];
        for (const block of readingBlocks) {
            const rect = block.node.getBoundingClientRect();
            if (!rect.height || !rect.width) continue;
            positions.push({ top: rect.top + scrollY, block });
        }
        positions.sort((a, b) => a.top - b.top);
        for (const { block } of positions) {
            for (const page of block.pages) if (!firstByPage.has(page)) firstByPage.set(page, block);
        }
    };
    // 到达目标后保留工具栏下方的空隙，翻页判断则使用可读区域的中线。
    const readingOffset = () => parseFloat(getComputedStyle(document.body).getPropertyValue("--mineru-preview-offset")) || 24;
    // 在已排序的块顶坐标上二分，滚动过程中无需逐块读取布局。
    const readingBlock = () => {
        indexPositions();
        if (scrollY > 0 && scrollY + innerHeight >= document.documentElement.scrollHeight - 1) return positions.at(-1)?.block;
        const line = scrollY + (innerHeight + readingOffset()) / 2;
        if (scrollY <= 0) return positions[0]?.block;
        let start = 0, end = positions.length;
        while (start < end) {
            const middle = (start + end) >>> 1;
            if (positions[middle].top <= line) start = middle + 1;
            else end = middle;
        }
        return positions[Math.max(0, start - 1)]?.block;
    };
    // 只在来源页集合变化时通知 PDF，同一页的多栏及同一张跨页表格内均可自由阅读。
    const reportReadingPage = () => {
        pageFrame = 0;
        if (!pageSync || !htmlDrives) return;
        const block = readingBlock();
        if (!block || block.key === lastPages) return;
        lastPages = block.key;
        send({ type: "html-page", id: block.id });
    };
    // 滚动和手动打断共用下一帧的阅读位置，避免同一帧重复同步。
    const scheduleReadingPage = () => {
        if (pageSync && htmlDrives && !pageFrame) pageFrame = requestAnimationFrame(reportReadingPage);
    };
    // 释放目录或精确溯源的旧锚点，仅在阅读位置确实属于别页时滚动正文。
    const followPdfPage = (pageIdx, scroll) => {
        htmlDrives = false;
        alignOnResize = false;
        lastPages = "";
        motion.cancel();
        if (!scroll) return;
        const current = readingBlock();
        const target = firstByPage.get(pageIdx);
        if (!target || current?.pages.includes(pageIdx)) return;
        window.dispatchEvent(new CustomEvent("mineru-preview-navigate", { detail: "page" }));
        motion.move(scrollY + target.node.getBoundingClientRect().top - readingOffset());
    };
    // 初次加载和父窗口主动握手共用结果快照，覆盖两侧就绪先后顺序。
    const announceReady = () => send({ type: "ready", entries, initialPageIdx: config.initialPageIdx,
        layoutRegions: config.layoutRegions, pdfPageSync: config.pdfPageSync });
    // 页码条贴近选中内容并限制在当前 iframe 的可见范围内。
    const positionNavigation = () => {
        if (!selected) return;
        const rect = selected.getBoundingClientRect();
        const offset = parseFloat(getComputedStyle(document.body).getPropertyValue("--mineru-preview-offset")) || 8;
        navigation.style.top = `${Math.max(offset, Math.min(innerHeight - navigation.offsetHeight - 8, rect.top))}px`;
    };
    // 仅调整结果 iframe 自身滚动，避免带动整个 Gradio 页面。
    const align = () => {
        if (!selected || !alignOnResize) return;
        const rect = selected.getBoundingClientRect();
        const offset = parseFloat(getComputedStyle(document.body).getPropertyValue("--mineru-preview-offset")) || 24;
        motion.move(scrollY + rect.top - Math.max(offset, (innerHeight - Math.min(rect.height, innerHeight - 48)) / 2));
        positionNavigation();
    };
    // 一个目标对应多个原文页时，用去重后的原文页号切换来源。
    const select = (id, pageIdx, scroll, selectedSourceKey) => {
        htmlDrives = false;
        lastPages = "";
        if (scroll) window.dispatchEvent(new CustomEvent("mineru-preview-navigate", { detail: "trace" }));
        selected?.removeAttribute("data-mineru-selected");
        selected = nodes.get(id);
        if (!selected) return;
        selectedPage = pageIdx;
        selected.setAttribute("data-mineru-selected", "true");
        navigation.replaceChildren();
        const entry = entries.find((item) => item.id === id);
        const source = entry.sources.find(item => sourceKey(item) === selectedSourceKey)
            || entry.sources.find(item => item.page_idx === pageIdx) || entry.sources[0];
        setTheme(selected, source);
        setTheme(navigation, source);
        const pages = [...new Set(entry.sources.map((source) => source.page_idx))].sort((a, b) => a - b);
        for (const page of pages) {
            const button = document.createElement("button");
            button.textContent = `${pageLabel} ${page + 1}`;
            button.setAttribute("aria-pressed", String(page === selectedPage));
            button.addEventListener("click", () => send({ type: "select", id, pageIdx: page }));
            navigation.append(button);
        }
        navigation.hidden = pages.length < 2;
        alignOnResize = scroll;
        // 展开源内容所属的 details 后再读取其真实位置。
        for (let ancestor = selected.parentElement; ancestor; ancestor = ancestor.parentElement) {
            if (ancestor.tagName === "DETAILS") ancestor.open = true;
        }
        align();
        positionNavigation();
    };
    // 文字选择、链接及原生控件优先处理，普通块点击和键盘激活发送定位请求。
    const activate = (event) => {
        if (event.type === "keydown" && !["Enter", " "].includes(event.key)) return;
        if (event.target.closest("a,button,input,textarea,select,summary,[contenteditable=true]")) return;
        if (event.type === "click" && !window.getSelection()?.isCollapsed) return;
        const node = event.target.closest("[data-docvortex-trace-id]");
        if (!node || !nodes.has(node.dataset.docvortexTraceId)) return;
        if (event.type === "keydown") event.preventDefault();
        send({ type: "select", id: node.dataset.docvortexTraceId });
    };
    document.addEventListener("click", activate);
    document.addEventListener("keydown", activate);
    window.addEventListener("message", (event) => {
        const data = event.data;
        if (
            event.source !== parent ||
            event.origin !== parentOrigin ||
            data?.channel !== "mineru-trace" ||
            data.documentId !== config.documentId
        )
            return;
        if (data.type === "request-ready") {
            announceReady();
            return;
        }
        if (data.type === "labels") {
            pageLabel = data.pageLabel;
            navigation.setAttribute("aria-label", pageLabel);
        }
        if (data.type === "labels" || data.type === "page-sync-mode") {
            motion.cancel();
            pageSync = config.pdfPageSync && data.pageSync === true;
            htmlDrives = false;
            lastPages = "";
        }
        if (data.type === "pdf-page" && pageSync) followPdfPage(data.pageIdx, data.scroll);
        if (data.type === "select") select(data.id, data.pageIdx, data.scroll, data.sourceKey);
    });
    // 用户主动滚动后停止异步布局校正；新的定位请求重新启用。
    for (const name of ["wheel", "touchstart", "pointerdown", "keydown"]) {
        window.addEventListener(
            name,
            (event) => {
                const interrupted = motion.cancel();
                alignOnResize = false;
                htmlDrives = event.type !== "wheel" || !(event.ctrlKey || event.metaKey);
                if (interrupted) scheduleReadingPage();
            },
            { passive: true }
        );
    }
    window.addEventListener("scroll", () => {
        positionNavigation();
        scheduleReadingPage();
    }, { passive: true });
    // 预览区缩小时重新约束控件位置，即使用户操作已停用正文自动对齐。
    window.addEventListener("resize", () => {
        positions = null;
        positionNavigation();
    });
    // 目录接管正文导航时，结束旧来源块的自动对齐。
    window.addEventListener("mineru-preview-navigate", (event) => {
        motion.cancel();
        alignOnResize = false;
        htmlDrives = false;
        lastPages = "";
        const id = event.detail?.heading?.closest("[data-docvortex-trace-id]")?.dataset.docvortexTraceId;
        if (pageSync && id && nodes.has(id)) send({ type: "outline", id });
    });
    // 图片、公式、字体和折叠内容可能在首屏渲染后改变块位置，下一次使用时再测量。
    const invalidatePositions = () => { positions = null; };
    document.addEventListener("load", invalidatePositions, true);
    document.addEventListener("toggle", invalidatePositions, true);
    document.fonts?.addEventListener("loadingdone", invalidatePositions);
    const observer = new ResizeObserver(() => {
        invalidatePositions();
        cancelAnimationFrame(resizeFrame);
        resizeFrame = requestAnimationFrame(align);
    });
    observer.observe(document.body);
    window.addEventListener(
        "pagehide",
        () => {
            observer.disconnect();
            motion.cancel();
            cancelAnimationFrame(pageFrame);
            cancelAnimationFrame(resizeFrame);
        },
        { once: true }
    );
    announceReady();
}
