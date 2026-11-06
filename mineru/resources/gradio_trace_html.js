// 此脚本运行于隔离的结果 iframe，全部内容定位在本窗口完成。
(config) => {
    let pageLabel = config.pageLabels[navigator.language.startsWith("zh") ? 1 : 0];
    const nodes = new Map();
    const entries = [];
    let selected = null;
    let selectedPage = null;
    let alignOnResize = false;
    let resizeFrame = 0;
    // 来源只从 renderer 创建的属性读取，不从正文字符串推断。
    for (const node of document.querySelectorAll("[data-docvortex-trace-id]")) {
        const id = node.dataset.docvortexTraceId;
        const sources = JSON.parse(node.dataset.docvortexSources);
        if (!sources.length) continue;
        nodes.set(id, node);
        entries.push({ id, sources, depth: sources[0].child_path.length });
        node.tabIndex = 0;
    }
    const navigation = document.createElement("nav");
    navigation.className = "mineru-trace-pages";
    navigation.setAttribute("aria-label", pageLabel);
    navigation.hidden = true;
    document.body.append(navigation);
    // 所有事件带产物身份，父窗口仍需核对实际发送窗口。
    const send = (message) =>
        parent.postMessage({ ...message, channel: "mineru-trace", documentId: config.documentId }, config.parentOrigin);
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
        window.scrollTo({
            top: scrollY + rect.top - Math.max(offset, (innerHeight - Math.min(rect.height, innerHeight - 48)) / 2)
        });
        positionNavigation();
    };
    // 一个目标对应多个原文页时，用去重后的原文页号切换来源。
    const select = (id, pageIdx, scroll) => {
        if (scroll) window.dispatchEvent(new CustomEvent("mineru-preview-navigate", { detail: "trace" }));
        selected?.removeAttribute("data-mineru-selected");
        selected = nodes.get(id);
        if (!selected) return;
        selectedPage = pageIdx;
        selected.setAttribute("data-mineru-selected", "true");
        navigation.replaceChildren();
        const entry = entries.find((item) => item.id === id);
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
            event.origin !== config.parentOrigin ||
            data?.channel !== "mineru-trace" ||
            data.documentId !== config.documentId
        )
            return;
        if (data.type === "labels") {
            pageLabel = data.pageLabel;
            navigation.setAttribute("aria-label", pageLabel);
        }
        if (data.type === "select") select(data.id, data.pageIdx, data.scroll);
    });
    // 用户主动滚动后停止异步布局校正；新的定位请求重新启用。
    for (const name of ["wheel", "touchstart", "pointerdown", "keydown"]) {
        window.addEventListener(
            name,
            () => {
                alignOnResize = false;
            },
            { passive: true }
        );
    }
    window.addEventListener("scroll", positionNavigation, { passive: true });
    // 目录接管正文导航时，结束旧来源块的自动对齐。
    window.addEventListener("mineru-preview-navigate", () => { alignOnResize = false; });
    const observer = new ResizeObserver(() => {
        cancelAnimationFrame(resizeFrame);
        resizeFrame = requestAnimationFrame(align);
    });
    observer.observe(document.body);
    window.addEventListener(
        "pagehide",
        () => {
            observer.disconnect();
            cancelAnimationFrame(resizeFrame);
        },
        { once: true }
    );
    send({ type: "ready", entries, pageIndices: config.pageIndices });
}
