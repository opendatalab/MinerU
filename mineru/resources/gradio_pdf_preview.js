// 仅通过原生 FileData 和事件值连接预览器，不访问 Gradio 内部 DOM 或组件状态。
(action, ...args) => {
    const state = window.__mineruPdfPreview ??= { source: "", conversion: "", revision: 0, result: "" };
    // 路径用于比较文档身份；仅 Windows 路径转换分隔符，保留 POSIX 文件名中的反斜杠。
    const path = (value) => /^[a-z]:[\\/]|^\\\\/i.test(value) ? value.replace(/\\/g, "/") : value;
    const key = (file) => path(String(file?.path || file?.url || ""));
    // 输出只更新自有 HTML 组件；迟到回调通过空更新保持当前预览。
    const skip = () => ({ __type__: "update" });
    const empty = () => ({ __type__: "update", value: "", visible: true });
    // 使用 FileData 的路由前缀与原始路径，避免空格、#、?、% 等文件名字符被误解释为 URL 语法。
    const fileUrl = (file, routeBase = null) => {
        if (!file?.url || !file?.path || /^https?:/i.test(file.path)) throw new Error("Invalid preview file");
        const marker = "/gradio_api/file=";
        const supplied = new URL(file.url, window.location.href);
        if (supplied.origin !== window.location.origin) throw new Error("Invalid preview origin");
        if (!supplied.pathname.includes(marker)) throw new Error("Invalid preview route");
        // 事件结果的相对 URL 可能省略挂载前缀，使用初始静态文件的完整路由作为统一基址。
        const route = routeBase || supplied.href;
        const index = route.indexOf(marker);
        const encoded = path(file.path).split("/").map(encodeURIComponent).join("/");
        const url = new URL(route.slice(0, index + marker.length) + encoded, window.location.href);
        if (url.origin !== window.location.origin) throw new Error("Invalid preview origin");
        return url.href;
    };
    // 文件名和本地化文本都必须作为属性文本编码，不允许注入活动 HTML。
    const escapeHtml = (value) => String(value).replace(/[&<>"']/g, (character) => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    })[character]);

    if (action === "reset" || action === "clear") {
        state.source = action === "clear" ? "" : key(args[0]);
        state.conversion = "";
        state.result = "";
        state.revision += 1;
        return empty();
    }
    if (action === "begin") {
        state.conversion = state.source;
        return [];
    }
    const [file, source, viewer, runId] = args;
    if (!state.source || key(source) !== state.source) return skip();
    if (action === "source") {
        // 原文的 File 缓存路径与上传文件一致；不能让晚到的原文事件覆盖布局结果。
        if (state.result || (file && key(file) !== key(source))) return skip();
    } else if (action === "result") {
        if (!runId || state.conversion !== state.source) return skip();
        state.result = runId;
    } else {
        return skip();
    }
    if (!file) return empty();
    const { text } = window.__mineruI18n;
    try {
        const messages = Object.fromEntries([
            "preview", "pdf_loading", "pdf_previous", "pdf_next", "pdf_page", "pdf_zoom_out", "pdf_zoom_in",
            "pdf_fit_width", "pdf_invalid", "pdf_password", "pdf_load_failed",
        ].map((name) => [name, text(name)]));
        const viewerUrl = fileUrl(viewer);
        const params = new URLSearchParams({ file: fileUrl(file, viewerUrl), messages: JSON.stringify(messages) });
        const url = new URL(viewerUrl);
        // Gradio 6 会复用 iframe 节点；只改 hash 不会重新加载文档，必须改变查询参数。
        url.searchParams.set("document", String(++state.revision));
        url.hash = params.toString();
        return {
            __type__: "update", visible: true,
            value: `<iframe class="mineru-pdf-frame" title="${escapeHtml(text("preview"))}" ` +
                `src="${escapeHtml(url.href)}" sandbox="allow-scripts allow-same-origin"></iframe>`,
        };
    } catch (error) {
        return { __type__: "update", visible: true, value: `<div role="alert">${escapeHtml(text("pdf_load_failed"))}</div>` };
    }
}
