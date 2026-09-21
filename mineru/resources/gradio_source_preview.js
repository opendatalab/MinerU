(action, ...args) => {
    // 浏览器会话保存最新请求标识，清除和重复上传同一文件也会使旧请求失效。
    const key = "__mineruSourcePreview";
    const fileKey = `${key}File`;
    const skip = () => ({__type__: "update"});
    const text = (name, fallback) => window.__mineruI18n?.text?.(name) || fallback;
    // 复用 Gradio 文件路由，避免直接拼接原始路径造成空格、#、? 等字符歧义。
    const fileUrl = (file, routeBase = null) => {
        if (!file?.url || !file?.path || /^https?:/i.test(file.path)) throw new Error("Invalid preview file");
        const marker = "/gradio_api/file=";
        const supplied = new URL(file.url, window.location.href);
        if (supplied.origin !== window.location.origin || !supplied.pathname.includes(marker)) {
            throw new Error("Invalid preview origin");
        }
        const route = routeBase || supplied.href;
        const index = route.indexOf(marker);
        const encoded = (/^[a-z]:[\\/]|^\\\\/i.test(file.path) ? String(file.path).replace(/\\/g, "/") : String(file.path)).split("/").map(encodeURIComponent).join("/");
        const url = new URL(route.slice(0, index + marker.length) + encoded, window.location.href);
        if (url.origin !== window.location.origin) throw new Error("Invalid preview origin");
        return url.href;
    };
    const escapeHtml = (value) => String(value).replace(/[&<>"']/g, (character) => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    })[character]);

    if (action === "begin" || action === "clear") {
        const file = action === "clear" ? null : args[0];
        window[key] = (window[key] || 0) + 1;
        window[fileKey] = file ? {path: file.path || "", url: file.url || ""} : null;
        const ticket = {id: window[key], path: file?.path || ""};
        return [JSON.stringify(ticket), ""];
    }
    const receipt = JSON.parse(args[0]);
    if (receipt.id !== window[key]) return skip();
    if (receipt.kind !== "epub") return receipt.html;

    // EPUB viewer 本身是静态资源，书籍和 viewer 都必须来自当前 Gradio origin。
    const viewer = args[1];
    try {
        const viewerUrl = fileUrl(viewer);
        const book = fileUrl(window[fileKey], viewerUrl);
        const params = new URLSearchParams({
            book,
            messages: JSON.stringify(Object.fromEntries(["epub_preview_failed", "epub_loading", "epub_previous", "epub_next", "epub_page", "epub_spine", "epub_contents"]
                .map((name) => [name, text(name, "")]))),
        });
        const url = new URL(viewerUrl);
        url.search = params.toString();
        url.searchParams.set("document", String(receipt.id));
        return `<iframe class="mineru-epub-frame" title="${escapeHtml(text("preview", "Document preview"))}" ` +
            `src="${escapeHtml(url.href)}" sandbox="allow-scripts allow-same-origin"></iframe>`;
    } catch (_error) {
        return `<div role="alert">${escapeHtml(text("epub_preview_failed", "Could not load EPUB preview."))}</div>`;
    }
}
