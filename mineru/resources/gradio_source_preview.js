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

    // 源 HTML 可以执行自身脚本；可取消的整页导航直接阻止，其余导航探测后静态恢复。
    // HTML 的桌面视口缩放只作用于 iframe 外层舞台：WebKit 直接 transform iframe 时会出现
    // 内部页面布局正确、但绘制层只更新局部区域的现象，典型表现就是大块空白和正文被截断。
    const navigationGuardKey = `${key}NavigationGuard`;
    const sizeGuardKey = `${key}SizeGuard`;
    const navigationStates = new WeakMap();
    let probeSequence = 0;
    const fitSourceFrame = (frame, reportedWidth = null) => {
        if (!frame?.classList?.contains("mineru-source-frame")) return;
        const stage = frame.parentElement?.classList?.contains("mineru-source-stage") ? frame.parentElement : null;
        const viewport = stage?.parentElement;
        if (!stage || !viewport?.classList?.contains("mineru-source-viewport")) return;
        const availableWidth = viewport.clientWidth;
        const availableHeight = viewport.clientHeight;
        const measured = Number(reportedWidth ?? frame.dataset?.mineruSourceContentWidth ?? availableWidth);
        if (!Number.isFinite(measured) || availableWidth <= 0 || availableHeight <= 0) return;
        const contentWidth = Math.min(Math.max(measured, availableWidth), 2400);
        frame.dataset.mineruSourceContentWidth = String(contentWidth);
        const scale = Math.min(1, availableWidth / contentWidth);

        // iframe 始终只填满逻辑舞台，不在自身创建缩放合成层，避免 Safari/WebKit 的绘制裁剪。
        frame.style.width = "100%";
        frame.style.height = "100%";
        frame.style.transform = "";
        frame.style.transformOrigin = "";
        if (scale >= 0.995) {
            stage.style.width = "100%";
            stage.style.height = "100%";
            stage.style.transform = "";
            stage.style.transformOrigin = "";
            return;
        }
        stage.style.width = `${contentWidth}px`;
        stage.style.height = `${availableHeight / scale}px`;
        stage.style.transformOrigin = "0 0";
        stage.style.transform = `scale(${scale})`;
    };
    const observeSourceFrame = (frame) => {
        if (!frame || frame.dataset?.mineruSourceObserved === "1") return;
        const stage = frame.parentElement?.classList?.contains("mineru-source-stage") ? frame.parentElement : null;
        const viewport = stage?.parentElement;
        frame.dataset.mineruSourceObserved = "1";
        if (typeof ResizeObserver === "function" && viewport?.classList?.contains("mineru-source-viewport")) {
            new ResizeObserver(() => fitSourceFrame(frame)).observe(viewport);
        }
    };
    // 跳转后只恢复一次原文，并禁用源脚本，避免恢复本身再次触发跳转。
    const restoreSourceFrame = (frame, state) => {
        if (state.restored) return;
        const source = frame.getAttribute?.("srcdoc");
        if (!source) return;
        state.restored = true;
        state.probe = null;
        frame.setAttribute("sandbox", "allow-popups");
        frame.srcdoc = source;
    };
    const installNavigationGuard = () => {
        if (window[navigationGuardKey] || typeof document === "undefined") return;
        document.addEventListener("load", (event) => {
            const frame = event.target;
            if (!frame || frame.tagName !== "IFRAME" || !frame.classList?.contains("mineru-source-frame")) return;
            const frameId = frame.dataset?.mineruSourcePreviewId;
            if (!frameId || frameId !== String(window[key])) return;
            observeSourceFrame(frame);
            fitSourceFrame(frame);
            let state = navigationStates.get(frame);
            if (!state) {
                state = {restored: false, verified: false, probe: null};
                navigationStates.set(frame, state);
                // OFD 等无脚本预览不需要探测；HTML/MHTML 首次 load 可能已是重定向目标。
                if (!frame.getAttribute?.("sandbox")?.split(/\s+/).includes("allow-scripts")) {
                    state.verified = true;
                    return;
                }
                const probe = String(++probeSequence);
                state.probe = probe;
                frame.contentWindow?.postMessage({type: "mineru-source-preview-probe", probe}, "*");
                setTimeout(() => {
                    if (!state.verified && state.probe === probe && frame.dataset?.mineruSourcePreviewId === String(window[key])) {
                        restoreSourceFrame(frame, state);
                    }
                }, 250);
                return;
            }
            restoreSourceFrame(frame, state);
        }, true);
        window[navigationGuardKey] = true;
    };
    const installSizeGuard = () => {
        if (window[sizeGuardKey] || typeof window.addEventListener !== "function") return;
        window.addEventListener("message", (event) => {
            const type = event.data?.type;
            if (type !== "mineru-source-preview-size" && type !== "mineru-source-preview-probe-ack") return;
            for (const frame of document.querySelectorAll?.("iframe.mineru-source-frame") || []) {
                if (frame.contentWindow !== event.source) continue;
                const frameId = frame.dataset?.mineruSourcePreviewId;
                if (!frameId || frameId !== String(window[key])) return;
                if (type === "mineru-source-preview-probe-ack") {
                    const state = navigationStates.get(frame);
                    if (state && !state.restored && state.probe === event.data.probe) {
                        state.verified = true;
                        state.probe = null;
                    }
                    return;
                }
                observeSourceFrame(frame);
                fitSourceFrame(frame, event.data.width);
                return;
            }
        });
        window[sizeGuardKey] = true;
    };
    const markSourceFrame = (markup, id) => String(markup).replace(
        '<iframe class="mineru-source-frame"',
        `<iframe class="mineru-source-frame" data-mineru-source-preview-id="${escapeHtml(id)}"`,
    );
    installNavigationGuard();
    installSizeGuard();

    if (action === "begin" || action === "clear") {
        const file = action === "clear" ? null : args[0];
        window[key] = (window[key] || 0) + 1;
        window[fileKey] = file ? {path: file.path || "", url: file.url || ""} : null;
        const ticket = {id: window[key], path: file?.path || ""};
        return [JSON.stringify(ticket), ""];
    }
    const receipt = JSON.parse(args[0]);
    if (receipt.id !== window[key]) return skip();
    if (receipt.kind !== "epub") return markSourceFrame(receipt.html, receipt.id);

    // EPUB viewer 本身是静态资源，书籍和 viewer 都必须来自当前 Gradio origin。
    const viewer = args[1];
    try {
        const viewerUrl = fileUrl(viewer);
        const book = fileUrl(window[fileKey], viewerUrl);
        const params = new URLSearchParams({
            book,
            messages: JSON.stringify(Object.fromEntries(["epub_preview_failed", "epub_loading", "epub_previous", "epub_next", "epub_spine", "epub_contents"]
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
