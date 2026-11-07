// 稳定容器由本控制器独占；Gradio 只传文件回执，不再替换 iframe HTML。
(action, ...args) => {
    if (!window.__mineruPdfPreview) {
        const state = { source: "", sourceId: "", revision: 0, conversion: "", result: "", blob: "", frame: null, viewer: null, uploading: false, localSelected: false, readyPath: "", file: null };
        // 使用全上下文可用的随机数，兼容局域网 HTTP。
        const identity = () => Array.from(crypto.getRandomValues(new Uint8Array(16)), value => value.toString(16).padStart(2, "0")).join("");
        // 文件路径只作为已完成上传的身份，不依据文件名猜测对应关系。
        const key = file => String(file?.path || file?.url || "");
        const skip = () => ({ __type__: "update" });
        const host = () => document.querySelector(".mineru-pdf-host");
        // 继续使用 Gradio 的同源文件路由，正确编码路径中的特殊字符和代理前缀。
        const fileUrl = (file, routeBase = null) => {
            if (!file?.url || !file?.path || /^https?:/i.test(file.path)) throw new Error("Invalid preview file");
            const supplied = new URL(file.url, location.href);
            const marker = "/gradio_api/file=";
            if (supplied.origin !== location.origin || !supplied.pathname.includes(marker)) throw new Error("Invalid preview route");
            const route = routeBase || supplied.href;
            const path = /^[a-z]:[\\/]|^\\\\/i.test(file.path) ? file.path.replace(/\\/g, "/") : file.path;
            return route.slice(0, route.indexOf(marker) + marker.length) + path.split("/").map(encodeURIComponent).join("/");
        };
        // 切换输入才销毁旧 PDF；先停止使用再撤销 Blob URL。
        const dispose = () => {
            try { state.frame?.contentWindow?.__mineruPdfDisposePreview?.(); } catch {}
            state.frame?.remove();
            state.frame = null;
            state.file = null;
            if (state.blob) URL.revokeObjectURL(state.blob);
            state.blob = "";
            host()?.replaceChildren();
        };
        // 状态只写文本节点，错误和文件名均不进入活动 HTML。
        const notice = (message, error = false) => {
            if (!host()) return;
            const node = document.createElement("div");
            node.setAttribute("role", error ? "alert" : "status");
            node.textContent = message;
            host().replaceChildren(node);
        };
        // 只为当前源文件挂载一次，上传回执和转换结果不会改变文档 URL。
        const mount = file => {
            if (file) state.file = file;
            file = state.file;
            if (state.frame || !state.viewer || !host() || !window.__mineruI18n || (!state.blob && !file)) return;
            const { text } = window.__mineruI18n;
            try {
                const viewerUrl = fileUrl(state.viewer);
                const source = state.blob || (file ? fileUrl(file, viewerUrl) : "");
                if (!source) return;
                const messages = Object.fromEntries([
                    "preview", "pdf_loading", "pdf_previous", "pdf_next", "pdf_page", "pdf_zoom_out", "pdf_zoom_in",
                    "trace_regions", "trace_source", "trace_merged", "trace_merged_description", "pdf_fit_width",
                    "pdf_invalid", "pdf_password", "pdf_load_failed",
                ].map(name => [name, text(name)]));
                const url = new URL(viewerUrl);
                url.searchParams.set("source", state.sourceId);
                url.hash = new URLSearchParams({ file: source, sourceId: state.sourceId, messages: JSON.stringify(messages) }).toString();
                const frame = document.createElement("iframe");
                frame.className = "mineru-pdf-frame";
                frame.title = text("preview");
                frame.setAttribute("sandbox", "allow-scripts allow-same-origin");
                frame.src = url.href;
                state.frame = frame;
                host().replaceChildren(frame);
            } catch (error) { notice(text("pdf_load_failed"), true); }
        };
        // 原生文件组件清空可能是上传开始时的桥接事件，不能撤销刚建立的本地预览。
        const reset = file => {
            const path = key(file);
            if ((!path && state.localSelected) || (path && path === state.source)) return;
            window.__mineruTrace?.reset();
            dispose();
            state.source = path;
            state.sourceId = identity();
            state.revision += 1;
            state.conversion = "";
            state.result = "";
            state.readyPath = "";
            state.uploading = false;
            state.localSelected = false;
            window.__mineruUpload?.sync(file);
        };
        state.local = (file, id) => {
            // 选文件时立即建立身份，PDF 本地读取与上传并行。
            reset(null);
            window.__mineruTrace?.reset();
            dispose();
            state.source = "";
            state.sourceId = id;
            state.revision += 1;
            state.conversion = "";
            state.result = "";
            state.readyPath = "";
            state.uploading = true;
            state.localSelected = true;
            if (/\.pdf$/i.test(file.name)) {
                state.blob = URL.createObjectURL(file);
                mount(null);
            } else if (window.__mineruUpload?.isImage(file.name)) {
                notice(window.__mineruI18n.text("preparing_pdf"));
            }
        };
        state.bind = (file, id) => {
            // 仅当前上传闭包可以绑定远端路径，迟到上传不能进入 Gradio 文件组件。
            if (id !== state.sourceId) return false;
            state.source = key(file);
            state.revision += 1;
            state.uploading = false;
            return true;
        };
        state.clear = () => {
            // 清除同时使本地上传失效，但不重复触发控件事件。
            state.uploading = false;
            state.localSelected = false;
            state.source = "";
            reset(null);
            window.__mineruUpload?.invalidate();
        };
        state.invoke = (operation, ...values) => {
            // 统一接收框架事件，所有返回值均保持稳定 HTML 容器。
            if (operation === "setup") { state.viewer = values[0] || state.viewer; mount(null); return []; }
            if (operation === "clear") { state.clear(); return skip(); }
            if (operation === "reset") { reset(values[0]); return skip(); }
            if (operation === "prepare") {
                if (!values[0] && state.localSelected) return skip();
                reset(values[0]);
                return JSON.stringify({ id: state.sourceId, revision: state.revision, path: key(values[0]) });
            }
            if (operation === "prepared") {
                let receipt;
                try { receipt = JSON.parse(values[0]); } catch { return skip(); }
                if (!receipt || receipt.id !== state.sourceId || receipt.path !== state.source) return skip();
                window.__mineruUpload?.prepared(receipt.error);
                if (receipt.error) { if (!state.frame) notice(receipt.error, true); return ""; }
                if (receipt.pdf) mount(receipt.pdf);
                state.readyPath = receipt.path;
                return receipt.path;
            }
            if (operation === "begin") {
                window.__mineruTrace?.reset();
                state.conversion = state.sourceId;
                state.result = "";
                return [];
            }
            const [file, source, viewer, runId] = values;
            if (viewer) state.viewer = viewer;
            if (!state.source || key(source) !== state.source) return skip();
            if (operation === "source") {
                if (!state.result && file && key(file) === key(source)) mount(file);
            } else if (operation === "result" && runId && state.conversion === state.sourceId) {
                state.result = runId;
                if (file) {
                    mount(file);
                    window.__mineruTrace?.activate(runId, state.sourceId);
                }
            }
            return skip();
        };
        window.addEventListener("pagehide", dispose, { once: true });
        window.__mineruPdfPreview = state;
    }
    return window.__mineruPdfPreview.invoke(action, ...args);
}
