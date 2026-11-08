// 上传沿用 Gradio 的 HTTP 端点和 props 桥接；XHR 仅补充真实字节进度与中止能力。
({ element, props, server, accepted, imageExtensions }) => {
    let id = "";
    let localFile = null;
    let uploadedFile = null;
    let shownFile = null;
    let pending = false;
    let bridged = "";
    let transfer = null;
    let progress = 0;
    let transferred = 0;
    let message = "";
    let failed = false;
    let settled = false;
    let settleTimer = null;
    // DOM 按需取得，兼容 Gradio 对属性更新的模板协调。
    const node = name => element.querySelector(`.mineru-upload-${name}`);
    const input = () => element.querySelector('input[type="file"]');
    const text = key => window.__mineruI18n?.text(key) || key;
    const identity = () => Array.from(crypto.getRandomValues(new Uint8Array(16)), value => value.toString(16).padStart(2, "0")).join("");
    const isImage = name => imageExtensions.some(extension => name.toLowerCase().endsWith(`.${extension}`));
    // 文件图标只按用途分组，不改变可上传格式；图片仍复用后端提供的扩展名集合。
    const iconByExtension = {
        pdf: "pdf", ofd: "ofd",
        doc: "document", docx: "document", odt: "document", rtf: "document",
        xls: "sheet", xlsx: "sheet", ods: "sheet", csv: "sheet", tsv: "sheet",
        ppt: "slides", pptx: "slides", odp: "slides",
        html: "web", htm: "web", shtml: "web", mhtml: "web", mht: "web",
        epub: "ebook",
    };
    // 大小写后缀使用相同图标，无后缀或未知格式回退到普通文件图标。
    const fileIconKind = name => {
        if (isImage(name)) return "image";
        const dot = name.lastIndexOf(".");
        const extension = dot > 0 ? name.slice(dot + 1).toLowerCase() : "";
        return Object.hasOwn(iconByExtension, extension) ? iconByExtension[extension] : "file";
    };
    // 本地 File 和 Gradio FileData 共用文件名规则，不把文件名当作 HTML。
    const fileName = file => file?.orig_name || file?.name || file?.path?.split(/[\\/]/).pop() || "";
    // 使用十进制单位；未知大小不能伪装成空文件。
    const formatSize = bytes => {
        if (!Number.isFinite(bytes) || bytes < 0) return text("upload_size_unknown");
        const unit = bytes > 0 ? Math.min(4, Math.floor(Math.log10(bytes) / 3)) : 0;
        return `${unit ? (bytes / 1000 ** unit).toFixed(1) : bytes} ${["B", "KB", "MB", "GB", "TB"][unit]}`;
    };
    // 只切换空状态和文件条，进度、文件名及错误均写入固定叶节点。
    const render = () => {
        const name = fileName(shownFile);
        const complete = Boolean(uploadedFile && !pending && !failed);
        const size = shownFile?.size;
        const extension = name.lastIndexOf(".");
        node("target").hidden = Boolean(name);
        node("file-row").hidden = !name;
        node("file-row").classList.toggle("is-settled", settled);
        node("stem").textContent = extension > 0 ? name.slice(0, extension) : name;
        node("extension").textContent = extension > 0 ? name.slice(extension) : "";
        node("label").title = name;
        node("size").textContent = pending && progress !== null
            ? `${formatSize(transferred)} / ${formatSize(size)}` : formatSize(size);
        // 完成状态放在文件大小旁，右侧仅保留上传百分比或清除操作。
        node("finished").hidden = !complete;
        node("finished").textContent = `· ${text("upload_complete")}`;
        const kind = fileIconKind(name);
        const hasBadge = kind === "pdf" || kind === "ofd";
        node("file-icon").setAttribute("data-file-kind", kind);
        element.querySelectorAll("[data-mineru-upload-icon]").forEach(icon => {
            icon.toggleAttribute("hidden", icon.dataset.mineruUploadIcon !== (hasBadge ? "file" : kind));
        });
        node("format-badge").hidden = !hasBadge;
        node("format-badge").textContent = hasBadge ? kind.toUpperCase() : "";
        node("indicator").hidden = !pending || progress === null;
        node("percent").hidden = !pending || progress === null;
        node("percent").textContent = `${Math.floor(progress || 0)}%`;
        node("fill").style.width = `${progress || 0}%`;
        node("fill").hidden = failed || (!pending && !uploadedFile);
        node("meter").hidden = !pending;
        node("meter").setAttribute("aria-label", text("upload_progress"));
        if (progress === null) node("meter").removeAttribute("aria-valuenow");
        else node("meter").setAttribute("aria-valuenow", String(Math.floor(progress)));
        node("meter").setAttribute("aria-valuetext", progress === null ? text("uploading")
            : `${Math.floor(progress)}% · ${node("size").textContent}`);
        const clearLabel = text(pending ? "upload_cancel" : "upload_clear");
        node("remove").title = clearLabel;
        node("remove").setAttribute("aria-label", `${clearLabel}: ${name}`);
        node("status").textContent = message || (pending && progress === 100 ? text("upload_confirming")
            : pending && progress === null ? text("uploading") : "");
        node("status").setAttribute("role", failed ? "alert" : "status");
        node("retry").hidden = !failed;
        node("retry").textContent = text("upload_retry");
    };
    // 统一保存当前可见文件，语言初始化和预览准备回执不能清空已选文件名。
    const display = (file, notice = "", error = false) => {
        shownFile = file;
        message = notice;
        failed = error;
        render();
    };
    // 取消正在传输的请求与完成动画；身份失效先于 abort，迟到回调不会重绘新文件。
    const stopTransfer = () => {
        clearTimeout(settleTimer);
        settled = false;
        const previous = transfer;
        transfer = null;
        previous?.abort();
    };
    // 由公开组件服务函数提供部署地址；复用原生上传路由的登录、大小限制和缓存管理。
    const uploadFile = async (file, current) => {
        const context = await server.upload_context();
        if (id !== current) throw new DOMException("Upload cancelled", "AbortError");
        return new Promise((resolve, reject) => {
            const request = new XMLHttpRequest();
            transfer = request;
            request.open("POST", context.upload_url);
            request.withCredentials = true;
            // multipart 的真实传输比例映射到文件大小，完全不使用模拟进度计时器。
            request.upload.addEventListener("progress", event => {
                if (id !== current) return;
                progress = event.lengthComputable && event.total > 0 ? Math.min(100, event.loaded / event.total * 100) : null;
                transferred = progress === null ? 0 : Math.min(file.size, Math.floor(file.size * progress / 100));
                render();
            });
            // 只有服务端返回有效缓存路径才算上传成功，传输到 100% 时仍等待确认。
            request.addEventListener("load", () => {
                if (transfer === request) transfer = null;
                try {
                    if (request.status < 200 || request.status >= 300) {
                        throw new Error(request.responseText || `${text("upload_failed")} (${request.status})`);
                    }
                    const paths = JSON.parse(request.responseText);
                    if (!Array.isArray(paths) || paths.length !== 1 || typeof paths[0] !== "string" || !paths[0]) {
                        throw new Error(text("upload_failed"));
                    }
                    const path = paths[0];
                    resolve({ path, url: context.file_url_prefix + encodeURIComponent(path).replaceAll("%2F", "/") });
                } catch (error) { reject(error); }
            });
            // 网络失败显示可重试错误，主动取消由选择身份校验静默收尾。
            request.addEventListener("error", () => {
                if (transfer === request) transfer = null;
                reject(new Error(text("upload_failed")));
            });
            request.addEventListener("abort", () => reject(new DOMException("Upload cancelled", "AbortError")));
            const body = new FormData();
            body.append("files", file, file.name);
            request.send(body);
        });
    };
    // 每次选择有独立身份，文件替换会中止旧传输；准备失败重试复用已上传的路径。
    const submit = async file => {
        if (!file) return;
        if (!accepted.some(extension => file.name.toLowerCase().endsWith(extension))) {
            display(shownFile, text("upload_unsupported"));
            return;
        }
        const previous = localFile === file ? uploadedFile : null;
        id = identity();
        const current = id;
        stopTransfer();
        localFile = file;
        uploadedFile = null;
        pending = true;
        progress = previous ? 100 : 0;
        transferred = previous ? file.size : 0;
        window.__mineruPdfPreview.local(file, current);
        display(file);
        props.value = JSON.stringify({ id: current, stage: "selected" });
        try {
            const result = previous || await uploadFile(file, current);
            if (id !== current) return;
            const data = { ...result, orig_name: file.name, size: file.size, mime_type: file.type, meta: { _type: "gradio.FileData" } };
            if (!window.__mineruPdfPreview.bind(data, current)) return;
            uploadedFile = data;
            pending = false;
            progress = 100;
            transferred = file.size;
            display(file, isImage(file.name) ? text("preparing_pdf") : "");
            // 完成填满后淡出底色；该计时器不参与实际上传进度计算。
            settleTimer = setTimeout(() => {
                if (id !== current) return;
                settled = true;
                render();
            }, 650);
            // 让清空事件先提交，重复准备同一服务器路径仍能触发文件变更。
            await new Promise(resolve => setTimeout(resolve, 0));
            if (id === current) props.value = JSON.stringify({ id: current, stage: "uploaded", file: data });
        } catch (error) {
            if (id !== current) return;
            pending = false;
            display(file, String(error?.message || error), true);
        }
    };
    // 文件条叉号和原生清除共用失效入口，资源租期与解析任务仍由已有清除事件回收。
    const invalidate = () => {
        id = "";
        stopTransfer();
        localFile = null;
        uploadedFile = null;
        pending = false;
        progress = 0;
        transferred = 0;
        input().value = "";
        display(null);
    };
    window.__mineruUpload = {
        isImage, invalidate,
        // 同一回执重复派发不写回 FileData；迟到回执还须同时匹配本控件和预览身份。
        bridge(value) {
            let receipt;
            try { receipt = JSON.parse(value); } catch { return { __type__: "update" }; }
            const key = `${receipt.id}:${receipt.stage}`;
            if (receipt.id !== id || receipt.id !== window.__mineruPdfPreview.sourceId || key === bridged) return { __type__: "update" };
            bridged = key;
            return receipt.stage === "uploaded" ? receipt.file : null;
        },
        // 启动脚本晚于组件挂载时补齐当前状态的语言，不重置文件选择。
        localize() { render(); },
        // 示例写入隐藏 File 时中止旧上传；本地上传开始产生的空值桥接仍保留传输。
        sync(file) {
            if ((pending && !file) || (file?.path && uploadedFile?.path === file.path)) return;
            invalidate();
            uploadedFile = file;
            progress = file ? 100 : 0;
            settled = true;
            display(file);
        },
        // 图片转换为 PDF 等后续准备失败时保留文件条，仍可重试准备。
        prepared(error) { display(shownFile, error || "", Boolean(error && localFile)); },
    };
    // 选择、拖放和粘贴统一进入同一个上传生命周期。
    element.addEventListener("change", event => {
        if (event.target === input()) submit(input().files?.[0]);
    });
    element.addEventListener("dragover", event => { event.preventDefault(); });
    element.addEventListener("drop", event => {
        event.preventDefault();
        submit(event.dataTransfer?.files?.[0]);
    });
    node("retry").addEventListener("click", () => submit(localFile));
    // 叉号位于文件选择 label 之外，复用清除按钮的全部事件而不会打开文件选择器。
    node("remove").addEventListener("click", event => {
        event.preventDefault();
        window.__mineruPdfPreview.clear();
        document.getElementById("mineru-kit-clear").click();
        input().focus({ preventScroll: true });
    });
    // 剪贴板文件直接进入传输，截图按 MIME 补全扩展名，不注入合成 FileList。
    const paste = event => {
        const mimeExtensions = {
            "image/png": "png", "image/jpeg": "jpg", "image/webp": "webp", "image/gif": "gif",
            "image/bmp": "bmp", "image/tiff": "tiff", "application/pdf": "pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
            "application/vnd.openxmlformats-officedocument.presentationml.presentation": "pptx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
        };
        const data = event.clipboardData;
        const files = data?.files?.length ? Array.from(data.files) : Array.from(data?.items || []).map(item => item.getAsFile()).filter(Boolean);
        const normalized = files.map(file => {
            const extension = mimeExtensions[file.type];
            return !/\.[^.]+$/.test(file.name) && extension
                ? new File([file], `clipboard-${Date.now()}.${extension}`, { type: file.type }) : file;
        });
        const file = normalized.find(item => accepted.some(extension => item.name.toLowerCase().endsWith(extension)));
        if (file) { event.preventDefault(); submit(file); }
    };
    document.addEventListener("paste", paste);
    // 关闭页面同步停止传输，防止已卸载组件继续收到进度或完成回调。
    window.addEventListener("pagehide", () => {
        invalidate();
        document.removeEventListener("paste", paste);
    }, { once: true });
    display(null);
}
