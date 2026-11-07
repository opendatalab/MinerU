// 上传控件只通过 Gradio 的公开 upload/props 接口写入文件，不读取框架内部状态。
({ element, props, upload, accepted, imageExtensions }) => {
    let id = "";
    let localFile = null;
    let uploadedFile = null;
    let pending = false;
    let bridged = "";
    // DOM 引用按需取得，兼容组件对属性更新的模板协调。
    const input = () => element.querySelector('input[type="file"]');
    const label = () => element.querySelector(".mineru-upload-label");
    const status = () => element.querySelector(".mineru-upload-status");
    const retry = () => element.querySelector(".mineru-upload-retry");
    const text = key => window.__mineruI18n?.text(key) || key;
    const identity = () => Array.from(crypto.getRandomValues(new Uint8Array(16)), value => value.toString(16).padStart(2, "0")).join("");
    const isImage = name => imageExtensions.some(extension => name.toLowerCase().endsWith(`.${extension}`));
    // 文件名和服务器错误只显示为文本；错误状态允许使用同一文件重试。
    const display = (name, message = "", failed = false) => {
        label().textContent = name || text("upload_drop");
        status().textContent = message;
        status().setAttribute("role", failed ? "alert" : "status");
        retry().hidden = !failed;
        retry().textContent = text("upload_retry");
    };
    // 每次文件选择有独立闭包身份，旧上传即使成功也不会写入原生文件桥接组件。
    const submit = async file => {
        if (!file || !accepted.some(extension => file.name.toLowerCase().endsWith(extension))) {
            display(file?.name, text("upload_unsupported"), false);
            return;
        }
        const previous = localFile === file ? uploadedFile : null;
        id = identity();
        const current = id;
        localFile = file;
        uploadedFile = null;
        pending = true;
        window.__mineruPdfPreview.local(file, current);
        display(file.name, text("uploading"));
        props.value = JSON.stringify({ id: current, stage: "selected" });
        try {
            // 已上传但 PDF 准备失败时只重试准备，不重复上传图片。
            const result = previous || await upload(file);
            if (id !== current) return;
            const data = { ...result, orig_name: file.name, size: file.size, mime_type: file.type, meta: { _type: "gradio.FileData" } };
            if (!window.__mineruPdfPreview.bind(data, current)) return;
            uploadedFile = data;
            pending = false;
            display(file.name, isImage(file.name) ? text("preparing_pdf") : "");
            // 让清空事件先提交，重复准备同一服务器路径仍能触发文件变更。
            await new Promise(resolve => setTimeout(resolve, 0));
            if (id === current) props.value = JSON.stringify({ id: current, stage: "uploaded", file: data });
        } catch (error) {
            if (id !== current) return;
            pending = false;
            display(file.name, String(error?.message || error), true);
        }
    };
    // 清除只撤销本控件身份，真正文件取消和租期回收由原生组件事件协调。
    const invalidate = () => {
        id = "";
        localFile = null;
        uploadedFile = null;
        pending = false;
        input().value = "";
        display("");
    };
    window.__mineruUpload = {
        isImage, invalidate,
        // 同一次控件 change 的重复派发不再写回相同 FileData，避免触发重复源事件。
        bridge(value) {
            let receipt;
            try { receipt = JSON.parse(value); } catch { return { __type__: "update" }; }
            const key = `${receipt.id}:${receipt.stage}`;
            if (receipt.id !== window.__mineruPdfPreview.sourceId || key === bridged) return { __type__: "update" };
            bridged = key;
            return receipt.stage === "uploaded" ? receipt.file : null;
        },
        // 启动脚本晚于上传组件挂载时补齐初始语言。
        localize() { if (!localFile && !uploadedFile) display(""); },
        // 示例写入隐藏文件组件时同步文件名，不能把它关联到旧本地 File。
        sync(file) {
            if (pending || (file?.path && uploadedFile?.path === file.path)) return;
            invalidate();
            if (file) display(file.orig_name || file.path.split(/[\\/]/).pop());
        },
        // 输入准备成功或失败均在上传控件中显示，原图不会作为降级预览出现。
        prepared(error) { display(localFile?.name || label().textContent, error || "", Boolean(error && localFile)); },
    };
    element.addEventListener("change", event => {
        if (event.target === input()) submit(input().files?.[0]);
    });
    element.addEventListener("dragover", event => { event.preventDefault(); });
    element.addEventListener("drop", event => {
        event.preventDefault();
        submit(event.dataTransfer?.files?.[0]);
    });
    retry().addEventListener("click", () => submit(localFile));
    // 剪贴板与文件选择共用唯一上传流程，不再向隐藏 gr.File 注入合成 FileList。
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
            // 无扩展名截图沿用 MIME 命名；文件不经过 FileList 注入或二次上传。
            const extension = mimeExtensions[file.type];
            return !/\.[^.]+$/.test(file.name) && extension
                ? new File([file], `clipboard-${Date.now()}.${extension}`, { type: file.type }) : file;
        });
        const file = normalized.find(item => accepted.some(extension => item.name.toLowerCase().endsWith(extension)));
        if (file) { event.preventDefault(); submit(file); }
    };
    document.addEventListener("paste", paste);
    window.addEventListener("pagehide", () => document.removeEventListener("paste", paste), { once: true });
    display("");
}
