() => {
    const APP_SCRIPT_VERSION = "v1-only-options";
    if (window.__mineruGradioAppInstalled === APP_SCRIPT_VERSION) {
        return;
    }
    window.__mineruGradioAppInstalled = APP_SCRIPT_VERSION;

    const OFFICE_PREVIEW_NOTICE_STORAGE_KEY = "mineru.officePreviewNoticeIgnored";
    const CLIPBOARD_MIME_EXTENSIONS = {
        "image/png": "png",
        "image/jpeg": "jpg",
        "image/webp": "webp",
        "image/gif": "gif",
        "image/bmp": "bmp",
        "image/tiff": "tiff",
        "application/pdf": "pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation": "pptx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
    };
    // 只把中文浏览器语言映射为中文；其他所有语言统一降级到英文。
    const normalizeMineruLocale = (locale) => {
        const normalized = String(locale || "").toLowerCase();
        if (normalized.startsWith("zh")) {
            return "zh";
        }
        return "en";
    };

    // 以浏览器首选语言为准；非中文语言包括 en/ja/ko/fr 等都统一使用英文文案。
    const resolveMineruLocale = () => {
        if (typeof navigator !== "undefined") {
            const languages = Array.from(navigator.languages || []);
            const primaryLocale = languages[0] || navigator.language;
            if (primaryLocale) {
                return normalizeMineruLocale(primaryLocale);
            }
        }
        return normalizeMineruLocale(document.documentElement.getAttribute("lang"));
    };

    // Gradio 只会自动翻译组件属性；header/status 这类自定义 HTML 需要前端按浏览器语言补一次。
    const localizeMineruCustomText = () => {
        const locale = resolveMineruLocale();
        document.querySelectorAll("[data-mineru-i18n-key]").forEach((item) => {
            const localizedText = item.getAttribute(`data-mineru-i18n-${locale}`)
                || item.getAttribute("data-mineru-i18n-en");
            if (localizedText !== null && item.textContent !== localizedText) {
                item.textContent = localizedText;
            }
        });
    };

    // 读取浏览器本地偏好时做容错，避免隐私模式禁用 localStorage 影响页面初始化。
    const getOfficePreviewNoticeIgnored = () => {
        try {
            return localStorage.getItem(OFFICE_PREVIEW_NOTICE_STORAGE_KEY) === "1";
        } catch (error) {
            return false;
        }
    };

    // 保存“不再提示”偏好；失败时仅降级为本次点击隐藏，不阻断预览。
    const setOfficePreviewNoticeIgnored = () => {
        try {
            localStorage.setItem(OFFICE_PREVIEW_NOTICE_STORAGE_KEY, "1");
        } catch (error) {
            return false;
        }
        return true;
    };

    const findOfficePreviewNotices = () =>
        document.querySelectorAll(".office-preview-notice");

    // 根据浏览器持久偏好隐藏新挂载的 Office 预览提示。
    const applyOfficePreviewNoticePreference = () => {
        if (!getOfficePreviewNoticeIgnored()) {
            return;
        }
        findOfficePreviewNotices().forEach((notice) => {
            notice.classList.add("is-dismissed");
        });
    };

    // 自定义 HTML 由 Gradio 动态重绘，统一在 DOM 变更后补本地化和忽略状态。
    const refreshMineruCustomHtml = () => {
        localizeMineruCustomText();
        applyOfficePreviewNoticePreference();
    };

    const findUploadFileInput = () => {
        const uploadRoot = document.querySelector(".mineru-upload-file");
        if (!uploadRoot) {
            return null;
        }
        return uploadRoot.querySelector('input[type="file"]');
    };

    // 读取上传控件 accept 规则，后续粘贴文件仍复用 gr.File 的支持格式边界。
    const getUploadAcceptedTypes = (uploadInput) => {
        const accept = uploadInput?.getAttribute("accept") || "";
        return accept.split(",").map((item) => item.trim().toLowerCase()).filter(Boolean);
    };

    // 判断剪贴板文件是否匹配 gr.File 当前支持的扩展名或 MIME 类型。
    const fileMatchesAcceptedType = (file, acceptedTypes) => {
        if (!acceptedTypes.length) {
            return true;
        }
        const name = (file.name || "").toLowerCase();
        const type = (file.type || "").toLowerCase();
        return acceptedTypes.some((accepted) => {
            if (accepted.startsWith(".")) {
                return name.endsWith(accepted);
            }
            if (accepted.endsWith("/*")) {
                return type.startsWith(accepted.slice(0, -1));
            }
            return type === accepted;
        });
    };

    // 为截图等无文件名剪贴板图片补一个扩展名，确保后端按普通图片文件解析。
    const buildClipboardFileName = (file) => {
        const type = (file.type || "").toLowerCase();
        const extension = CLIPBOARD_MIME_EXTENSIONS[type];
        if (!extension) {
            return "";
        }
        const timestamp = new Date().toISOString()
            .replace(/[-:]/g, "")
            .replace(/[.].+/, "")
            .replace("T", "-");
        const prefix = type.startsWith("image/") ? "clipboard-image" : "clipboard-file";
        return `${prefix}-${timestamp}.${extension}`;
    };

    // 保留浏览器暴露的原始文件；仅在文件名缺少扩展名时复制一份并补齐名称。
    const normalizeClipboardFile = (file) => {
        if (/[.][^.]+$/.test(file.name || "")) {
            return file;
        }
        const fileName = buildClipboardFileName(file);
        if (!fileName || typeof File === "undefined") {
            return file;
        }
        return new File([file], fileName, {
            type: file.type,
            lastModified: file.lastModified || Date.now(),
        });
    };

    // 同时兼容剪贴板 files 与 items，两种入口在不同浏览器里暴露情况不一致。
    const collectClipboardFiles = (clipboardData) => {
        const files = Array.from(clipboardData.files || []);
        if (files.length) {
            return files;
        }
        return Array.from(clipboardData.items || [])
            .filter((item) => item.kind === "file")
            .map((item) => item.getAsFile())
            .filter(Boolean);
    };

    // 构造只包含目标文件的 FileList；部分浏览器不允许构造 DataTransfer，需要降级处理。
    const createUploadFileList = (file) => {
        try {
            const transfer = new DataTransfer();
            transfer.items.add(file);
            return transfer.files;
        } catch (error) {
            return null;
        }
    };

    // 把文件列表赋值给 gr.File 的原生 input，并触发 Gradio 监听的变更事件。
    const assignClipboardFileToUpload = (uploadInput, uploadFiles) => {
        if (!uploadFiles) {
            return false;
        }
        try {
            uploadInput.files = uploadFiles;
        } catch (error) {
            return false;
        }
        uploadInput.dispatchEvent(new Event("input", { bubbles: true }));
        uploadInput.dispatchEvent(new Event("change", { bubbles: true }));
        return true;
    };

    // 将剪贴板文件注入现有 gr.File input，避免为图片、PDF、Office 维护第二套上传链路。
    const uploadClipboardFile = (event) => {
        const clipboardData = event.clipboardData;
        const uploadInput = findUploadFileInput();
        if (!clipboardData || !uploadInput) {
            return false;
        }

        const acceptedTypes = getUploadAcceptedTypes(uploadInput);
        const rawClipboardFiles = clipboardData.files || null;
        const clipboardFiles = collectClipboardFiles(clipboardData)
            .map((rawFile) => ({ rawFile, uploadFile: normalizeClipboardFile(rawFile) }))
            .filter(({ uploadFile }) => fileMatchesAcceptedType(uploadFile, acceptedTypes));
        if (!clipboardFiles.length) {
            return false;
        }

        const { rawFile, uploadFile } = clipboardFiles[0];
        const uploadFiles = createUploadFileList(uploadFile)
            || (
                rawClipboardFiles?.length === 1
                && rawClipboardFiles[0] === rawFile
                && rawFile === uploadFile
                    ? rawClipboardFiles
                    : null
            );
        return assignClipboardFileToUpload(uploadInput, uploadFiles);
    };

    refreshMineruCustomHtml();
    requestAnimationFrame(() => {
        refreshMineruCustomHtml();
    });
    if (typeof MutationObserver !== "undefined") {
        const uiObserver = new MutationObserver(() => {
            refreshMineruCustomHtml();
        });
        uiObserver.observe(document.body, { childList: true, subtree: true });
    }

    document.addEventListener("click", (event) => {
        const target = event.target;
        if (!(target instanceof Element)) {
            return;
        }
        if (target.closest(".office-preview-ignore-forever")) {
            const notice = target.closest(".office-preview-notice");
            if (setOfficePreviewNoticeIgnored()) {
                applyOfficePreviewNoticePreference();
            } else {
                notice?.classList.add("is-dismissed");
            }
            return;
        }
        if (target.closest(".office-preview-ignore-once")) {
            target.closest(".office-preview-notice")?.classList.add("is-dismissed");
            return;
        }
    });

    document.addEventListener("paste", (event) => {
        if (uploadClipboardFile(event)) {
            event.preventDefault();
        }
    });
}
