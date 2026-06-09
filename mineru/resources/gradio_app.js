() => {
    const POPOVER_SCRIPT_VERSION = "office-preview-dismiss-v1";
    if (window.__mineruAdvancedPopoverInstalled === POPOVER_SCRIPT_VERSION) {
        return;
    }
    window.__mineruAdvancedPopoverInstalled = POPOVER_SCRIPT_VERSION;

    const POPOVER_OPEN_CLASS = "mineru-advanced-popover-open";
    const CLIENT_OPTIONS_VISIBLE_CLASS = "mineru-show-client-options";
    const IMAGE_ANALYSIS_VISIBLE_CLASS = "mineru-show-image-analysis";
    const OFFICE_PREVIEW_NOTICE_STORAGE_KEY = "mineru.officePreviewNoticeIgnored";
    const OPEN_DELAY_MS = 120;
    const CLOSE_DELAY_MS = 280;
    const ANIMATION_DELAY_MS = 140;
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
        refreshMineruOptionVisibility();
    };

    // 兼容 Gradio 将 elem_classes 挂到按钮自身或按钮外层容器的两种 DOM 结构。
    const findButton = () => document.querySelector(
        "button.mineru-advanced-open, .mineru-advanced-open button, .mineru-advanced-open"
    );
    const findPopover = () => document.querySelector(".mineru-advanced-popover");
    const findBackendRoot = () => document.querySelector(".mineru-backend-select");
    const findEffortRoot = () => document.querySelector(".mineru-hybrid-effort");
    let openTimer = null;
    let closeTimer = null;
    let visibilityTimer = null;
    let hoverHandlersInstalled = false;

    // 读取 Gradio Dropdown 当前值；value 属性比可见文本更稳定，避免中英文文案影响判断。
    const getBackendValue = () => {
        const backendRoot = findBackendRoot();
        const backendControl = backendRoot?.querySelector('[role="listbox"]');
        return (backendControl?.value || backendControl?.textContent || "").trim();
    };

    // 读取 Hybrid effort 当前值；控件在非 hybrid 后端会被 Gradio 隐藏，缺失时按空值处理。
    const getEffortValue = () => {
        const effortRoot = findEffortRoot();
        const checkedRadio = effortRoot?.querySelector(
            'input[type="radio"]:checked, input[type="radio"][aria-checked="true"]'
        );
        return (checkedRadio?.value || "").trim();
    };

    // 根据当前 backend/effort 刷新前端状态类，避免依赖 Gradio 重新挂载隐藏组件。
    const refreshMineruOptionVisibility = () => {
        const backend = getBackendValue();
        const effort = getEffortValue();
        const showClientOptions = backend.endsWith("http-client");
        const showImageAnalysis = backend.startsWith("vlm")
            || (backend.startsWith("hybrid") && effort === "high");

        document.body.classList.toggle(CLIENT_OPTIONS_VISIBLE_CLASS, showClientOptions);
        document.body.classList.toggle(IMAGE_ANALYSIS_VISIBLE_CLASS, showImageAnalysis);
        if (document.body.classList.contains(POPOVER_OPEN_CLASS)) {
            positionPopover();
        }
    };

    // Gradio 控件会异步写回 value，延后一帧再读可以覆盖 Dropdown option 点击和 Radio 切换。
    const queueMineruOptionVisibilityRefresh = () => {
        requestAnimationFrame(() => {
            refreshMineruOptionVisibility();
            requestAnimationFrame(refreshMineruOptionVisibility);
        });
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

    // 修正 Gradio Dropdown 在 fixed 浮层里按视口定位导致的下拉列表漂移。
    const positionAdvancedDropdowns = () => {
        const popover = findPopover();
        if (!popover || !document.body.classList.contains(POPOVER_OPEN_CLASS)) {
            return;
        }

        popover.querySelectorAll("ul.options").forEach((options) => {
            const wrap = options.closest(".wrap");
            if (!wrap) {
                return;
            }

            popover.querySelectorAll(".wrap").forEach((item) => {
                item.style.removeProperty("z-index");
            });

            const wrapRect = wrap.getBoundingClientRect();
            const popoverRect = popover.getBoundingClientRect();
            const viewportPadding = 12;
            const gap = 6;
            const belowSpace = Math.max(0, popoverRect.bottom - wrapRect.bottom - viewportPadding);
            const aboveSpace = Math.max(0, wrapRect.top - popoverRect.top - viewportPadding);
            const naturalHeight = Math.max(36, Math.min(options.scrollHeight || 220, 240));
            const openBelow = belowSpace >= Math.min(180, naturalHeight) || belowSpace >= aboveSpace;
            const availableHeight = Math.max(84, openBelow ? belowSpace : aboveSpace);
            const height = Math.min(naturalHeight, availableHeight);
            const top = openBelow ? wrap.offsetHeight + gap : -height - gap;

            wrap.style.setProperty("z-index", "1003", "important");
            options.style.setProperty("position", "absolute", "important");
            options.style.setProperty("left", "0", "important");
            options.style.setProperty("top", `${top}px`, "important");
            options.style.setProperty("bottom", "auto", "important");
            options.style.setProperty("width", `${wrapRect.width}px`, "important");
            options.style.setProperty("max-height", `${height}px`, "important");
            options.style.setProperty("z-index", "1004", "important");
        });
    };

    // 只在真正支持鼠标悬浮的桌面环境启用 hover 浮窗，触屏设备继续使用点击兜底。
    const supportsHoverPopover = () => (
        typeof window.matchMedia === "function"
        && window.matchMedia("(hover: hover) and (pointer: fine)").matches
    );

    // 取消尚未执行的打开/关闭计时，避免鼠标在按钮和气泡之间移动时闪烁。
    const cancelPopoverTimers = () => {
        if (openTimer !== null) {
            clearTimeout(openTimer);
            openTimer = null;
        }
        if (closeTimer !== null) {
            clearTimeout(closeTimer);
            closeTimer = null;
        }
        if (visibilityTimer !== null) {
            clearTimeout(visibilityTimer);
            visibilityTimer = null;
        }
    };

    // 清理旧版 display 开关留下的内联样式，后续统一交给 CSS 的可见性和动画状态控制。
    const clearLegacyPopoverDisplay = (popover) => {
        if (popover) {
            popover.style.removeProperty("display");
        }
    };

    // 用内联 important 同步动画属性，避免 Gradio 自动 scoped CSS 抬高隐藏规则优先级。
    const applyOpenPopoverStyle = (popover) => {
        if (!popover) {
            return;
        }
        popover.style.setProperty("visibility", "visible", "important");
        popover.style.setProperty("opacity", "1", "important");
        popover.style.setProperty("pointer-events", "auto", "important");
        popover.style.setProperty("transform", "translateY(0) scale(1)", "important");
    };

    // 关闭时先取消交互并播放淡出，动画结束后再隐藏可见性。
    const applyClosedPopoverStyle = (popover) => {
        if (!popover) {
            return;
        }
        popover.style.setProperty("opacity", "0", "important");
        popover.style.setProperty("pointer-events", "none", "important");
        popover.style.setProperty("transform", "translateY(-4px) scale(0.985)", "important");
        visibilityTimer = window.setTimeout(() => {
            if (!document.body.classList.contains(POPOVER_OPEN_CLASS)) {
                popover.style.setProperty("visibility", "hidden", "important");
            }
            visibilityTimer = null;
        }, ANIMATION_DELAY_MS);
    };

    // 等待 Gradio 完成下拉列表挂载后，再按当前输入框位置校正。
    const queueDropdownPosition = () => {
        requestAnimationFrame(() => {
            requestAnimationFrame(positionAdvancedDropdowns);
        });
    };

    // 根据高级选项按钮的位置，把气泡贴在左侧控制栏右侧并限制在视口内。
    const positionPopover = () => {
        const button = findButton();
        const popover = findPopover();
        if (!button || !popover) {
            return;
        }

        const buttonRect = button.getBoundingClientRect();
        const preferredWidth = Math.min(420, window.innerWidth - 36);
        const left = Math.min(
            Math.max(18, buttonRect.right + 12),
            Math.max(18, window.innerWidth - preferredWidth - 18)
        );
        const availableHeight = Math.max(260, window.innerHeight - 36);
        const measuredHeight = Math.min(
            popover.scrollHeight || 520,
            availableHeight,
            Math.round(window.innerHeight * 0.7)
        );
        const centeredTop = buttonRect.top + buttonRect.height / 2 - measuredHeight / 2;
        const top = Math.min(
            Math.max(18, centeredTop),
            Math.max(18, window.innerHeight - measuredHeight - 18)
        );

        popover.style.setProperty("--mineru-popover-left", `${left}px`);
        popover.style.setProperty("--mineru-popover-top", `${top}px`);
    };

    // 打开气泡时保持组件 DOM 挂载，只切换 body 状态类并重新计算位置。
    const openPopover = () => {
        const popover = findPopover();
        cancelPopoverTimers();
        clearLegacyPopoverDisplay(popover);
        document.body.classList.add(POPOVER_OPEN_CLASS);
        applyOpenPopoverStyle(popover);
        requestAnimationFrame(() => {
            positionPopover();
            queueDropdownPosition();
        });
    };

    // 收起气泡时不卸载 Gradio 控件，用户已经修改的高级配置会保留在原组件上。
    const closePopover = () => {
        const popover = findPopover();
        cancelPopoverTimers();
        clearLegacyPopoverDisplay(popover);
        document.body.classList.remove(POPOVER_OPEN_CLASS);
        applyClosedPopoverStyle(popover);
    };

    // 鼠标进入按钮后延迟打开，防止只是路过按钮时频繁弹出。
    const scheduleHoverOpen = () => {
        if (!supportsHoverPopover()) {
            return;
        }
        cancelPopoverTimers();
        openTimer = window.setTimeout(() => {
            openTimer = null;
            openPopover();
        }, OPEN_DELAY_MS);
    };

    // 鼠标离开按钮或气泡后延迟关闭，给用户从按钮移动到气泡留出缓冲时间。
    const scheduleHoverClose = () => {
        if (!supportsHoverPopover()) {
            return;
        }
        cancelPopoverTimers();
        closeTimer = window.setTimeout(() => {
            closeTimer = null;
            closePopover();
        }, CLOSE_DELAY_MS);
    };

    // 给真实桌面指针安装 hover 事件；如果 Gradio 稍后才挂载 DOM，就通过观察器重试。
    const installHoverPopoverHandlers = () => {
        if (hoverHandlersInstalled || !supportsHoverPopover()) {
            return;
        }
        const button = findButton();
        const popover = findPopover();
        if (!button || !popover) {
            return;
        }
        button.addEventListener("pointerenter", scheduleHoverOpen);
        button.addEventListener("pointerleave", scheduleHoverClose);
        button.addEventListener("mouseenter", scheduleHoverOpen);
        button.addEventListener("mouseleave", scheduleHoverClose);
        popover.addEventListener("pointerenter", cancelPopoverTimers);
        popover.addEventListener("pointerleave", scheduleHoverClose);
        popover.addEventListener("mouseenter", cancelPopoverTimers);
        popover.addEventListener("mouseleave", scheduleHoverClose);
        hoverHandlersInstalled = true;
    };

    refreshMineruCustomHtml();
    installHoverPopoverHandlers();
    requestAnimationFrame(() => {
        refreshMineruCustomHtml();
        installHoverPopoverHandlers();
    });
    if (typeof MutationObserver !== "undefined") {
        const uiObserver = new MutationObserver(() => {
            refreshMineruCustomHtml();
            installHoverPopoverHandlers();
        });
        uiObserver.observe(document.body, { childList: true, subtree: true });
    }

    document.addEventListener("click", (event) => {
        const target = event.target;
        if (!(target instanceof Element)) {
            return;
        }
        queueMineruOptionVisibilityRefresh();
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
        if (target.closest(".mineru-advanced-open")) {
            if (document.body.classList.contains(POPOVER_OPEN_CLASS)) {
                closePopover();
            } else {
                openPopover();
            }
            return;
        }
        if (target.closest(".mineru-advanced-popover")) {
            queueDropdownPosition();
        }
        if (!target.closest(".mineru-advanced-popover")) {
            closePopover();
        }
    });

    document.addEventListener("focusin", (event) => {
        const target = event.target;
        if (target instanceof Element && target.closest(".mineru-advanced-popover")) {
            queueDropdownPosition();
        }
    });

    document.addEventListener("input", (event) => {
        const target = event.target;
        queueMineruOptionVisibilityRefresh();
        if (target instanceof Element && target.closest(".mineru-advanced-popover")) {
            queueDropdownPosition();
        }
    });

    document.addEventListener("change", () => {
        queueMineruOptionVisibilityRefresh();
    });

    document.addEventListener("keydown", (event) => {
        if (event.key === "Escape") {
            closePopover();
            return;
        }
        const target = event.target;
        if (target instanceof Element && target.closest(".mineru-advanced-popover")) {
            queueDropdownPosition();
        }
    });

    document.addEventListener("paste", (event) => {
        if (uploadClipboardFile(event)) {
            event.preventDefault();
        }
    });

    window.addEventListener("resize", () => {
        if (document.body.classList.contains(POPOVER_OPEN_CLASS)) {
            positionPopover();
            positionAdvancedDropdowns();
        }
    });
}
