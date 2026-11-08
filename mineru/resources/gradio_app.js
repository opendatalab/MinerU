(viewerEntry) => {
    const preview = __MINERU_PDF_PREVIEW__;
    const APP_SCRIPT_VERSION = "v10-page-gestures";
    if (window.__mineruGradioAppInstalled === APP_SCRIPT_VERSION) {
        preview("setup", viewerEntry);
        return;
    }
    window.__mineruGradioAppInstalled = APP_SCRIPT_VERSION;

    const OFFICE_PREVIEW_NOTICE_STORAGE_KEY = "mineru.officePreviewNoticeIgnored";
    const i18n = window.__mineruI18n = __MINERU_I18N__;
    preview("setup", viewerEntry);
    window.__mineruUpload?.localize();
    window.__mineruTier?.localize();
    (__MINERU_TRACE__)();
    const statusTimer = (__MINERU_STATUS_TIMER__)(i18n);
    window.__mineruStatusPanel = statusTimer;
    // HTML 回调和前端事件使用同一份词典与首选语言规则。
    const localizeMineruCustomText = () => i18n.localize();

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
        statusTimer.sync();
        localizeMineruCustomText();
        applyOfficePreviewNoticePreference();
        window.__mineruPageRange?.localize();
    };

    refreshMineruCustomHtml();
    requestAnimationFrame(() => {
        refreshMineruCustomHtml();
    });
    if (typeof MutationObserver !== "undefined") {
        let refreshPending = false;
        const observationOptions = {
            childList: true, subtree: true, characterData: true, attributes: true,
            attributeFilter: ["title", "aria-label", "data-mineru-i18n-en", "data-mineru-i18n-zh"],
        };
        const uiObserver = new MutationObserver((mutations) => {
            // 本地计时、选页预览与排队动画无需重新扫描整页，避免递归本地化。
            if (mutations.every(({ target }) => {
                const element = target.nodeType === 1 ? target : target.parentElement;
                return element?.closest?.("[data-mineru-local-timer], [data-mineru-local-animation], [data-mineru-page-live]");
            })) {
                return;
            }
            if (refreshPending) return;
            refreshPending = true;
            // 合并同帧重绘，并隔离自身写入，避免本地化与计时更新形成微任务循环。
            requestAnimationFrame(() => {
                refreshPending = false;
                uiObserver.disconnect();
                try {
                    refreshMineruCustomHtml();
                } finally {
                    uiObserver.observe(document.body, observationOptions);
                }
            });
        });
        uiObserver.observe(document.body, observationOptions);
    }

    document.addEventListener("click", (event) => {
        const target = event.target;
        if (!(target instanceof Element)) {
            return;
        }
        const convertRoot = target.closest(".mineru-convert-button");
        const convert = convertRoot?.matches("button") ? convertRoot : convertRoot?.querySelector("button");
        if (convert && !convert.disabled && convert.getAttribute("aria-disabled") !== "true") {
            statusTimer.showPreparing();
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
}
