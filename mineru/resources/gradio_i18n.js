(messages) => {
    // 首选语言独立于服务端和 Gradio 当前 locale，所有中文地区统一简体中文。
    const resolveLocale = () => {
        const primary = typeof navigator === "undefined" ? "" : navigator.languages?.[0] || navigator.language || "";
        return /^zh(?:-|$)/i.test(primary) ? "zh" : "en";
    };
    // 参数按字面替换，不能把用户文件名中的美元符号解释为替换表达式。
    const text = (key, values = {}, locale = resolveLocale()) => {
        const template = messages[key]?.[locale === "zh" ? 1 : 0] ?? key;
        return template.replace(/\{(\w+)\}/g, (match, name) => String(values[name] ?? match));
    };
    // 只识别应用词典中的完整固定消息，不改写外部错误详情。
    const message = (value) => {
        const entry = Object.entries(messages).find(([, pair]) => pair.includes(value));
        return entry ? text(entry[0]) : value;
    };
    // 仅校正 MinerU 自己定义的组件文案；框架的上传提示、工具栏和页脚保留原生语言。
    const localizeProjectLabels = (root) => {
        const selectors = [
            ".mineru-upload-file > label", ".mineru-force-ocr", ".mineru-actions button",
            ".mineru-tier-label", ".mineru-page-handle-a label", ".mineru-page-handle-b label",
            ".mineru-kit-pdf-preview > label", ".mineru-kit-image-preview > label",
            ".mineru-markdown-tabs [role=tab]", ".mineru-markdown-tabs .tab-wrapper [aria-hidden] button",
            ".mineru-kit-download-options button", "#mineru-kit-examples > .label",
        ];
        root.querySelectorAll(selectors.join(",")).forEach((element) => {
            const walker = document.createTreeWalker(element, NodeFilter.SHOW_TEXT);
            let node;
            while ((node = walker.nextNode())) {
                const original = node.textContent.trim();
                let translated = message(original);
                const tier = /^(?:解析 tier：|Parsing tier: )(\w+)$/.exec(original);
                if (tier) translated = text("tier_value", { tier: tier[1], notice: "" });
                if (translated !== original) node.textContent = node.textContent.replace(original, () => translated);
            }
        });
    };
    // 自定义 HTML 只处理明确标记的叶节点或属性，绝不遍历文档正文做替换。
    const localize = (root = document) => {
        const locale = resolveLocale();
        root.querySelectorAll("[data-mineru-i18n-key]").forEach((item) => {
            const key = item.getAttribute("data-mineru-i18n-key");
            const value = item.getAttribute(`data-mineru-i18n-${locale}`) ?? text(key);
            const attributes = item.getAttribute("data-mineru-i18n-attr");
            if (attributes) {
                attributes.split(" ").forEach((name) => {
                    if (item.getAttribute(name) !== value) item.setAttribute(name, value);
                });
            } else if (item.textContent !== value) item.textContent = value;
        });
        localizeProjectLabels(root);
        if (document.title !== text("header_title")) document.title = text("header_title");
    };
    return { text, message, resolveLocale, localize };
}
