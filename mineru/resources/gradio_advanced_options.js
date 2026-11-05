() => {
    let available = false;
    let opened = false;
    let openTimer;
    let closeTimer;
    // 所有查找都使用自有类名，兼容 Gradio 重绘及按钮外层结构。
    const root = () => document.querySelector(".mineru-advanced-open");
    const button = () => {
        const element = root();
        return element?.matches("button") ? element : element?.querySelector("button");
    };
    const panel = () => document.querySelector(".mineru-advanced-popover");
    // 浮窗优先位于按钮右侧，窄屏改为按钮下方，并限制在当前 iframe 视口内。
    const position = () => {
        const trigger = button();
        const popup = panel();
        if (!trigger || !popup || !opened) return;
        const bounds = trigger.getBoundingClientRect();
        const width = Math.min(300, window.innerWidth - 24);
        popup.style.width = `${width}px`;
        const height = popup.getBoundingClientRect().height;
        const fitsRight = bounds.right + 12 + width <= window.innerWidth - 12;
        const left = fitsRight ? bounds.right + 12 : Math.max(12, Math.min(bounds.left, window.innerWidth - width - 12));
        const preferredTop = fitsRight ? bounds.top : bounds.bottom + 8;
        const top = Math.max(12, Math.min(preferredTop, window.innerHeight - height - 12));
        popup.style.left = `${left}px`;
        popup.style.top = `${top}px`;
    };
    // 幂等同步可见性和无障碍属性，隐藏浮窗不重置原生 checkbox 的值。
    const sync = () => {
        const triggerRoot = root();
        const trigger = button();
        const popup = panel();
        if (!triggerRoot || !trigger || !popup) return;
        const visible = available && opened;
        // 属性只在变化时写入，避免与界面 MutationObserver 形成循环。
        if (triggerRoot.dataset.available !== String(available)) triggerRoot.dataset.available = String(available);
        if (popup.dataset.open !== String(visible)) popup.dataset.open = String(visible);
        if (popup.inert !== !visible) popup.inert = !visible;
        for (const [name, value] of Object.entries({
            "aria-expanded": String(visible), "aria-haspopup": "dialog", "aria-controls": "mineru-advanced-popover",
        })) {
            if (trigger.getAttribute(name) !== value) trigger.setAttribute(name, value);
        }
        if (popup.getAttribute("role") !== "dialog") popup.setAttribute("role", "dialog");
        if (popup.getAttribute("aria-labelledby") !== "mineru-llm-enhancement-heading") {
            popup.setAttribute("aria-labelledby", "mineru-llm-enhancement-heading");
        }
    };
    // 统一取消悬停定时器，避免清除文件后又被延迟打开。
    const stopTimers = () => { clearTimeout(openTimer); clearTimeout(closeTimer); };
    // 关闭只影响浮窗，不改变增强选择。
    const close = () => { stopTimers(); opened = false; sync(); };
    // 展开时重新计算位置，兼容滚动、窗口缩放和语言切换。
    const open = () => { stopTimers(); opened = available; sync(); position(); };
    // 文件组件提供真实路径，前端仅负责显隐；后端仍独立校验格式。
    const setFile = (file, extensions) => {
        const path = (typeof file === "string" ? file : file?.path) || "";
        available = extensions.some((extension) => path.toLowerCase().endsWith(`.${extension}`));
        close();
    };
    // 判断事件是否仍处于按钮或浮窗内部，穿过两者间距时延迟关闭。
    const inside = (target) => target instanceof Element
        && Boolean(target.closest(".mineru-advanced-open, .mineru-advanced-popover"));
    document.addEventListener("pointerover", (event) => {
        if (!inside(event.target)) return;
        clearTimeout(closeTimer);
        if (!opened && !inside(event.relatedTarget) && event.pointerType !== "touch") {
            openTimer = setTimeout(open, 120);
        }
    });
    document.addEventListener("pointerout", (event) => {
        if (inside(event.target) && !inside(event.relatedTarget)) {
            clearTimeout(openTimer);
            closeTimer = setTimeout(() => {
                if (!inside(document.activeElement)) close();
            }, 280);
        }
    });
    document.addEventListener("click", (event) => {
        if (!(event.target instanceof Element)) return;
        if (event.target.closest(".mineru-advanced-open")) {
            if (opened) close();
            else open();
            if (opened && event.detail === 0) panel()?.querySelector("input")?.focus();
        } else if (!event.target.closest(".mineru-advanced-popover")) close();
    });
    document.addEventListener("keydown", (event) => {
        if (event.key === "Escape" && opened) { close(); button()?.focus(); }
    });
    document.addEventListener("focusout", (event) => {
        if (inside(event.target) && !inside(event.relatedTarget)) close();
    });
    window.addEventListener("resize", position);
    window.addEventListener("scroll", position, true);
    return { setFile, sync, close };
}
