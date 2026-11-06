// 目录只管理当前结果 iframe 的正文、焦点与滚动位置。
(labels) => {
    const article = document.querySelector("article.docvortex-document");
    const headings = [...article.querySelectorAll("[data-mineru-outline-label]")];
    if (!headings.length) return;
    const language = navigator.languages?.[0] || navigator.language || "";
    const locale = /^zh(?:-|$)/i.test(language) ? 1 : 0;
    const wide = window.matchMedia("(min-width: 900px)");
    const controller = new AbortController();
    const { signal } = controller;
    let active = null;
    let target = null;
    let positions = [];
    let frame = 0;
    let layoutDirty = true;
    let disposed = false;

    // 元素均由自有代码创建，文档标题始终按纯文本写入。
    const create = (tag, className) => {
        const element = document.createElement(tag);
        element.className = className;
        return element;
    };
    const toolbar = create("div", "mineru-outline-toolbar");
    const toggle = create("button", "mineru-outline-toggle");
    toggle.type = "button";
    toggle.textContent = labels.title[locale];
    toggle.setAttribute("aria-label", labels.title[locale]);
    const panel = create("nav", "mineru-outline-panel");
    // 控件 id 同样避让源文档书签。
    let panelId = "mineru-outline-panel";
    while (document.getElementById(panelId)) panelId += "-toc";
    panel.id = panelId;
    panel.setAttribute("aria-label", labels.title[locale]);
    toggle.setAttribute("aria-controls", panelId);
    const backdrop = create("div", "mineru-outline-backdrop");
    backdrop.setAttribute("aria-hidden", "true");
    toolbar.append(toggle);
    document.body.append(toolbar, backdrop, panel);
    document.body.classList.add("mineru-outline-enabled");
    const root = document.createElement("ul");
    panel.append(root);
    const stack = [];
    const entries = [];

    // 初始仅显示两层树节点；层级跳跃直接归属最近的较低级标题。
    for (const heading of headings) {
        const level = Number(heading.tagName.slice(1));
        while (stack.length && stack.at(-1).level >= level) stack.pop();
        const parent = stack.at(-1) || null;
        const item = document.createElement("li");
        const row = create("div", "mineru-outline-row");
        const link = create("a", "mineru-outline-link");
        link.href = `#${encodeURIComponent(heading.id)}`;
        link.textContent = heading.dataset.mineruOutlineLabel;
        const children = document.createElement("ul");
        const entry = { heading, level, parent, item, row, link, children, branch: null, depth: stack.length };
        row.append(link);
        item.append(row, children);
        (parent?.children || root).append(item);
        children.hidden = entry.depth >= 1;
        entries.push(entry);
        stack.push(entry);
    }

    // 折叠开关只影响所属子树，章节链接单独负责导航。
    const expand = (entry, open) => {
        entry.children.hidden = !open;
        entry.branch.setAttribute("aria-expanded", String(open));
        entry.branch.setAttribute("aria-label", labels[open ? "collapse" : "expand"][locale]);
    };
    for (const entry of entries) {
        if (!entry.children.children.length) {
            entry.children.remove();
            continue;
        }
        const branch = create("button", "mineru-outline-branch");
        branch.type = "button";
        entry.branch = branch;
        entry.row.prepend(branch);
        expand(entry, !entry.children.hidden);
        branch.addEventListener("click", () => expand(entry, entry.children.hidden), { signal });
    }

    // 仅调整目录自己的滚动条，避免 scrollIntoView 带动外层 Gradio。
    const revealActive = () => {
        if (!active || panel.hidden) return;
        const rect = active.link.getBoundingClientRect();
        const bounds = panel.getBoundingClientRect();
        if (rect.top < bounds.top + 12) panel.scrollTop += rect.top - bounds.top - 12;
        else if (rect.bottom > bounds.bottom - 12) panel.scrollTop += rect.bottom - bounds.bottom + 12;
    };
    // 当前章节变化后展开祖先，用户仍可折叠其他章节。
    const markActive = (entry) => {
        if (active === entry) return;
        active?.link.removeAttribute("aria-current");
        active = entry;
        active.link.setAttribute("aria-current", "location");
        for (let ancestor = entry.parent; ancestor; ancestor = ancestor.parent) expand(ancestor, true);
        revealActive();
    };
    // 缓存文档坐标后用二分查找定位，长章节沿用其起始标题。
    const update = () => {
        frame = 0;
        if (layoutDirty) {
            positions = entries.map((entry) => entry.heading.getBoundingClientRect().top + window.scrollY);
            layoutDirty = false;
            if (target) window.scrollTo({ top: positions[entries.indexOf(target)] - 52, behavior: "instant" });
        }
        const top = window.scrollY + 54;
        let start = 0;
        let end = positions.length;
        while (start < end) {
            const middle = (start + end) >>> 1;
            if (positions[middle] <= top) start = middle + 1;
            else end = middle;
        }
        let index = Math.max(0, start - 1);
        if (window.scrollY > 0 && window.scrollY + window.innerHeight >= document.documentElement.scrollHeight - 2) {
            index = entries.length - 1;
        }
        markActive(entries[index]);
    };
    // 多个布局和滚动通知合并到一个动画帧，卸载后拒绝新任务。
    const schedule = (recalculate = false) => {
        if (disposed) return;
        layoutDirty ||= recalculate;
        if (!frame) frame = requestAnimationFrame(update);
    };
    // 展开模式由 iframe 自身宽度决定，工具条开关保留到下次模式切换。
    const setOpen = (open) => {
        panel.hidden = !open;
        backdrop.hidden = !open || wide.matches;
        toggle.setAttribute("aria-expanded", String(open));
        document.body.dataset.outlineOpen = String(open);
        schedule(true);
        revealActive();
    };
    // 导航只滚动本窗口，并通知溯源脚本停止此前定位的布局校正。
    const navigate = (event, entry) => {
        event.preventDefault();
        window.dispatchEvent(new CustomEvent("mineru-preview-navigate", { detail: "outline" }));
        target = entry;
        for (let ancestor = entry.heading.parentElement; ancestor; ancestor = ancestor.parentElement) {
            if (ancestor.tagName === "DETAILS") ancestor.open = true;
        }
        if (!wide.matches) setOpen(false);
        entry.heading.tabIndex = -1;
        entry.heading.focus({ preventScroll: true });
        schedule(true);
    };
    for (const entry of entries) {
        entry.link.addEventListener("click", (event) => navigate(event, entry), { signal });
    }
    toggle.addEventListener("click", () => setOpen(panel.hidden), { signal });
    backdrop.addEventListener("click", () => setOpen(false), { signal });
    window.addEventListener("keydown", (event) => {
        if (event.key === "Escape" && !panel.hidden) {
            setOpen(false);
            toggle.focus({ preventScroll: true });
        }
    }, { signal });
    // 用户再次交互即接管滚动，防止异步资源把阅读位置拉回。
    const releaseTarget = () => { target = null; };
    for (const name of ["wheel", "touchstart", "pointerdown", "keydown"]) {
        window.addEventListener(name, releaseTarget, { passive: true, signal });
    }
    window.addEventListener("mineru-preview-navigate", releaseTarget, { signal });
    window.addEventListener("scroll", () => schedule(), { passive: true, signal });
    window.addEventListener("resize", () => schedule(true), { signal });
    document.addEventListener("load", () => schedule(true), { capture: true, signal });
    document.addEventListener("toggle", () => schedule(true), { capture: true, signal });
    wide.addEventListener("change", () => setOpen(wide.matches), { signal });
    const observer = new ResizeObserver(() => schedule(true));
    observer.observe(article);
    document.fonts.ready.then(() => schedule(true));
    // pagehide 显式释放观察器、事件和待执行动画帧。
    window.addEventListener("pagehide", () => {
        disposed = true;
        observer.disconnect();
        controller.abort();
        cancelAnimationFrame(frame);
    }, { once: true });
    setOpen(wide.matches);
}
