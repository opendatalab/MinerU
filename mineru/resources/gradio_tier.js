// 胶囊只负责呈现与用户输入；文件锁档、可用档位和会话偏好仍由共享联动事件决定。
({ element, props, trigger, watch, tiers, defaultIndex }) => {
    const capsule = element.querySelector(".mineru-tier-capsule");
    const range = element.querySelector(".mineru-tier-range");
    const reset = element.querySelector(".mineru-tier-reset");
    const dots = element.querySelector(".mineru-tier-dots");
    let dragging = false;
    let committed = Number(props.value);
    let locked = props.locked;
    const maximum = tiers.length - 1;
    // 词典可能晚于组件挂载；初始化完成后由应用入口统一补齐本地化。
    const text = (key, values = {}) => window.__mineruI18n?.text(key, values) || (key === "tier" ? "解析等级" : "");
    // 所有提交值都夹紧为当前服务实际开放的整数档位。
    const clamp = value => Math.max(0, Math.min(maximum, Math.round(value)));

    // 仅在整行文字与重置按钮都能容纳时显示重置，判断控件宽度而非视口宽度。
    const resize = () => capsule.classList.toggle("is-wide", capsule.getBoundingClientRect().width >= 340);
    const resizeObserver = new ResizeObserver(resize);
    resizeObserver.observe(capsule);
    resize();

    for (let index = 0; index < tiers.length; index++) {
        const dot = document.createElement("span");
        dot.className = "mineru-tier-dot";
        dots.appendChild(dot);
    }
    range.max = String(Math.max(1, maximum));

    // 连续位置只绘制外观，文字按最近档位显示，不向 Gradio 写入小数。
    const render = (value, announce = false) => {
        const index = clamp(value);
        const tier = locked ? "flash" : tiers[index];
        const disabled = locked || maximum === 0;
        capsule.dataset.tier = tier;
        capsule.style.setProperty("--tier-position", String(maximum ? value / maximum : 0));
        element.querySelector(".mineru-tier-title").textContent = text("tier");
        element.querySelector(".mineru-tier-current").textContent = text(`tier_compact_${tier}`) || tier;
        element.querySelector(".mineru-tier-options").hidden = disabled;
        range.disabled = disabled;
        range.setAttribute("aria-label", text("tier"));
        range.setAttribute("aria-valuetext", text(`tier_${tier}`) || tier);
        const defaultLabel = text(`tier_compact_${tiers[defaultIndex]}`) || tiers[defaultIndex];
        const resetLabel = text("tier_reset", { tier: defaultLabel }) || `恢复默认等级：${defaultLabel}`;
        reset.disabled = disabled || index === defaultIndex;
        reset.setAttribute("aria-label", resetLabel);
        reset.setAttribute("title", locked ? text("tier_locked") : index === defaultIndex
            ? text("tier_at_default", { tier: defaultLabel }) : resetLabel);
        for (const [i, dot] of [...dots.children].entries()) {
            dot.dataset.passed = String(i <= index);
        }
        if (announce) element.querySelector(".mineru-tier-announcement").textContent = text(`tier_${tier}`) || tier;
    };

    // 文件锁档立即中止拖动；无关页码回执不打断当前跟手位置。
    const sync = () => {
        const changed = committed !== Number(props.value) || locked !== props.locked;
        committed = Number(props.value);
        locked = props.locked;
        if (changed) dragging = false;
        capsule.classList.toggle("is-dragging", dragging);
        if (!dragging) range.value = String(committed);
        render(Number(range.value));
    };

    // 松手、重置和键盘都走同一提交入口，只派发一次用户 input。
    const commit = value => {
        dragging = false;
        capsule.classList.remove("is-dragging");
        if (locked || maximum === 0) { sync(); return; }
        const index = clamp(value);
        range.value = String(index);
        render(index, true);
        if (index === committed) return;
        committed = index;
        props.value = index;
        trigger("input");
    };

    range.addEventListener("pointerdown", event => {
        if (range.disabled || event.button !== 0) return;
        dragging = true;
        capsule.classList.add("is-dragging");
        range.setPointerCapture(event.pointerId);
    });
    range.addEventListener("input", () => render(Number(range.value)));
    range.addEventListener("change", () => commit(Number(range.value)));
    range.addEventListener("pointerup", () => { if (dragging) commit(Number(range.value)); });
    // 触屏滚动或窗口失焦取消尚未提交的拖动，恢复最后一次有效选择。
    range.addEventListener("pointercancel", () => { dragging = false; sync(); });
    range.addEventListener("lostpointercapture", () => { if (dragging) { dragging = false; sync(); } });
    range.addEventListener("blur", () => { if (dragging) commit(Number(range.value)); });
    range.addEventListener("keydown", event => {
        const index = clamp(Number(range.value));
        const values = { ArrowRight: index + 1, ArrowUp: index + 1, ArrowLeft: index - 1, ArrowDown: index - 1,
            Home: 0, End: maximum, PageUp: index + 1, PageDown: index - 1 };
        if (!Object.hasOwn(values, event.key)) return;
        event.preventDefault();
        commit(values[event.key]);
    });
    // 默认档位由服务端共享选择器注入；锁档时不改写会话的未锁定偏好。
    reset.addEventListener("click", () => {
        if (!reset.disabled) commit(defaultIndex);
    });
    watch(["value", "selected_tier", "locked"], sync);
    window.__mineruTier = { localize: sync };
    sync();
}
