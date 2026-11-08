// 胶囊只负责呈现与用户输入；文件锁档、可用档位和会话偏好仍由共享联动事件决定。
({ element, props, trigger, watch, tiers }) => {
    const capsule = element.querySelector(".mineru-tier-capsule");
    const range = element.querySelector(".mineru-tier-range");
    const choices = element.querySelector(".mineru-tier-choices");
    const dots = element.querySelector(".mineru-tier-dots");
    let dragging = false;
    let committed = Number(props.value);
    let locked = props.locked;
    const maximum = tiers.length - 1;
    // 词典可能晚于组件挂载；初始化完成后由应用入口统一补齐本地化。
    const text = key => window.__mineruI18n?.text(key) || (key === "tier" ? "解析等级" : "");
    // 所有提交值都夹紧为当前服务实际开放的整数档位。
    const clamp = value => Math.max(0, Math.min(maximum, Math.round(value)));

    for (const [index, tier] of tiers.entries()) {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "mineru-tier-choice";
        button.dataset.index = String(index);
        button.textContent = tier;
        choices.appendChild(button);
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
        for (const [i, button] of [...choices.children].entries()) {
            button.disabled = disabled;
            button.setAttribute("aria-pressed", String(!locked && i === index));
            button.setAttribute("aria-label", text(`tier_${tiers[i]}`) || tiers[i]);
            dots.children[i].dataset.passed = String(i <= index);
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

    // 松手、点击文字和键盘都走同一提交入口，只派发一次用户 input。
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
    choices.addEventListener("click", event => {
        const button = event.target.closest(".mineru-tier-choice");
        if (button && !button.disabled) commit(Number(button.dataset.index));
    });
    watch(["value", "selected_tier", "locked"], sync);
    window.__mineruTier = { localize: sync };
    sync();
}
