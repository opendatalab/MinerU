// 指针拖动只更新本地外观，松手后通过一个组件值原子提交两端。
({ element, props, trigger, watch, maxPages }) => {
    const { move, translate } = (__MINERU_PAGE_RANGE_MODEL__)();
    const rail = element.querySelector(".mineru-page-sliders");
    const inputs = ["a", "b"].map(handle => element.querySelector(`.mineru-page-handle-${handle} input`));
    let committed = JSON.parse(props.value || "{}");
    let drag = null;
    let visual = [1, 1];
    // 词典由应用初始化；组件先挂载时保留已有无障碍标签。
    const text = (key, values) => window.__mineruI18n?.text(key, values) || "";
    // 只更新变化的叶节点，不派发 input，不让 Gradio 每帧重建摘要。
    const setText = (selector, value) => {
        const node = element.closest(".mineru-kit-page-range")?.querySelector(selector);
        if (node && value && node.textContent !== value) node.textContent = value;
    };
    // 平移只取整一次偏移量，避免两个浮点端点分别舍入后偶发多选一页。
    const snap = (a, b) => {
        const count = Math.max(1, committed.page_count || 1);
        if (drag?.handle === "pan") return translate(...drag.initial, Math.round(a - drag.initial[0]), count);
        const handle = drag?.handle === "b" ? "b" : "a";
        return move(Math.round(a), Math.round(b), handle, Math.round(handle === "a" ? a : b), count, maxPages);
    };
    // 连续位置负责跟手；数字、无障碍值和页数始终采用整数。
    const render = (a, b) => {
        visual = [a, b];
        const count = committed.page_count || 0;
        const span = Math.max(1, count - 1);
        const values = snap(a, b);
        const start = Math.min(...values);
        const end = Math.max(...values);
        const startHandle = a < b ? "a" : b < a ? "b" : committed.start_handle || "a";
        rail.style.setProperty("--range-start", `${(Math.min(a, b) - 1) / span * 100}%`);
        rail.style.setProperty("--range-end", `${(Math.max(a, b) - 1) / span * 100}%`);
        rail.dataset.active = drag?.handle || "";
        rail.dataset.disabled = String(count < 2);
        for (const [index, input] of inputs.entries()) {
            const handle = index === 0 ? "a" : "b";
            input.parentElement.style.setProperty("--page-position", String((visual[index] - 1) / span));
            input.max = String(Math.max(2, count));
            input.value = String(values[index]);
            input.disabled = count < 2;
            const label = text(startHandle === handle ? "start_page" : "end_page");
            if (label && input.getAttribute("aria-label") !== label) input.setAttribute("aria-label", label);
        }
        const root = element.closest(".mineru-kit-page-range");
        for (const [selector, value] of [[".mineru-page-start input", start], [".mineru-page-end input", end]]) {
            const input = root?.querySelector(selector);
            if (input && input !== document.activeElement) input.value = String(value);
        }
        if (count) {
            setText(".mineru-page-selected-count", String(end - start + 1));
            setText(".mineru-page-interval", start === end ? text("page_single", { page: start })
                : text("page_interval", { start, end }));
        }
    };
    // 取消捕获先清空拖动态，防止 lostpointercapture 二次提交。
    const release = () => {
        const previous = drag;
        drag = null;
        if (previous && rail.hasPointerCapture(previous.id)) rail.releasePointerCapture(previous.id);
    };
    // 取消、窗口失焦和文件切换恢复最后的正式范围，不提交半途选择。
    const cancel = () => {
        release();
        render(committed.handle_a || 1, committed.handle_b || 1);
    };
    // 一次 input 携带完整范围，整段平移不能拆成两个可能相互覆盖的事件。
    const commit = () => {
        const [a, b] = snap(...visual);
        release();
        if (a === committed.handle_a && b === committed.handle_b) { render(a, b); return; }
        committed = { ...committed, handle_a: a, handle_b: b,
            start_handle: a < b ? "a" : b < a ? "b" : committed.start_handle };
        render(a, b);
        props.value = JSON.stringify(committed);
        trigger("input");
    };
    // 外部数字输入、快捷选择及文件切换会同步状态；无关回执不打断手势。
    const sync = () => {
        const next = JSON.parse(props.value || "{}");
        const changed = ["path", "page_count", "handle_a", "handle_b", "start_handle"]
            .some(key => next[key] !== committed[key]);
        committed = next;
        if (changed) release();
        if (!drag) render(committed.handle_a || 1, committed.handle_b || 1);
    };
    // 中段只在端点命中区之外启用；接近或重合时优先保留端点的可抓取性。
    const hit = (x, bounds) => {
        const span = Math.max(1, committed.page_count - 1);
        const centers = visual.map(value => bounds.left + 14 + (value - 1) / span * (bounds.width - 28));
        const distances = centers.map(center => Math.abs(x - center));
        if (Math.abs(centers[0] - centers[1]) < 1 && Math.min(...distances) <= 22) return "pending";
        if (Math.min(...distances) <= 22) return distances[0] <= distances[1] ? "a" : "b";
        if (x > Math.min(...centers) && x < Math.max(...centers)) return "pan";
        return distances[0] <= distances[1] ? "a" : "b";
    };
    // 同一指针保持物理端点身份；重合时由第一次有效移动方向决定起止端。
    const preview = event => {
        if (!drag || event.pointerId !== drag.id) return;
        const delta = (event.clientX - drag.x) / drag.width * (committed.page_count - 1);
        if (drag.handle === "pending") {
            if (Math.abs(event.clientX - drag.x) < 2) return;
            const start = committed.start_handle || "a";
            drag.handle = delta < 0 ? start : start === "a" ? "b" : "a";
            inputs[drag.handle === "a" ? 0 : 1].focus({ preventScroll: true });
        }
        if (drag.handle === "pan") render(...translate(...drag.initial, delta, committed.page_count));
        else {
            const index = drag.handle === "a" ? 0 : 1;
            render(...move(...visual, drag.handle, drag.initial[index] + delta, committed.page_count, maxPages));
        }
    };
    rail.addEventListener("pointerdown", event => {
        if (event.button !== 0 || event.isPrimary === false || drag || (committed.page_count || 0) < 2) return;
        event.preventDefault();
        rail.dataset.keyboard = "false";
        const bounds = rail.getBoundingClientRect();
        const handle = hit(event.clientX, bounds);
        const width = Math.max(1, bounds.width - 28);
        drag = { id: event.pointerId, x: event.clientX, width, handle, initial: [...visual] };
        rail.setPointerCapture(event.pointerId);
        if (handle === "a" || handle === "b") {
            const index = handle === "a" ? 0 : 1;
            const center = bounds.left + 14 + (visual[index] - 1) / (committed.page_count - 1) * width;
            // 命中透明热区时不跳动；点击轨道空白处才直接定位最近端点。
            if (Math.abs(event.clientX - center) > 22) {
                render(...move(...visual, handle, 1 + (event.clientX - bounds.left - 14) / width
                    * (committed.page_count - 1), committed.page_count, maxPages));
                drag.initial = [...visual];
            }
            inputs[index].focus({ preventScroll: true });
        } else {
            inputs[committed.start_handle === "b" ? 1 : 0].focus({ preventScroll: true });
        }
        render(...visual);
    });
    rail.addEventListener("pointermove", event => {
        if (drag) { preview(event); return; }
        if (committed.page_count > 1) rail.dataset.hover = hit(event.clientX, rail.getBoundingClientRect());
    });
    rail.addEventListener("pointerleave", () => { delete rail.dataset.hover; });
    rail.addEventListener("pointerup", event => {
        if (!drag || event.pointerId !== drag.id) return;
        preview(event);
        commit();
    });
    rail.addEventListener("pointercancel", event => { if (drag?.id === event.pointerId) cancel(); });
    rail.addEventListener("lostpointercapture", event => { if (drag?.id === event.pointerId) cancel(); });
    // 原生 range 保留焦点和读屏语义；方向键逐页、Home/End 到文档边界。
    for (const [index, input] of inputs.entries()) {
        // 指针聚焦不保留键盘焦点环；Tab 进入时仍给出清晰的焦点提示。
        input.addEventListener("focus", () => { rail.dataset.keyboard = String(!drag); });
        input.addEventListener("keydown", event => {
            if (event.key === "Escape") { if (drag) { event.preventDefault(); cancel(); } return; }
            if (input.disabled || drag) return;
            const value = Math.round(visual[index]);
            const keys = { ArrowLeft: value - 1, ArrowDown: value - 1, ArrowRight: value + 1,
                ArrowUp: value + 1, Home: 1, End: committed.page_count, PageDown: value - 10, PageUp: value + 10 };
            if (!Object.hasOwn(keys, event.key)) return;
            event.preventDefault();
            rail.dataset.keyboard = "true";
            render(...move(...visual, index === 0 ? "a" : "b", keys[event.key], committed.page_count, maxPages));
            commit();
        });
        // 辅助技术直接修改原生值时也经过相同的限制和原子提交。
        input.addEventListener("input", () => {
            if (drag || input.disabled) return;
            render(...move(...visual, index === 0 ? "a" : "b", Number(input.value), committed.page_count, maxPages));
            commit();
        });
    }
    window.addEventListener("blur", cancel);
    watch(["value"], sync);
    window.__mineruPageRange = { localize: sync };
    sync();
}
