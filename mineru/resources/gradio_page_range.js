(tiers, maxPages, file, position, metadata, previous, handleAValue, handleBValue) => {
    // 纯前端状态转换：经原生组件事件读写值，不逐次向 Python 发送拖动请求。
    // 内部元数据使用 JSON 文本，避免不同 Gradio 版本的 JSON 组件封装差异。
    metadata = JSON.parse(metadata || "{}");
    previous = JSON.parse(previous || "{}");
    // 统一构造原生组件的局部更新，未指定属性保持不变。
    const update = (props = {}) => ({ __type__: "update", ...props });
    // 页数读取错误只作为普通文本显示，禁止把异常内容解释为 HTML。
    const escapeHtml = (text) => String(text).replace(/[&<>"']/g, (char) => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    })[char]);
    const path = (typeof file === "string" ? file : file?.path) || "";
    const isPdf = path.toLowerCase().endsWith(".pdf");
    const needsRange = isPdf && tiers[position] !== "flash";
    const fileChanged = previous?.path !== path;
    const state = !fileChanged ? { ...previous } : {
        path, page_count: 0, handle_a: 1, handle_b: 1, start_handle: "a", error: "",
    };

    // 只缓存当前文件的元数据；晚到的旧文件响应不能覆盖已经读好的新文件页数。
    let initialized = false;
    if (isPdf && metadata?.path === path && !state.page_count && !state.error) {
        state.page_count = metadata.page_count || 0;
        state.error = metadata.error || "";
        state.handle_a = 1;
        state.handle_b = Math.min(state.page_count || 1, maxPages ?? Infinity);
        state.start_handle = "a";
        initialized = true;
    }

    const count = state.page_count || 0;
    if (count > 0 && !fileChanged && !initialized) {
        // 物理滑块只受文档边界约束；交叉时不交换组件值，保持鼠标和焦点所在的实体。
        const clamp = (value) => Math.min(count, Math.max(1, Math.round(value)));
        // 超限时把另一滑块推到拖动端附近，方向由实际位置而不是起止角色决定。
        const linkedPosition = (active, other) => maxPages !== null && Math.abs(active - other) + 1 > maxPages
            ? active + Math.sign(other - active) * (maxPages - 1)
            : other;
        if (handleAValue !== state.handle_a) {
            state.handle_a = clamp(handleAValue);
            state.handle_b = linkedPosition(state.handle_a, state.handle_b);
        } else if (handleBValue !== state.handle_b) {
            state.handle_b = clamp(handleBValue);
            state.handle_a = linkedPosition(state.handle_b, state.handle_a);
        }
    }

    // 只有严格越过才交换角色；重合时沿用上一次角色，避免边界附近的标签抖动。
    if (state.handle_a < state.handle_b) {
        state.start_handle = "a";
    } else if (state.handle_b < state.handle_a) {
        state.start_handle = "b";
    }
    const visible = needsRange && count > 0;
    const interactive = visible && count > 1;
    const start = Math.min(state.handle_a, state.handle_b);
    const end = Math.max(state.handle_a, state.handle_b);
    const selected = end - start + 1;
    const limitText = maxPages === null ? "不限页数" : `最多 ${maxPages} 页`;
    const summary = `<div class="mineru-page-values" data-range-visible="${visible}" data-start-handle="${state.start_handle}">`
        + `<span>起始页 <strong>${start}</strong></span>`
        + `<span class="mineru-page-selection">[${start}-${end}] · ${selected} 页</span>`
        + `<span>结束页 <strong>${end}</strong></span></div>`
        + `<div class="mineru-page-axis"><span>1</span><span>${limitText}</span><span>${count}</span></div>`;
    const notice = needsRange && !count ? (state.error || "正在读取 PDF 页数…") : "";
    const range = visible ? (start === end ? String(start) : `${start}-${end}`) : "";
    // 两个端点始终保留完整文档跨度，不能把轨道范围截短为页数上限。
    const slider = (value, label) => update({ minimum: 1, maximum: Math.max(1, count), value, interactive, label });
    return [
        slider(state.handle_a, state.start_handle === "a" ? "起始页" : "结束页"),
        slider(state.handle_b, state.start_handle === "b" ? "起始页" : "结束页"),
        summary, range, JSON.stringify(state),
        update({ interactive: Boolean(path) && (!needsRange || count > 0) }),
        update({ value: escapeHtml(notice), visible: Boolean(notice) }),
    ];
}
