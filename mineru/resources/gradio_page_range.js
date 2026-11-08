(tiers, flashOnlyExtensions, maxPages, file, position, metadata, previous, selection, tierSelection, sourceReady,
    startValue = null, endValue = null, preset = false) => {
    // 纯前端状态转换：经原生组件事件读写值，不逐次向 Python 发送拖动请求。
    // 内部元数据使用 JSON 文本，避免不同 Gradio 版本的 JSON 组件封装差异。
    const { text, message } = window.__mineruI18n;
    metadata = JSON.parse(metadata || "{}");
    previous = JSON.parse(previous || "{}");
    tierSelection = JSON.parse(tierSelection);
    selection = JSON.parse(selection || "{}");
    const { clamp, move } = (__MINERU_PAGE_RANGE_MODEL__)();
    // 统一构造原生组件的局部更新，未指定属性保持不变。
    const update = (props = {}) => ({ __type__: "update", ...props });
    // 页数读取错误只作为普通文本显示，禁止把异常内容解释为 HTML。
    const escapeHtml = (text) => String(text).replace(/[&<>"']/g, (char) => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    })[char]);
    const path = (typeof file === "string" ? file : file?.path) || "";
    const flashOnly = flashOnlyExtensions.some((extension) => path.toLowerCase().endsWith(`.${extension}`));
    // 只记住未锁定时的用户选择；文件切换和晚到元数据不能把临时 Flash 写回偏好。
    if (!tierSelection.locked && tiers[position] !== undefined) tierSelection.tier = tiers[position];
    const effectivePosition = Math.max(0, tiers.indexOf(tierSelection.tier));
    const selectedTier = flashOnly ? "flash" : tiers[effectivePosition];
    tierSelection.locked = flashOnly;
    const isPdf = path.toLowerCase().endsWith(".pdf");
    const needsRange = isPdf;
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
        // 数字输入和滑块复用同一端点更新；交叉后仍以当前语义角色定位物理滑块。
        const moveHandle = (handle, value) => {
            [state.handle_a, state.handle_b] = move(
                state.handle_a, state.handle_b, handle, Math.round(value), count, maxPages
            );
        };
        if (preset) {
            state.handle_a = 1;
            state.handle_b = Math.min(count, maxPages ?? count);
            state.start_handle = "a";
        } else if (selection.path === path && selection.page_count === count
            && Number.isFinite(selection.handle_a) && Number.isFinite(selection.handle_b)
            && (selection.handle_a !== state.handle_a || selection.handle_b !== state.handle_b)) {
            // 整段平移必须接收同一提交中的两端；旧文件手势和越界输入不能污染新状态。
            state.handle_a = clamp(Math.round(selection.handle_a), count);
            state.handle_b = clamp(Math.round(selection.handle_b), count);
            moveHandle("a", state.handle_a);
        } else if (Number.isFinite(startValue) && startValue !== Math.min(state.handle_a, state.handle_b)) {
            moveHandle(state.start_handle, startValue);
        } else if (Number.isFinite(endValue) && endValue !== Math.max(state.handle_a, state.handle_b)) {
            moveHandle(state.start_handle === "a" ? "b" : "a", endValue);
        }
    }

    // 只有严格越过才交换角色；重合时沿用上一次角色，避免边界附近的标签抖动。
    if (state.handle_a < state.handle_b) {
        state.start_handle = "a";
    } else if (state.handle_b < state.handle_a) {
        state.start_handle = "b";
    }
    // PDF 页数读取期间保留选页控件的占位，避免切换文件时整列高度跳动。
    const visible = needsRange;
    const interactive = visible && count > 1;
    const start = Math.min(state.handle_a, state.handle_b);
    const end = Math.max(state.handle_a, state.handle_b);
    const selected = end - start + 1;
    const limitText = maxPages === null ? text("page_unlimited") : text("page_limit", { count: maxPages });
    const selectionText = `<span class="mineru-page-selected-count" data-mineru-page-live>${count ? selected : "–"}</span> / `
        + text("page_count", { count: count || "–" });
    const intervalText = count ? (start === end ? text("page_single", { page: start }) : text("page_interval", { start, end }))
        : text(state.error ? "page_read_failed_short" : "reading_pages");
    const summary = `<div class="mineru-page-values" data-range-visible="${visible}" data-start-handle="${state.start_handle}">`
        + `<span class="mineru-page-heading"><span class="mineru-page-title">${text("page_range")}</span>`
        + `<span class="mineru-page-limit" title="${escapeHtml(limitText)}">${limitText}</span></span>`
        + `<span class="mineru-page-selection">${selectionText}</span>`
        + `</div>`
        + `<div class="mineru-page-axis"><span>1</span><span class="mineru-page-interval" data-mineru-page-live>${intervalText}</span>`
        + `<span>${count || "–"}</span></div>`;
    const notice = needsRange && !count ? message(state.error) : "";
    const range = count ? (start === end ? String(start) : `${start}-${end}`) : "";
    // 数字框确认输入时夹紧到真实页数；单页保持可见但不可编辑。
    const number = (value, label) => update({ minimum: 1, maximum: Math.max(1, count), value, interactive, label });
    return [
        update({ value: JSON.stringify(state) }),
        summary, range, JSON.stringify(state),
        update({ interactive: Boolean(path) && sourceReady === path && (!needsRange || count > 0) }),
        update({ value: escapeHtml(notice), visible: Boolean(notice) }),
        // 同时更新公开 API 数值与胶囊状态；程序赋值不触发胶囊的 input 事件。
        update({ value: effectivePosition }),
        update({ value: effectivePosition, props: { selected_tier: selectedTier, locked: flashOnly } }),
        JSON.stringify(tierSelection),
        number(start, text("start_page_short")),
        number(end, text("end_page_short")),
        update({ value: maxPages !== null && count > maxPages ? text("page_first", { count: maxPages }) : text("page_all"),
            interactive }),
    ];
}
