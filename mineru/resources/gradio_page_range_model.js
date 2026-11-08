// 拖动预览与正式提交共用边界算法，物理端点交叉时不交换身份。
() => {
    // 连续坐标只限制文档边界，取整由提交入口统一完成。
    const clamp = (value, count) => Math.max(1, Math.min(count, value));
    // 超出页数上限时同步推移另一端，保留当前拖动端的跟手位置。
    const move = (a, b, handle, value, count, limit) => {
        const active = clamp(value, count);
        let other = handle === "a" ? b : a;
        const distance = (limit ?? count) - 1;
        if (Math.abs(active - other) > distance) other = active + Math.sign(other - active) * distance;
        return handle === "a" ? [active, other] : [other, active];
    };
    // 整段移动共用一个偏移量，到达文档边界时保持选区长度。
    const translate = (a, b, delta, count) => {
        const offset = Math.max(1 - Math.min(a, b), Math.min(count - Math.max(a, b), delta));
        return [a + offset, b + offset];
    };
    return { clamp, move, translate };
}
