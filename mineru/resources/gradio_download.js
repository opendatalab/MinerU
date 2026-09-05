// 按结果标识协调下载请求、按钮状态和浏览器下载，丢弃旧文档的回执。
(action, formats, format, label, ...args) => {
    const state = window.__mineruGradioDownloads ??= { runId: "", sequence: 0, pending: new Map() };
    // 用标准组件更新对象兼容 Gradio 5/6；空更新不会覆盖新文档的按钮状态。
    const skip = () => ({ __type__: "update" });
    const button = (value, interactive) => ({ __type__: "update", value, interactive });
    // 下载失败文案仅作为文本展示，不能把服务端错误插入为活动 HTML。
    const escapeHtml = (value) => String(value).replace(/[&<>"']/g, (char) => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    })[char]);

    if (action === "reset") {
        state.runId = "";
        state.pending.clear();
        return [
            "", ...formats.map(() => null), ...formats.map(() => ""), ...formats.map(() => ""),
            ...formats.map(([, name]) => button(name, false)), "",
        ];
    }
    if (action === "activate") {
        if (state.runId !== args[0]) state.pending.clear();
        state.runId = args[0] || "";
        return [];
    }
    if (action === "begin") {
        const [runId] = args;
        if (!runId || state.runId !== runId || state.pending.has(format)) return [skip(), skip(), skip()];
        const token = JSON.stringify({ run_id: runId, sequence: ++state.sequence });
        state.pending.set(format, token);
        return [token, button(`${label} · 准备中…`, false), ""];
    }
    if (action === "busy") {
        const [token, runId] = args;
        // 即使开始回执晚于换文件或清除，也只能修改当前仍在等待的同一个请求。
        if (!token || state.pending.get(format) !== token || !runId || state.runId !== runId) return [skip(), skip()];
        return [button(`${label} · 准备中…`, false), ""];
    }
    if (action === "complete") {
        const [file, receiptText, runId] = args;
        const receipt = JSON.parse(receiptText || "{}");
        const token = state.pending.get(format);
        // 已清除、已换文档或已消费的回执不得再次下载，也不得恢复旧按钮。
        if (!token || token !== receipt.request || !runId || state.runId !== runId) return [skip(), skip()];
        const request = JSON.parse(token);
        if (request.run_id !== runId) return [skip(), skip()];
        state.pending.delete(format);
        if (receipt.error || !file?.url) {
            const message = receipt.error || "未获得下载文件，请重试。";
            return [button(label, true), `<div role="alert">${escapeHtml(label)} 下载失败：${escapeHtml(message)}</div>`];
        }
        const anchor = document.createElement("a");
        anchor.href = file.url;
        anchor.download = file.orig_name || "download";
        document.body.appendChild(anchor);
        anchor.click();
        anchor.remove();
        return [button(label, true), ""];
    }
    return [];
}
