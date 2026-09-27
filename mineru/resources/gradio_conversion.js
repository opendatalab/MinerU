// 普通响应携带完整快照；任务身份与序号检查先于任何可见组件更新。
(action, ...args) => {
    const state = window.__mineruConversion ??= { revision: 0, runId: "", sequence: 0, terminal: false, applied: false };
    // 使用 Gradio 的空更新保留当前组件，且本脚本的响应不经过生成器差分。
    const skip = () => ({ __type__: "update" });
    const timer = (active) => ({ __type__: "update", active });
    // 容忍已清除的空回执，格式异常也不能破坏当前任务。
    const parse = (value) => {
        try { return JSON.parse(value || "null"); } catch { return null; }
    };
    // 每个任务最多保留一个预览加载监听器，换文件或重新提交立即回收。
    const stopPreviewLog = () => {
        if (state.previewLoad) document.removeEventListener("load", state.previewLoad, true);
        state.previewLoad = null;
    };

    if (action === "begin") {
        stopPreviewLog();
        state.runId = crypto.randomUUID().replaceAll("-", "");
        state.revision += 1;
        state.sequence = 0;
        state.terminal = false;
        state.applied = false;
        window.__mineruStatusPanel?.showPreparing();
        return [JSON.stringify({ run_id: state.runId, revision: state.revision }), timer(true)];
    }
    if (action === "cancel") {
        stopPreviewLog();
        const previous = state.runId ? JSON.stringify({ run_id: state.runId, revision: state.revision }) : "";
        state.runId = "";
        state.terminal = true;
        state.applied = false;
        return ["", previous, timer(false)];
    }
    if (action === "status") {
        const snapshot = parse(args[0]);
        if (!snapshot || !state.runId || snapshot.run_id !== state.runId || state.terminal
            || !Number.isInteger(snapshot.sequence) || snapshot.sequence <= state.sequence) return [skip(), skip()];
        state.sequence = snapshot.sequence;
        state.terminal = Boolean(snapshot.terminal);
        return [snapshot.html, timer(!state.terminal)];
    }
    if (action === "result") {
        const receipt = parse(args[0]);
        // 返回 15 个可见/文件输出和 Timer；服务端的 artifact State 不经过浏览器写入。
        if (!receipt || !state.runId || receipt.run_id !== state.runId || state.applied
            || !Array.isArray(receipt.outputs) || receipt.outputs.length !== 15
            || !Number.isInteger(receipt.sequence) || receipt.sequence < state.sequence) {
            return Array.from({ length: 16 }, skip);
        }
        state.sequence = receipt.sequence;
        state.terminal = true;
        state.applied = true;
        console.info("[MinerU WebUI] result received", JSON.stringify({
            run_id: receipt.run_id, ready_at: receipt.ready_at, received_at: Date.now() / 1000,
        }));
        stopPreviewLog();
        // 在框架写入 iframe 之前捕获 load，避免快速 srcdoc 在下一动画帧前已经加载。
        if (receipt.outputs[1]) {
            state.previewLoad = (event) => {
                if (!event.target?.matches?.("iframe.mineru-rendered-html-frame")) return;
                stopPreviewLog();
                if (state.runId === receipt.run_id) console.info("[MinerU WebUI] preview loaded", JSON.stringify({
                    run_id: receipt.run_id, loaded_at: Date.now() / 1000,
                }));
            };
            document.addEventListener("load", state.previewLoad, true);
        }
        return [...receipt.outputs, timer(false)];
    }
    return [];
}
