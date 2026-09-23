(i18n) => {
    let interval = null;
    let activeSpan = null;
    let activeStart = null;
    let startedAt = 0;

    // 结束解析或重置界面时立即停止计时，避免隐藏页面继续占用定时器。
    const stop = () => {
        if (interval !== null) {
            clearInterval(interval);
            interval = null;
        }
        activeSpan?.removeAttribute("data-mineru-local-timer");
        activeSpan = null;
        activeStart = null;
    };

    // 使用单调时钟计算实际耗时；后台标签页暂停定时器后恢复也不会累计误差。
    const render = () => {
        if (activeSpan?.isConnected === false) {
            sync();
            return;
        }
        if (!activeSpan) return;
        const elapsed = Math.max(0, performance.now() - startedAt);
        const display = (Math.round(elapsed / 10) / 100).toFixed(2);
        const value = i18n.text("processing_elapsed", { elapsed: display });
        if (activeSpan.textContent !== value) activeSpan.textContent = value;
    };

    // Gradio 重绘同一解析状态时续用原计时起点；新任务才重新开始。
    const sync = () => {
        const status = document.querySelector(".mineru-status-panel .status-latest[data-mineru-processing-start]");
        const span = status?.querySelector('[data-mineru-i18n-key="status_message"]');
        const start = status?.getAttribute("data-mineru-processing-start");
        if (!span || start === null) {
            stop();
            return;
        }
        if (span === activeSpan && start === activeStart) return;
        activeSpan?.removeAttribute("data-mineru-local-timer");
        const serverElapsed = Number(status.getAttribute("data-mineru-processing-elapsed"));
        const serverElapsedMs = Number.isFinite(serverElapsed) ? Math.max(0, serverElapsed * 1000) : 0;
        const observedStart = performance.now() - serverElapsedMs;
        startedAt = start === activeStart ? Math.min(startedAt, observedStart) : observedStart;
        activeSpan = span;
        activeStart = start;
        activeSpan.setAttribute("data-mineru-local-timer", "");
        if (interval === null) interval = setInterval(render, 10);
        render();
    };

    return { sync };
}
