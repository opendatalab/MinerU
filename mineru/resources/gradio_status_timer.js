(i18n) => {
    let interval = null;
    let activeSpan = null;
    let activeStart = null;
    let startedAt = 0;
    let activeMode = null;
    let activeQueueKey = null;
    let queueStartedAt = 0;

    // 离开排队或解析阶段时停止本地动画，避免隐藏页面继续占用定时器。
    const stop = () => {
        if (interval !== null) {
            clearInterval(interval);
            interval = null;
        }
        activeSpan?.removeAttribute("data-mineru-local-timer");
        activeSpan?.removeAttribute("data-mineru-local-animation");
        activeSpan = null;
        activeStart = null;
        activeMode = null;
        activeQueueKey = null;
    };

    // 用单调时钟绘制解析耗时或排队圆点；后台标签页恢复后按实际时间追上。
    const render = () => {
        if (activeSpan?.isConnected === false) {
            sync();
            return;
        }
        if (!activeSpan) return;
        if (activeMode === "queue") {
            const dots = Math.floor(Math.max(0, performance.now() - queueStartedAt) / 1000) % 10 + 1;
            const value = `${i18n.text(activeQueueKey)}${".".repeat(dots)}`;
            if (activeSpan.textContent !== value) activeSpan.textContent = value;
            return;
        }
        const elapsed = Math.max(0, performance.now() - startedAt);
        const display = (Math.round(elapsed / 10) / 100).toFixed(2);
        const value = i18n.text("processing_elapsed", { elapsed: display });
        if (activeSpan.textContent !== value) activeSpan.textContent = value;
    };

    // Gradio 重绘同一阶段时保留本地起点；真实阶段切换时重设动画频率。
    const sync = () => {
        const status = document.querySelector(".mineru-status-panel .status-latest[data-mineru-processing-start]");
        const span = status?.querySelector('[data-mineru-i18n-key="status_message"]');
        const start = status?.getAttribute("data-mineru-processing-start");
        if (span && start !== null) {
            if (activeMode === "processing" && span === activeSpan && start === activeStart) return;
            if (activeMode !== "processing") stop();
            else activeSpan?.removeAttribute("data-mineru-local-timer");
            const serverElapsed = Number(status.getAttribute("data-mineru-processing-elapsed"));
            const serverElapsedMs = Number.isFinite(serverElapsed) ? Math.max(0, serverElapsed * 1000) : 0;
            const observedStart = performance.now() - serverElapsedMs;
            startedAt = activeMode === "processing" && start === activeStart
                ? Math.min(startedAt, observedStart) : observedStart;
            activeSpan = span;
            activeStart = start;
            activeMode = "processing";
            activeSpan.setAttribute("data-mineru-local-timer", "");
            if (interval === null) interval = setInterval(render, 10);
            render();
            return;
        }
        const queueStatus = document.querySelector(".mineru-status-panel .status-latest[data-mineru-queue-key]");
        const queueSpan = queueStatus?.querySelector('[data-mineru-i18n-key="status_message"]');
        const queueKey = queueStatus?.getAttribute("data-mineru-queue-key");
        if (queueSpan && (queueKey === "queued_locally" || queueKey === "queued_on_server")) {
            if (activeMode === "queue" && queueSpan === activeSpan && queueKey === activeQueueKey) {
                render();
                return;
            }
            if (activeMode !== "queue" || queueKey !== activeQueueKey) stop();
            else activeSpan?.removeAttribute("data-mineru-local-animation");
            if (activeMode !== "queue") queueStartedAt = performance.now();
            activeSpan = queueSpan;
            activeQueueKey = queueKey;
            activeMode = "queue";
            activeSpan.setAttribute("data-mineru-local-animation", "");
            if (interval === null) interval = setInterval(render, 1000);
            render();
            return;
        }
        stop();
    };

    // 点击转换时直接展示准备阶段，避开 Gradio 等待服务端回调后才更新 HTML 输出的延迟。
    const showPreparing = () => {
        const panel = document.querySelector(".mineru-status-panel .status-steps-panel");
        if (!panel) return;
        stop();
        panel.querySelectorAll(".status-step").forEach((step, index) => {
            step.classList.toggle("is-active", index === 0);
            step.classList.toggle("is-pending", index !== 0);
            step.classList.remove("is-done", "is-error");
        });
        // 使用原有双语节点，浏览器语言变化时仍能按项目词典更新。
        const setText = (element, key) => {
            if (!element) return;
            element.setAttribute("data-mineru-i18n-key", key);
            element.setAttribute("data-mineru-i18n-en", i18n.text(key, {}, "en"));
            element.setAttribute("data-mineru-i18n-zh", i18n.text(key, {}, "zh"));
            element.textContent = i18n.text(key);
        };
        setText(panel.querySelector(".status-panel-title [data-mineru-i18n-key]"), "status_latest");
        const latest = panel.querySelector(".status-latest");
        latest?.removeAttribute("data-mineru-processing-start");
        latest?.removeAttribute("data-mineru-processing-elapsed");
        latest?.removeAttribute("data-mineru-queue-key");
        setText(latest?.querySelector("[data-mineru-i18n-key]"), "preparing_request");
    };

    return { sync, showPreparing };
}
