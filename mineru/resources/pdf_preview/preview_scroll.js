// 两个隔离预览共用同一段滚动曲线，所有坐标均属于传入的滚动容器。
function createMineruPreviewScroll(element) {
    let frame = 0;
    let destination = null;
    // 手动接管、换目标和卸载时取消剩余帧，保留已经滚到的位置。
    const cancel = () => {
        cancelAnimationFrame(frame);
        frame = 0;
        destination = null;
    };
    // 从当前位置缓入缓出，长距离跳转也在有限时间内完成。
    const move = (top, left = element.scrollLeft, onFinish) => {
        top = Math.max(0, Math.min(top, element.scrollHeight - element.clientHeight));
        left = Math.max(0, Math.min(left, element.scrollWidth - element.clientWidth));
        if (destination && Math.abs(destination.top - top) < 1 && Math.abs(destination.left - left) < 1) return;
        cancel();
        const x = element.scrollLeft, y = element.scrollTop;
        const distance = Math.hypot(left - x, top - y);
        if (distance < 1 || window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
            element.scrollTo({ top, left, behavior: "instant" });
            onFinish?.();
            return;
        }
        destination = { top, left };
        const duration = Math.min(360, 190 + 80 * Math.sqrt(distance / Math.max(1, element.clientHeight)));
        const start = performance.now();
        // 每帧使用初始位置和单调时间求值，丢帧不会拖长动画。
        const step = time => {
            const t = Math.max(0, Math.min(1, (time - start) / duration));
            const progress = t * t * (3 - 2 * t);
            element.scrollTo({ top: y + (top - y) * progress, left: x + (left - x) * progress, behavior: "instant" });
            if (t < 1) {
                frame = requestAnimationFrame(step);
            } else {
                frame = 0;
                destination = null;
                onFinish?.();
            }
        };
        frame = requestAnimationFrame(step);
    };
    return { move, cancel };
}
