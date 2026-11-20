// 在已解析的 demo1.pdf 或 demo2.pdf 上采样真实 iframe 的逐帧滚动，检查距离、方向、落点和用户接管。
async page => {
    const pdfHost = page.locator("iframe.mineru-pdf-frame");
    const htmlHost = page.locator("iframe.mineru-rendered-html-frame");
    const pdf = await (await pdfHost.elementHandle()).contentFrame();
    const html = await (await htmlHost.elementHandle()).contentFrame();
    const sync = pdf.locator("#page-sync-toggle");
    const input = pdf.locator("#page");
    // 采样两侧真实位置，大跨度定位若先瞬移到页首，也会计入首步位移。
    const start = (frame, pdfSide) => frame.evaluate(pdfSide => {
        const root = pdfSide ? document.querySelector("#viewer-container") : document.scrollingElement;
        const y = root.scrollTop;
        const probe = window.__previewMotionProbe = { points: [y], frame: 0 };
        const record = () => {
            if (root.scrollTop !== probe.points.at(-1)) probe.points.push(root.scrollTop);
            probe.frame = requestAnimationFrame(record);
        };
        probe.frame = requestAnimationFrame(record);
    }, pdfSide);
    // 已知产品动画上限为 360ms，再等一段时间以检测过期动画是否把视口拉回。
    const finish = async (frame, reduced = false) => {
        await frame.waitForTimeout(550);
        const stats = await frame.evaluate(() => {
            const { frame, points } = window.__previewMotionProbe;
            cancelAnimationFrame(frame);
            delete window.__previewMotionProbe;
            const delta = points.at(-1) - points[0];
            const maxStep = Math.max(0, ...points.slice(1).map((p, i) => Math.abs(p - points[i])));
            return { steps: points.length - 1, delta: Math.round(delta), maxStep: Math.round(maxStep) };
        });
        if (Math.abs(stats.delta) < 100 || (reduced ? stats.steps !== 1 : stats.steps < 4 || stats.maxStep >= Math.abs(stats.delta) * 0.65)) {
            throw new Error("预览滚动轨迹异常 " + JSON.stringify(stats));
        }
        return stats;
    };
    // 通过页码框显式跳页，两边应同时滚动，各自只执行最后一个目标。
    const go = async number => {
        await input.fill(String(number));
        await input.press("Enter");
    };
    const total = Number(await input.getAttribute("max"));
    const far = total - 1;
    if (await sync.getAttribute("aria-pressed") === "false") await sync.click();
    await go(1);
    await page.waitForTimeout(500);
    await Promise.all([start(pdf, true), start(html, false)]);
    await go(far);
    const [toolbarPdf, toolbarHtml] = await Promise.all([finish(pdf), finish(html)]);
    // 关闭自动翻页，分开两边的位置，分别验证两个方向的显式点击溯源。
    await sync.click();
    await go(1);
    await page.waitForTimeout(500);
    const box = pdf.locator('.page[data-page-number="1"] button[data-trace-id]').first();
    await start(html, false);
    await box.click();
    const traceHtml = await finish(html);
    const target = html.locator('[data-page-idx="' + (far - 1) + '"][data-block-type="text"][data-docvortex-trace-id]').first();
    await start(pdf, true);
    await target.press("Enter");
    const tracePdf = await finish(pdf);
    const id = await target.getAttribute("data-docvortex-trace-id");
    const visible = await pdf.evaluate(id => {
        const box = [...document.querySelectorAll("button.selected")].find(n => n.traceIds.includes(id));
        if (!box) return false;
        const rect = box.getBoundingClientRect(), host = document.querySelector("#viewer-container").getBoundingClientRect();
        return rect.bottom > host.top && rect.top < host.bottom && rect.right > host.left && rect.left < host.right;
    }, id);
    if (!visible) throw new Error("目标块没有出现在 PDF 视口");
    // 在动画中途接管 PDF，等待超过动画时长后仍停在用户的位置。
    await pdfHost.hover({ position: { x: 120, y: 230 } });
    const first = await pdf.locator("#viewer-container").evaluate(n => n.scrollTop);
    await go(1);
    await pdf.waitForFunction(y => document.querySelector("#viewer-container").scrollTop < y - 5, first);
    // 页码框获得焦点可能使外层页面移动，按此时真实可见的 PDF 视口投递滚轮。
    const bounds = await pdf.locator("#viewer-container").boundingBox();
    await page.mouse.move(bounds.x + 100, Math.max(30, bounds.y + 120));
    await pdf.evaluate(() => {
        const container = document.querySelector("#viewer-container");
        container.addEventListener("wheel", () => { window.__wheelTop = container.scrollTop; }, { once: true });
    });
    await page.mouse.wheel(0, 46);
    await pdf.waitForFunction(() => Number.isFinite(window.__wheelTop));
    await page.waitForTimeout(120);
    const stopped = await pdf.locator("#viewer-container").evaluate(n => n.scrollTop);
    await page.waitForTimeout(430);
    if (Math.abs(await pdf.locator("#viewer-container").evaluate(n => n.scrollTop) - stopped) > 2 || stopped < 100) {
        throw new Error("手动滚动未能取消旧目标 " + JSON.stringify({ first, stopped, wheel: await pdf.evaluate(() => window.__wheelTop) }));
    }
    await page.emulateMedia({ reducedMotion: "reduce" });
    try {
        await start(pdf, true);
        await go(1);
        const reduced = await finish(pdf, true);
        return { toolbarPdf, toolbarHtml, tracePdf, traceHtml, stopped, reduced };
    } finally {
        await page.emulateMedia({ reducedMotion: "no-preference" });
        await sync.click();
    }
}
