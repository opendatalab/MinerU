// 用 playwright-cli run-code --filename 执行；验证真实 WebUI 的跨区域松手和捕获丢失回归。
async (page) => {
    const viewport = page.viewportSize();
    await page.setViewportSize({ width: 1600, height: 1000 });
    const capsule = page.locator(".mineru-tier-capsule");
    const rail = capsule.locator(".mineru-tier-rail");
    const range = capsule.locator(".mineru-tier-range");
    const thumb = capsule.locator(".mineru-tier-thumb");
    // 只等待圆点和填充条的吸附过渡，跳过无限星点；最终状态不能被迟到的取消或捕获丢失覆盖。
    const settled = async expected => {
        await capsule.evaluate(async node => {
            const transitions = node.getAnimations({ subtree: true }).filter(animation =>
                animation instanceof CSSTransition
                && animation.effect.target.matches(".mineru-tier-thumb, .mineru-tier-fill"));
            await Promise.all(transitions.map(animation => animation.finished.catch(() => {})));
            await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
        });
        if (Number(await range.inputValue()) !== expected) throw new Error(`Tier bounced back; expected ${expected}`);
        if (await page.locator(".mineru-tier-drag-surface").count()) throw new Error("Drag surface leaked after release");
    };
    // 从真实圆点开始拖动，并记录真实 pointerId 以测试浏览器的捕获释放。
    const begin = async initial => {
        await range.press(initial === 0 ? "Home" : "End");
        await settled(initial);
        await rail.evaluate(node => {
            node.addEventListener("pointerdown", event => { node.dataset.testPointerId = String(event.pointerId); }, { once: true });
        });
        const knob = await thumb.boundingBox();
        await page.mouse.move(knob.x + knob.width / 2, knob.y + knob.height / 2);
        await page.mouse.down();
        return await rail.boundingBox();
    };
    const results = [];
    try {
        // 旧实现中真实释放捕获后再在空白处松手，会从 Basic 回弹到 Advanced。
        const bounds = await begin(3);
        const basicX = bounds.x + 15 + (bounds.width - 30) / 3;
        await page.mouse.move(bounds.x + 15 + (bounds.width - 30) * 0.8, bounds.y + 25);
        await rail.evaluate(node => node.releasePointerCapture(Number(node.dataset.testPointerId)));
        await page.mouse.move(basicX, bounds.y + 130);
        await page.mouse.up();
        await settled(1);
        results.push("capture-loss-blank-release");

        // 输入或窗口失焦只是中断交互，已拖到的位置应保留，迟到松手不能再次改变它。
        for (const target of ["input", "window"]) {
            const bounds = await begin(3);
            await page.mouse.move(bounds.x + 15 + (bounds.width - 30) / 3, bounds.y + 25);
            if (target === "input") await range.evaluate(node => node.blur());
            else await page.evaluate(() => window.dispatchEvent(new Event("blur")));
            await page.mouse.up();
            await settled(1);
            results.push(`${target}-blur-keeps-position`);
        }

        // 原生 HTML 拖放不能接管滑块手势，否则会显示禁拖光标并发送 pointercancel。
        await begin(3);
        const allowed = await rail.evaluate(node => node.dispatchEvent(new DragEvent("dragstart", { bubbles: true, cancelable: true })));
        if (allowed) throw new Error("Native HTML drag was not suppressed");
        await range.press("Escape");
        await page.mouse.up();
        await settled(3);
        results.push("native-drag-blocked-escape-cancels");

        // 使用真实 PDF 预览 iframe；仅上传并准备预览，不执行模型解析。
        await page.locator(".mineru-upload-file input[type=file]").setInputFiles("demo/pdfs/demo1.pdf");
        const iframe = page.locator("iframe.mineru-pdf-frame");
        await iframe.waitFor({ state: "visible" });
        await page.frameLocator("iframe.mineru-pdf-frame").locator("#viewer canvas").first().waitFor({ state: "visible" });
        const preview = await iframe.boundingBox();
        for (const loseCapture of [false, true]) {
            for (let attempt = 0; attempt < 10; attempt++) {
                await begin(0);
                if (loseCapture) await rail.evaluate(node => node.releasePointerCapture(Number(node.dataset.testPointerId)));
                await page.mouse.move(preview.x + 30 + attempt * 5, preview.y + 100, { steps: attempt % 2 === 0 ? 1 : 12 });
                // 透明层确实覆盖预览，不能仅凭最后档位正确就认为跨 iframe 路径得到覆盖。
                const hit = await page.evaluate(({ x, y }) => document.elementFromPoint(x, y)?.className,
                    { x: preview.x + 30 + attempt * 5, y: preview.y + 100 });
                if (hit !== "mineru-tier-drag-surface") throw new Error(`Preview intercepted pointer: ${hit}`);
                await page.mouse.up();
                await settled(3);
            }
            results.push(loseCapture ? "10-pdf-releases-after-capture-loss" : "10-pdf-releases-with-capture");
        }
        // 松手后预览必须重新可点击，不能遗留覆盖层。
        const after = await page.evaluate(({ x, y }) => document.elementFromPoint(x, y)?.tagName,
            { x: preview.x + 30, y: preview.y + 100 });
        if (after !== "IFRAME") throw new Error("PDF preview is still blocked after drag");
        return { passed: results, pdfReleases: 20 };
    } finally {
        await page.mouse.up();
        if (viewport) await page.setViewportSize(viewport);
    }
}
