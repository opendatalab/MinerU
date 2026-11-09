// 用 playwright-cli run-code --filename 执行；先在真实 WebUI 上传 91 页 PDF，分别验证不限页数和 20 页上限。
async (page) => {
    const viewport = page.viewportSize();
    await page.setViewportSize({ width: 1920, height: 1080 });
    const root = page.locator(".mineru-kit-page-range");
    const rail = root.locator(".mineru-page-sliders");
    const a = rail.locator(".mineru-page-handle-a input");
    const b = rail.locator(".mineru-page-handle-b input");
    const start = root.locator(".mineru-page-start input");
    const end = root.locator(".mineru-page-end input");
    const preset = root.locator(".mineru-page-preset");
    await rail.waitFor({ state: "visible" });
    await page.waitForFunction(() => !document.querySelector(".mineru-page-handle-a input").disabled);
    const count = Number(await a.getAttribute("max"));
    if (count !== 91) throw new Error(`Expected 91-page fixture, got ${count}`);
    // 等待本地联动完成，验证滑块、数字框和选中页数始终一致。
    const check = async (expectedStart, expectedEnd) => {
        await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
        const values = [Number(await a.inputValue()), Number(await b.inputValue())].sort((x, y) => x - y);
        const numbers = [Number(await start.inputValue()), Number(await end.inputValue())];
        if (JSON.stringify(values) !== JSON.stringify([expectedStart, expectedEnd])
            || JSON.stringify(numbers) !== JSON.stringify(values)) {
            throw new Error(`Range mismatch: ${JSON.stringify({ values, numbers, expectedStart, expectedEnd })}`);
        }
        if (Number(await root.locator(".mineru-page-selected-count").textContent()) !== expectedEnd - expectedStart + 1) {
            throw new Error("Selected page count is inconsistent");
        }
    };
    // 手势结束后透明层和激活状态必须清空，后续输入及预览仍可操作。
    const finished = async (expectedStart, expectedEnd) => {
        await check(expectedStart, expectedEnd);
        if (await rail.getAttribute("data-active") || await page.locator(".mineru-page-drag-surface").count()) {
            throw new Error("Page drag did not finish cleanly");
        }
    };
    // 按真实轨道几何换算连续页码坐标，不依赖固定窗口大小。
    const geometry = async () => {
        const bounds = await rail.boundingBox();
        return { left: bounds.x + 14, width: bounds.width - 28, y: bounds.y + bounds.height / 2 };
    };
    // 页码坐标统一包含圆点半径留白。
    const x = (g, value) => g.left + (value - 1) / (count - 1) * g.width;
    // 从可见端点开始，记录浏览器真实 pointerId 供捕获丢失检查使用。
    const begin = async handle => {
        await rail.evaluate(node => {
            node.addEventListener("pointerdown", event => { node.dataset.testPointerId = String(event.pointerId); }, { once: true });
        });
        const knob = await rail.locator(`.mineru-page-handle-${handle} .mineru-page-thumb`).boundingBox();
        await page.mouse.move(knob.x + knob.width / 2, knob.y + knob.height / 2);
        await page.mouse.down();
    };
    const requests = [];
    // 上传完成后的选页全部在前端联动，不应出现解析或 Python 事件请求。
    const onRequest = request => {
        if (request.method() === "POST" && /gradio_api\/(queue|run|call)/.test(request.url())) requests.push(request.url());
    };
    let limit;
    try {
        await preset.click();
        await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
        limit = Number(await end.inputValue());
        if (![20, 91].includes(limit)) throw new Error(`Unexpected configured page limit: ${limit}`);
        await finished(1, limit);
        await page.frameLocator("iframe.mineru-pdf-frame").locator("#viewer canvas").first().waitFor({ state: "visible" });
        page.on("request", onRequest);
        const lower = 10;
        const upper = lower + Math.min(40, limit - 1);
        // 用实际数字输入设置基线，覆盖 Gradio 前端状态回执。
        const baseline = async () => {
            await preset.click();
            await finished(1, limit);
            await start.fill(String(lower));
            await start.press("Enter");
            await finished(lower, limit);
            await end.fill(String(upper));
            await end.press("Enter");
            await finished(lower, upper);
            await rail.scrollIntoViewIfNeeded();
        };
        let g;
        for (const handle of ["a", "b"]) {
            await baseline();
            g = await geometry();
            const offset = Math.ceil((upper - lower) / 3);
            const clicked = handle === "a" ? lower + offset : upper - offset;
            await page.mouse.move(x(g, clicked), g.y);
            await page.mouse.down();
            if (await rail.getAttribute("data-active") !== handle) throw new Error("Track did not pick nearest endpoint");
            await check(handle === "a" ? clicked : lower, handle === "b" ? clicked : upper);
            const dragged = handle === "a" ? clicked + 2 : clicked - 2;
            await page.mouse.move(x(g, dragged), g.y + 80);
            await page.mouse.up();
            await finished(handle === "a" ? dragged : lower, handle === "b" ? dragged : upper);
        }

        // 点击选中区域后继续越过另一端，物理身份保持不变，语义起止随之交换。
        await baseline();
        g = await geometry();
        await page.mouse.move(x(g, lower + Math.ceil((upper - lower) / 3)), g.y);
        await page.mouse.down();
        await page.mouse.move(x(g, upper + 8), g.y);
        await page.mouse.up();
        await finished(upper, upper + 8);
        if (Number(await a.inputValue()) !== upper + 8) throw new Error("Crossing changed physical endpoint identity");

        // 模拟实际捕获丢失后在胶囊外松手，范围不能回弹到预设值。
        await preset.click();
        await finished(1, limit);
        g = await geometry();
        await begin("b");
        const target = Math.floor(limit * 0.6);
        await page.mouse.move(x(g, target), g.y);
        await rail.evaluate(node => node.releasePointerCapture(Number(node.dataset.testPointerId)));
        await page.mouse.move(x(g, target - 2), g.y + 120);
        await page.mouse.up();
        await finished(1, target - 2);

        // 原生拖放被阻止，Escape 仍可明确取消当前选择。
        await preset.click();
        await finished(1, limit);
        await begin("b");
        await page.mouse.move(x(g, target), g.y);
        const allowed = await rail.evaluate(node => node.dispatchEvent(new DragEvent("dragstart", { bubbles: true, cancelable: true })));
        if (allowed) throw new Error("Native drag took over page range");
        await b.press("Escape");
        await page.mouse.up();
        await finished(1, limit);

        // 交叉到真正的 PDF 预览区域，捕获存在及丢失两种路径均保持上限约束。
        const preview = await page.locator("iframe.mineru-pdf-frame").boundingBox();
        for (const loseCapture of [false, true]) {
            for (let attempt = 0; attempt < 5; attempt++) {
                await preset.click();
                await finished(1, limit);
                await begin("b");
                if (loseCapture) await rail.evaluate(node => node.releasePointerCapture(Number(node.dataset.testPointerId)));
                const point = { x: preview.x + 30 + attempt * 5, y: preview.y + 150 };
                await page.mouse.move(point.x, point.y, { steps: attempt % 2 === 0 ? 1 : 12 });
                if (await page.evaluate(p => document.elementFromPoint(p.x, p.y)?.className, point) !== "mineru-page-drag-surface") {
                    throw new Error("PDF preview intercepted the active page gesture");
                }
                await page.mouse.up();
                await finished(count - limit + 1, count);
            }
        }
        await b.press("ArrowLeft");
        await finished(count - limit + 1, count - 1);
        await preset.click();
        await finished(1, limit);

        // 窄屏短选区可能被两个圆点覆盖，仍须从真实端点连续拖动。
        await page.setViewportSize({ width: 320, height: 1000 });
        await rail.scrollIntoViewIfNeeded();
        g = await geometry();
        const clicked = Math.ceil(limit * 0.65);
        await begin("b");
        await page.mouse.move(x(g, clicked + 1), g.y);
        await page.mouse.up();
        await finished(1, clicked + 1);
        if (requests.length) throw new Error(`Unexpected backend calls: ${requests.join(", ")}`);
        return { pages: count, limit: limit === count ? "unlimited" : limit, pdfReleases: 10, backendCalls: requests.length,
            checks: ["nearest-left", "nearest-right", "continuous-track-drag", "crossing", "capture-loss-outside",
                "native-drag-blocked", "escape", "pdf-release", "keyboard", "reset", "narrow-screen"] };
    } finally {
        page.off("request", onRequest);
        await page.mouse.up();
        if (viewport) await page.setViewportSize(viewport);
    }
}
