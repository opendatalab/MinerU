// 用 playwright-cli run-code --filename 执行；须先打开包含四档能力的真实 WebUI。
async (page) => {
    const viewport = page.viewportSize();
    await page.setViewportSize({ width: 1920, height: 1080 });
    const capsule = page.locator(".mineru-tier-capsule");
    const rail = capsule.locator(".mineru-tier-rail");
    const range = capsule.locator(".mineru-tier-range");
    const thumb = capsule.locator(".mineru-tier-thumb");
    await rail.waitFor({ state: "visible" });
    await rail.scrollIntoViewIfNeeded();
    if (await range.getAttribute("max") !== "3") throw new Error("This check requires all four tiers");
    const requests = [];
    // 档位拖动和联动必须留在浏览器中，不能为每次移动发送 Python 请求。
    const onRequest = request => {
        if (request.method() === "POST" && /gradio_api\/(queue|run|call)/.test(request.url())) requests.push(request.url());
    };
    page.on("request", onRequest);
    // 等待现有 CSS 吸附动画完成，仅在测试需要稳定起点时使用。
    const settle = async () => {
        await capsule.evaluate(async node => {
            await Promise.all(node.getAnimations({ subtree: true }).map(animation => animation.finished.catch(() => {})));
            await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
        });
    };
    // 从实际布局读取轨道端点，覆盖缩放、窄屏及圆点半径留白。
    const geometry = async () => {
        const bounds = await rail.boundingBox();
        const knob = await thumb.boundingBox();
        return { left: bounds.x + knob.width / 2, width: bounds.width - knob.width, y: bounds.y + bounds.height / 2 };
    };
    // 核对最终值和拖动态，等待 Gradio 回执后也不能回弹。
    const checkValue = async value => {
        await settle();
        const actual = Number(await range.inputValue());
        if (actual !== value || await capsule.evaluate(node => node.classList.contains("is-dragging"))) {
            throw new Error(`Expected settled tier ${value}, got ${actual}`);
        }
        if (await page.locator(".mineru-tier-drag-surface").count()) throw new Error("Drag surface was not removed");
    };
    // 连续拖动期间，实际可见圆点必须跟随鼠标，而不是只在松手后得到正确整数。
    const checkPosition = async expectedX => {
        const box = await thumb.boundingBox();
        const error = Math.abs(box.x + box.width / 2 - expectedX);
        if (error > 1.5) throw new Error(`Tier thumb lags pointer by ${error}px`);
        if (!await capsule.evaluate(node => node.classList.contains("is-dragging"))) throw new Error("Drag was interrupted");
        return error;
    };
    // 检查最终计算样式，防止浏览器或 Gradio 的默认焦点框绕过交互状态标记。
    const checkFocus = async keyboard => {
        const styles = await capsule.evaluate(node => {
            const input = getComputedStyle(node.querySelector(".mineru-tier-range"));
            const knob = getComputedStyle(node.querySelector(".mineru-tier-thumb"));
            return { inputOutline: input.outlineStyle, inputShadow: input.boxShadow, knobOutline: knob.outlineStyle };
        });
        if (styles.inputOutline !== "none" || styles.inputShadow !== "none"
            || styles.knobOutline !== (keyboard ? "solid" : "none")) {
            throw new Error(`Unexpected tier focus decoration: ${JSON.stringify(styles)}`);
        }
    };
    let maxError = 0;
    try {
        // 宽屏保证真实覆盖重置按钮；先恢复默认，允许在已操作过的页面重复执行。
        const reset = capsule.locator(".mineru-tier-reset");
        await reset.waitFor({ state: "visible" });
        if (await reset.isEnabled()) await reset.click();
        await settle();
        const initial = Number(await range.inputValue());
        await range.press("End");
        await checkValue(3);
        let g = await geometry();
        await page.mouse.move(g.left + g.width, g.y);
        await page.mouse.down();
        if (await rail.getAttribute("data-keyboard") !== "false") throw new Error("Pointer focus kept keyboard outline");
        await checkFocus(false);
        for (const fraction of [0.9, 0.75, 0.51, 0.36, 0.16, 0, 0.33, 0.68, 1]) {
            const x = g.left + fraction * g.width;
            await page.mouse.move(x, g.y);
            maxError = Math.max(maxError, await checkPosition(x));
        }
        // 松手前最后一帧是小数，最终只能提交一个整数档位。
        await page.mouse.move(g.left + 0.43 * g.width, g.y);
        await page.mouse.up();
        await checkValue(1);
        await checkFocus(false);

        // 快速往返且斜向移出轨道，再在轨道外松手，不能回到旧档位。
        await range.press("End");
        await checkValue(3);
        for (let index = 0; index < 20; index++) {
            const target = index % 2 === 0 ? 0 : 3;
            const knob = await thumb.boundingBox();
            await page.mouse.move(knob.x + knob.width / 2, knob.y + knob.height / 2);
            await page.mouse.down();
            await page.mouse.move(g.left + target / 3 * g.width, g.y + (index % 2 === 0 ? 70 : -60));
            await page.mouse.up();
            await checkValue(target);
        }

        // 点击轨道要定位并吸附；按住圆点边缘应保留抓取偏移。
        await page.mouse.click(g.left + 0.34 * g.width, g.y);
        await checkValue(1);
        await page.mouse.move(g.left + g.width / 3 + 10, g.y);
        await page.mouse.down();
        await page.mouse.move(g.left + 2 * g.width / 3 + 10, g.y);
        maxError = Math.max(maxError, await checkPosition(g.left + 2 * g.width / 3));
        await page.mouse.up();
        await checkValue(2);

        // 在吸附动画期间再次抓住圆点，检查可见位置与重新拖动的位置衔接。
        await page.mouse.move(g.left + 2 * g.width / 3, g.y);
        await page.mouse.down();
        await page.mouse.move(g.left + 0.51 * g.width, g.y);
        await page.mouse.up();
        const moving = await thumb.boundingBox();
        const grabbedX = moving.x + moving.width / 2;
        await page.mouse.move(grabbedX, g.y);
        await page.mouse.down();
        const resumed = await thumb.boundingBox();
        const offset = grabbedX - resumed.x - resumed.width / 2;
        if (Math.abs(offset) > resumed.width / 2 + 1.5) throw new Error("Re-grab jumped away from visible thumb");
        await page.mouse.move(g.left + 0.2 * g.width + offset, g.y);
        maxError = Math.max(maxError, await checkPosition(g.left + 0.2 * g.width));
        await page.mouse.up();
        await checkValue(1);

        // Escape 取消后，迟到的 pointerup 不能提交取消前的位置。
        await page.mouse.move(g.left + g.width / 3, g.y);
        await page.mouse.down();
        await page.mouse.move(g.left + g.width, g.y);
        await range.press("Escape");
        await page.mouse.up();
        await checkValue(1);
        await range.press("Home");
        await checkValue(0);
        await range.press("ArrowRight");
        await checkValue(1);
        if (await rail.getAttribute("data-keyboard") !== "true") throw new Error("Keyboard focus indicator is missing");
        await checkFocus(true);
        await range.press("End");
        await checkValue(3);
        // 通过原生输入事件模拟辅助技术，指针修复不能移除无障碍操作通道。
        await range.evaluate(node => {
            node.value = "1";
            node.dispatchEvent(new Event("input", { bubbles: true }));
            node.dispatchEvent(new Event("change", { bubbles: true }));
        });
        await checkValue(1);
        await reset.click();
        await checkValue(initial);
        // 在窄屏重新读取几何后拖动，不依赖桌面固定宽度。
        await page.setViewportSize({ width: 320, height: 900 });
        await rail.scrollIntoViewIfNeeded();
        await settle();
        g = await geometry();
        await page.mouse.click(g.left, g.y);
        await checkValue(0);
        await page.mouse.move(g.left, g.y);
        await page.mouse.down();
        await page.mouse.move(g.left + g.width, g.y);
        await page.mouse.up();
        await checkValue(3);
        if (requests.length) throw new Error(`Unexpected backend calls during tier gestures: ${requests.join(", ")}`);
        return { continuousPositions: 9, rapidDrags: 20, maxErrorPx: maxError, backendCalls: requests.length,
            checks: ["snap", "outside-release", "track-click", "grab-offset", "animation-regrab", "escape",
                "keyboard", "focus-decoration", "assistive-input", "reset", "narrow-screen"] };
    } finally {
        page.off("request", onRequest);
        await page.mouse.up();
        if (viewport) await page.setViewportSize(viewport);
    }
}
