// 使用 playwright-cli run-code --filename 执行，页面须已通过真实 WebUI 完成 PDF 或图片解析。
async (page) => {
    const pdf = await (await page.locator("iframe.mineru-pdf-frame").elementHandle()).contentFrame();
    const html = await (await page.locator("iframe.mineru-rendered-html-frame").elementHandle()).contentFrame();
    const viewport = page.viewportSize() || await page.evaluate(() => ({ width: innerWidth, height: innerHeight }));
    // 从生产预览配置读取期望布局，逐项核对实际 DOM 和页面几何。
    const config = await html.evaluate(() => {
        const script = [...document.scripts].find(node => node.textContent.includes('"layoutRegions":'));
        return JSON.parse(script.textContent.slice(script.textContent.lastIndexOf(")(") + 2, -2));
    });
    const firstPage = Math.min(...config.layoutRegions.map(region => region.page_idx)) + 1;
    const pageCount = Number(await pdf.locator("#pages").textContent());
    const toggle = pdf.locator("#trace-toggle");
    if (await toggle.getAttribute("aria-pressed") !== "true") await toggle.click();
    // 翻页必须等待当前页浮层完成挂载，覆盖 PDF.js 缓存重建。
    const goToPage = async (number) => {
        await page.locator("iframe.mineru-pdf-frame").scrollIntoViewIfNeeded();
        await pdf.locator("#page").fill(String(number));
        await pdf.locator("#page").press("Enter");
        await pdf.locator(`.page[data-page-number="${number}"] .mineru-layout-layer`).waitFor({ state: "attached" });
    };
    // 坐标以实际页框为基准，标签允许收缩但类型、编号和身份必须完整。
    const checkPage = async (number) => {
        await goToPage(number);
        const expected = config.layoutRegions.filter(region => region.page_idx === number - 1);
        await pdf.waitForFunction(({ number, expected }) => {
            const host = document.querySelector(`.page[data-page-number="${number}"]`);
            const layer = host?.querySelector(".mineru-layout-layer");
            if (!layer) return false;
            const bounds = layer.getBoundingClientRect();
            const nodes = [...layer.children];
            if (nodes.length !== expected.length) throw new Error("Layout region count differs from shared collector");
            for (const region of expected) {
                const key = JSON.stringify([region.page_idx, region.block_index, region.child_path]);
                const node = nodes.find(item => item.dataset.sourceKey === key);
                if (!node || node.dataset.blockType !== region.type || node.dataset.blockIndex !== String(region.index ?? "")) {
                    throw new Error(`Incorrect original block identity ${key}`);
                }
                const rect = node.getBoundingClientRect();
                const actual = [(rect.left - bounds.left) / bounds.width, (rect.top - bounds.top) / bounds.height,
                    (rect.right - bounds.left) / bounds.width, (rect.bottom - bounds.top) / bounds.height];
                if (actual.some((value, index) => Math.abs(value - region.bbox[index]) > 0.003)) {
                    throw new Error(`Misaligned source region ${key}`);
                }
                const label = node.firstElementChild.getBoundingClientRect();
                if (label.left < bounds.left - 2 || label.right > bounds.right + 2 || label.top < bounds.top - 2) {
                    throw new Error(`Label outside page ${key}`);
                }
                const expanded = node.classList.contains("expanded");
                const number = region.index == null
                    ? `${region.block_index + 1}/${region.child_path.map(index => index + 1).join(".")}`
                    : String(region.index + 1);
                const expectedLabel = expanded ? `${region.type} · ${number}` : number;
                if (node.textContent !== expectedLabel) throw new Error(`Incorrect compact/expanded label ${key}`);
                const style = getComputedStyle(node.firstElementChild);
                if (style.fontSize !== (expanded ? "11px" : "10px")) throw new Error("Label scaled with PDF");
                if (!expanded && style.backgroundColor !== "rgba(0, 0, 0, 0)") throw new Error("Compact number obscures text");
            }
            if (document.querySelectorAll(".mineru-layout-region.expanded").length > 1) throw new Error("Multiple full labels shown");
            if (getComputedStyle(layer).pointerEvents !== "none") throw new Error("Decorations intercept source clicks");
            return true;
        }, { number, expected }, { polling: 100 });
        return expected.length;
    };
    const checkedPages = [];
    for (const number of [...new Set([firstPage, Math.min(firstPage + 1, pageCount), pageCount])]) {
        checkedPages.push({ number, regions: await checkPage(number) });
    }
    await goToPage(firstPage);
    const sourceButton = pdf.locator(` .page[data-page-number="${firstPage}"] button[data-trace-id]`.trim()).last();
    const sourceId = await sourceButton.getAttribute("data-trace-id");
    await sourceButton.click();
    await html.waitForFunction(id => document.querySelector("[data-mineru-selected]")?.dataset.docvortexTraceId === id, sourceId);
    // 左右高亮必须取当前物理块的同一类型色，覆盖同框父子以及跨页来源切换。
    const checkTheme = async () => {
        await pdf.waitForFunction(() => document.querySelector(".mineru-layout-region.expanded.selected"));
        const htmlTheme = await html.locator("[data-mineru-selected]").evaluate(node => ({
            color: getComputedStyle(node).outlineColor, fill: getComputedStyle(node).backgroundColor
        }));
        const result = await pdf.evaluate(() => {
            const region = document.querySelector(".mineru-layout-region.expanded");
            const box = [...region.closest(".page").querySelectorAll("button.selected")].find(node =>
                ["left", "top", "width", "height"].every(name => node.style[name] === region.style[name]));
            return { color: getComputedStyle(box).outlineColor, fill: getComputedStyle(box).backgroundColor, type: region.dataset.blockType };
        });
        if (result.color !== htmlTheme.color || result.fill !== htmlTheme.fill) {
            throw new Error(`Source/result highlight colors differ for ${result.type}`);
        }
        return result;
    };
    const colors = [await checkTheme()];
    // 使用真实正文键盘入口验证反向定位，原始页号允许抽页后从非零页开始。
    const targetId = await html.locator("[data-docvortex-trace-id]").first().getAttribute("data-docvortex-trace-id");
    await html.locator(`[data-docvortex-trace-id="${targetId}"]`).press("Enter");
    await pdf.waitForFunction(id => [...document.querySelectorAll("button.selected")].some(node => node.traceIds.includes(id)), targetId);
    colors.push(await checkTheme());
    // 检查其他块时只展开该块，移开后恢复已选中块的完整标签。
    const selectedKey = await pdf.locator(".mineru-layout-region.expanded").getAttribute("data-source-key");
    await toggle.hover();
    await sourceButton.hover();
    const hoveredKey = await sourceButton.getAttribute("data-source-key");
    await pdf.waitForFunction(key => document.querySelector(".mineru-layout-region.expanded")?.dataset.sourceKey === key, hoveredKey);
    await toggle.hover();
    await pdf.waitForFunction(key => document.querySelector(".mineru-layout-region.expanded")?.dataset.sourceKey === key, selectedKey);
    const inspector = pdf.locator(`.page[data-page-number="${firstPage}"] .mineru-layout-inspector`).first();
    if (await inspector.count()) {
        const key = await inspector.getAttribute("data-source-key");
        await inspector.focus();
        const region = config.layoutRegions.find(item => JSON.stringify([item.page_idx, item.block_index, item.child_path]) === key);
        await pdf.waitForFunction(({ key, type }) => [...document.querySelectorAll(".mineru-layout-region.expanded")]
            .some(node => node.dataset.sourceKey === key && node.textContent.startsWith(type)), { key, type: region.type });
        await toggle.focus();
    }
    // 合并后的一个正文目标应在不同来源页显示不同原始编号。
    const merged = await html.evaluate(() => [...document.querySelectorAll("[data-docvortex-trace-id]")].map(node => ({
        id: node.dataset.docvortexTraceId, sources: JSON.parse(node.dataset.docvortexSources)
    })).find(entry => new Set(entry.sources.map(source => source.page_idx)).size > 1));
    if (merged) {
        await html.locator(`[data-docvortex-trace-id="${merged.id}"]`).press("Enter");
        for (const source of merged.sources) {
            await html.getByRole("button", { name: new RegExp(` ${source.page_idx + 1}$`) }).click();
            const number = source.page_idx + 1;
            await pdf.waitForFunction(number => Number(document.querySelector("#page").value) === number, number);
            await checkPage(number);
            const key = JSON.stringify([source.page_idx, source.block_index, source.child_path]);
            const region = config.layoutRegions.find(item => JSON.stringify([item.page_idx, item.block_index, item.child_path]) === key);
            if (region) {
                const annotation = pdf.locator(".mineru-layout-region").filter({ has: pdf.locator(".mineru-layout-label") });
                const active = await annotation.evaluateAll((nodes, key) => nodes.find(node => node.dataset.sourceKey === key)?.classList.contains("selected"), key);
                if (!active) throw new Error(`Merged target lost source selection ${key}`);
            }
        }
    }
    // 一个合并目标在同一页有多个原始块时，点击后的完整标签绑定实际点击的来源。
    const samePage = await html.evaluate(() => [...document.querySelectorAll("[data-docvortex-trace-id]")]
        .map(node => JSON.parse(node.dataset.docvortexSources))
        .find(sources => sources.length > new Set(sources.map(source => source.page_idx)).size));
    if (samePage) {
        for (const source of samePage) {
            await goToPage(source.page_idx + 1);
            const key = JSON.stringify([source.page_idx, source.block_index, source.child_path]);
            const buttons = pdf.locator("button[data-trace-id]");
            const index = await buttons.evaluateAll((nodes, key) => nodes.findIndex(node => node.dataset.sourceKey === key), key);
            if (index < 0) continue;
            await buttons.nth(index).click();
            await pdf.waitForFunction(key => document.querySelector(".mineru-layout-region.expanded")?.dataset.sourceKey === key, key);
        }
    }
    // 开关仅控制常驻标注，当前选中标签始终可见。
    await toggle.click();
    await pdf.waitForFunction(() => [...document.querySelectorAll(".mineru-layout-region:not(.selected):not(.hovered)")]
        .every(node => getComputedStyle(node).visibility === "hidden"));
    await toggle.click();
    await goToPage(firstPage);
    const oldScale = await pdf.locator("#scale").innerText();
    await pdf.locator("#zoom-in").click();
    await pdf.waitForFunction(value => document.querySelector("#scale").textContent !== value, oldScale);
    await checkPage(firstPage);
    const selectedSources = await html.evaluate(() => {
        const selected = document.querySelector("[data-mineru-selected]");
        return selected ? JSON.parse(selected.dataset.docvortexSources) : [];
    });
    const selectedKeys = config.layoutRegions.filter(region => selectedSources.some(source =>
        source.page_idx === region.page_idx && source.bbox.every((value, index) => value === region.bbox[index])))
        .map(region => JSON.stringify([region.page_idx, region.block_index, region.child_path]));
    const activeSources = await pdf.locator(".mineru-layout-region.expanded").evaluateAll(nodes => nodes.map(node => node.dataset.sourceKey));
    if (activeSources.some(key => !selectedKeys.includes(key))) throw new Error("Zoom retained an old hover label");
    await pdf.locator("#fit").click();
    await page.setViewportSize({ width: 375, height: 900 });
    await page.locator("iframe.mineru-pdf-frame").scrollIntoViewIfNeeded();
    await checkPage(firstPage);
    // 辅助块的压缩标签也能通过键盘查看完整类型，且使用独立的只读检查元素。
    const compactInspector = await pdf.locator(".mineru-layout-inspector").evaluateAll(nodes => nodes.findIndex(node => {
        const region = [...node.closest(".page").querySelectorAll(".mineru-layout-region")]
            .find(item => item.dataset.sourceKey === node.dataset.sourceKey);
        return region && !region.textContent.startsWith(region.dataset.blockType);
    }));
    if (compactInspector >= 0) {
        const inspector = pdf.locator(".mineru-layout-inspector").nth(compactInspector);
        const key = await inspector.getAttribute("data-source-key");
        await inspector.focus();
        await pdf.waitForFunction(key => [...document.querySelectorAll(".mineru-layout-region.expanded")]
            .some(node => node.dataset.sourceKey === key && node.textContent.startsWith(node.dataset.blockType)), key, { polling: 100 });
        await toggle.focus();
    }
    await page.screenshot({ path: "output/playwright/layout-annotations/chrome-narrow.png" });
    const overflow = await pdf.evaluate(() => document.getElementById("toolbar").scrollWidth > innerWidth);
    if (overflow) throw new Error("Narrow toolbar overflows");
    await page.setViewportSize(viewport);
    await checkPage(firstPage);
    return {
        checkedPages, sourceId, targetId, colors, mergedSources: merged?.sources, samePageSources: samePage,
        layoutRegions: config.layoutRegions.length, narrow: true, zoom: true, inspectedAuxiliary: compactInspector >= 0
    };
}
