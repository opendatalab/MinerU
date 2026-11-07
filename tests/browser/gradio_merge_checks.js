// 使用 playwright-cli run-code --filename 执行，页面须已通过真实 WebUI 完成 Basic 解析。
async (page) => {
    const pdfElement = page.locator("iframe.mineru-pdf-frame");
    const pdf = await (await pdfElement.elementHandle()).contentFrame();
    const html = await (await page.locator("iframe.mineru-rendered-html-frame").elementHandle()).contentFrame();
    const viewport = page.viewportSize() || await page.evaluate(() => ({ width: innerWidth, height: innerHeight }));
    const data = await html.evaluate(() => {
        const script = [...document.scripts].find(node => node.textContent.includes('"layoutRegions":'));
        const config = JSON.parse(script.textContent.slice(script.textContent.lastIndexOf(")(") + 2, -2));
        const entries = [...document.querySelectorAll("[data-docvortex-trace-id]")].map(node => ({
            id: node.dataset.docvortexTraceId, type: node.dataset.blockType,
            sources: JSON.parse(node.dataset.docvortexSources)
        }));
        return { documentId: config.documentId, entries };
    });
    const merged = data.entries.filter(entry => entry.sources.length > 1);
    if (!merged.length) throw new Error("The real parse result has no merged sources to check");
    const checked = [];
    // 从真实结果块激活选择，覆盖 HTML 到 PDF 的消息链。
    const selectEntry = async (entry) => {
        await html.locator(`[data-docvortex-trace-id="${entry.id}"]`).press("Enter");
        await pdf.waitForFunction(id => document.querySelector(".mineru-merge-layer")?.dataset.traceId === id, entry.id);
    };
    // 通过页码控件取得端点页面的实际尺寸，再把线段中点滚入视口。
    const showEdge = async (entry, index) => {
        for (const source of entry.sources.slice(index, index + 2)) {
            const number = source.page_idx + 1;
            await pdf.locator("#page").fill(String(number));
            await pdf.locator("#page").press("Enter");
            await pdf.locator(`.page[data-page-number="${number}"] .mineru-trace-layer`).waitFor({ state: "attached" });
        }
        await pdf.evaluate(({ entry, index }) => {
            const host = document.getElementById("viewer-container");
            const sources = entry.sources.slice(index, index + 2);
            const bounds = sources.map(source => document.querySelector(
                `.page[data-page-number="${source.page_idx + 1}"]`
            ).getBoundingClientRect());
            const middle = (bounds[0].top + sources[0].bbox[3] * bounds[0].height +
                bounds[1].top + sources[1].bbox[1] * bounds[1].height) / 2;
            host.scrollTop += middle - host.getBoundingClientRect().top - host.clientHeight / 2;
        }, { entry, index });
        await pdf.locator("#toolbar").hover();
    };
    // 直接比较 SVG 端点和实际命中框的角点，允许最多 2 CSS px 的亚像素排版误差。
    const checkEdges = async (entry) => {
        await pdf.waitForFunction(({ entry }) => {
            const svg = document.querySelector(".mineru-merge-layer");
            if (svg?.dataset.traceId !== entry.id || !svg.children.length) return false;
            const origin = svg.getBoundingClientRect();
            for (const group of svg.children) {
                const from = JSON.parse(group.dataset.fromSource);
                const to = JSON.parse(group.dataset.toSource);
                const keys = entry.sources.map(source => JSON.stringify([source.page_idx, source.block_index, source.child_path]));
                const index = keys.indexOf(JSON.stringify(from));
                if (index < 0 || keys[index + 1] !== JSON.stringify(to)) throw new Error("Merge source order changed");
                const rects = entry.sources.slice(index, index + 2).map(source => {
                    const pageNode = document.querySelector(`.page[data-page-number="${source.page_idx + 1}"]`);
                    const node = [...pageNode.querySelectorAll("button[data-trace-id]")].find(node =>
                        node.traceIds.includes(entry.id) && node.source.bbox.every((value, i) => value === source.bbox[i]));
                    if (!node) throw new Error("Missing rendered source hit box");
                    return node.getBoundingClientRect();
                });
                const line = group.querySelector("line");
                const actual = [Number(line.getAttribute("x1")) + origin.left, Number(line.getAttribute("y1")) + origin.top,
                    Number(line.getAttribute("x2")) + origin.left, Number(line.getAttribute("y2")) + origin.top];
                const expected = [rects[0].right, rects[0].bottom, rects[1].left, rects[1].top];
                if (actual.some((value, i) => Math.abs(value - expected[i]) > 2)) return false;
                if (getComputedStyle(line).strokeWidth !== "1.5px") throw new Error("Merge stroke scaled with PDF");
                const label = group.querySelector("text");
                if (label && getComputedStyle(label).fontSize !== "11px") throw new Error("Merge label scaled with PDF");
            }
            if (getComputedStyle(svg).pointerEvents !== "none") throw new Error("Merge layer intercepts clicks");
            return true;
        }, { entry });
    };
    for (const entry of merged) {
        await selectEntry(entry);
        for (let index = 0; index < entry.sources.length - 1; index++) {
            await showEdge(entry, index);
            await checkEdges(entry);
        }
        checked.push({ id: entry.id, type: entry.type, sources: entry.sources });
    }
    // 优先记录跨栏后跨页的最长链，续表样本优先记录有具体几何的正文子块。
    const representative = [...merged].sort((a, b) => b.sources.length - a.sources.length ||
        b.sources[0].child_path.length - a.sources[0].child_path.length)[0];
    await selectEntry(representative);
    await showEdge(representative, 0);
    await checkEdges(representative);
    const prefix = `output/playwright/merge-connections/${data.documentId}`;
    await pdfElement.screenshot({ path: `${prefix}-desktop.png` });
    // 鼠标检查单块临时隐藏连接，离开后恢复选择；键盘检查复用同一规则。
    const single = data.entries.find(entry => entry.sources.length === 1 &&
        entry.sources[0].page_idx === representative.sources[0].page_idx);
    if (single) {
        const box = pdf.locator(`button[data-trace-id="${single.id}"]`).first();
        if (await box.count()) {
            await box.hover();
            await pdf.waitForFunction(() => !document.querySelector(".mineru-merge-layer").children.length);
            await pdf.locator("#toolbar").hover();
            await pdf.waitForFunction(id => document.querySelector(".mineru-merge-layer").dataset.traceId === id, representative.id);
            await box.focus();
            await pdf.waitForFunction(() => !document.querySelector(".mineru-merge-layer").children.length);
            await pdf.locator("#trace-toggle").focus();
        }
    }
    await showEdge(representative, 0);
    await pdf.locator("#trace-toggle").click();
    await checkEdges(representative);
    await pdf.locator("#trace-toggle").click();
    await pdf.locator("#zoom-in").click();
    await showEdge(representative, 0);
    await checkEdges(representative);
    await pdfElement.screenshot({ path: `${prefix}-zoom.png` });
    await pdf.locator("#fit").click();
    await page.setViewportSize({ width: 375, height: 900 });
    await pdfElement.scrollIntoViewIfNeeded();
    await showEdge(representative, 0);
    await checkEdges(representative);
    await pdfElement.screenshot({ path: `${prefix}-narrow.png` });
    await page.setViewportSize(viewport);
    await showEdge(representative, 0);
    await checkEdges(representative);
    return { documentId: data.documentId, checked, screenshots: prefix, geometryTolerance: 2, zoom: true, narrow: true };
}
