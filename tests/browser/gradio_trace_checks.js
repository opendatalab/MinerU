// 使用 playwright-cli run-code --filename 执行；页面须已通过 WebUI 解析至少两页 PDF。
// 在首次解析、换文件和重复解析后各执行一次，WebKit 会实际运行查看器的 ES module。
async (page) => {
    // 直接观察浏览器生成的 MessageEvent.source，覆盖 Node 消息替身无法体现的来源归属。
    await page.evaluate(() => {
        const controller = new AbortController();
        window.__mineruTraceBrowserCheck = { controller, events: [] };
        window.addEventListener("message", (event) => {
            if (event.data?.channel !== "mineru-trace" || event.data.type !== "select") return;
            const pdf = document.querySelector("iframe.mineru-pdf-frame");
            const html = document.querySelector("iframe.mineru-rendered-html-frame");
            window.__mineruTraceBrowserCheck.events.push({
                id: event.data.id,
                documentId: event.data.documentId,
                fromPdf: event.source === pdf?.contentWindow,
                fromHtml: event.source === html?.contentWindow,
                sameOrigin: event.origin === location.origin,
            });
        }, { signal: controller.signal });
    });
    try {
        await page.frameLocator("iframe.mineru-pdf-frame").locator("button[data-trace-id]").first().waitFor();
        const pdf = await (await page.locator("iframe.mineru-pdf-frame").elementHandle()).contentFrame();
        const html = await (await page.locator("iframe.mineru-rendered-html-frame").elementHandle()).contentFrame();
        const documentId = new URLSearchParams(new URL(pdf.url()).hash.slice(1)).get("documentId");
        if (!documentId) throw new Error("Convert the PDF before checking source navigation");
        const pageInput = pdf.locator("#page");
        if (Number(await pageInput.getAttribute("max")) < 2) throw new Error("Use a PDF with at least two pages");

        await pageInput.fill("2");
        await pageInput.press("Enter");
        const secondPage = pdf.locator('.page[data-page-number="2"]');
        const box = secondPage.locator("button[data-trace-id]").first();
        await box.waitFor();
        const targetId = await box.getAttribute("data-trace-id");
        await box.click();
        await page.waitForFunction(({ id, documentId }) =>
            window.__mineruTraceBrowserCheck.events.some(event => event.id === id && event.documentId === documentId),
        { id: targetId, documentId });
        const received = await page.evaluate(({ id, documentId }) =>
            window.__mineruTraceBrowserCheck.events.find(event => event.id === id && event.documentId === documentId),
        { id: targetId, documentId });
        if (!received.fromPdf || !received.sameOrigin) {
            throw new Error(`PDF message source was misidentified: ${JSON.stringify(received)}`);
        }
        // 等待消息往返和布局校正完成，同时验证目标块进入正文视口。
        await html.waitForFunction(id => {
            const selected = document.querySelector("[data-mineru-selected]");
            if (selected?.dataset.docvortexTraceId !== id) return false;
            const bounds = selected.getBoundingClientRect();
            return bounds.top < innerHeight && bounds.bottom > 0;
        }, targetId);
        const resultScroll = await html.evaluate(() => window.scrollY);

        // 从结果正文选择第一页的普通文字块，验证反向定位和 PDF 区域高亮。
        const firstText = html.locator('[data-page-idx="0"][data-docvortex-trace-id][data-block-type="text"]').first();
        const firstId = await firstText.getAttribute("data-docvortex-trace-id");
        await firstText.press("Enter");
        await pdf.waitForFunction(id => {
            return document.querySelector("#page").value === "1" &&
                [...document.querySelectorAll('.page[data-page-number="1"] button.selected')]
                    .some(node => node.traceIds.includes(id));
        }, firstId);
        const fromHtml = await page.evaluate(({ id, documentId }) =>
            window.__mineruTraceBrowserCheck.events.some(event =>
                event.id === id && event.documentId === documentId && event.fromHtml),
        { id: firstId, documentId });
        if (!fromHtml) throw new Error("HTML source window was not identified");
        return { documentId, pdfToHtml: targetId, htmlToPdf: firstId, resultScroll };
    } finally {
        // 验收结束后释放探针，便于同一会话继续切换文档并再次运行。
        await page.evaluate(() => {
            window.__mineruTraceBrowserCheck.controller.abort();
            delete window.__mineruTraceBrowserCheck;
        });
    }
}
