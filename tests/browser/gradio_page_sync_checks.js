// 使用 playwright-cli run-code --filename 在已转换的 demo1.pdf 或 demo2.pdf 上执行。
async (page) => {
    const pdfHost = page.locator("iframe.mineru-pdf-frame");
    const htmlHost = page.locator("iframe.mineru-rendered-html-frame");
    const pdf = await (await pdfHost.elementHandle()).contentFrame();
    const html = await (await htmlHost.elementHandle()).contentFrame();
    const sync = pdf.locator("#page-sync-toggle");
    const pageNumber = pdf.locator("#page");
    // 读取真实渲染节点，按物理横坐标挑选同一原始页左右两栏的正文及下一页的正文。
    const sample = await html.evaluate(() => {
        const blocks = [...document.querySelectorAll('[data-block-type="text"][data-docvortex-trace-id]')]
            .map(node => ({ id: node.dataset.docvortexTraceId, sources: JSON.parse(node.dataset.docvortexSources) }))
            .filter(({ sources }) => sources.length === 1);
        for (const left of blocks) {
            const source = left.sources[0];
            if (source.page_idx === 0 || source.bbox[2] > 0.53) continue;
            const right = blocks.find(b => b.sources[0].page_idx === source.page_idx && b.sources[0].bbox[0] >= 0.49);
            const next = [...document.querySelectorAll("[data-docvortex-trace-id]")].find(node =>
                node.getBoundingClientRect().height && JSON.parse(node.dataset.docvortexSources).every(s => s.page_idx === source.page_idx + 1));
            if (right && next) return { page: source.page_idx + 1, left: left.id, right: right.id, next: next.dataset.docvortexTraceId };
        }
        throw new Error("样本文档缺少相邻页及可区分的两栏正文");
    });
    // 等待消息、PDF.js 可见页计算和动画帧结束，捕获延迟回跳。
    const settle = async () => {
        for (const frame of [pdf, html]) await frame.evaluate(() =>
            new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
    };
    // 使用实际滚轮把目标块送到阅读线，保留 WebKit 的滚轮步长与异步滚动语义。
    const wheelTo = async (id, viewportFraction = 0) => {
        const node = html.locator('[data-docvortex-trace-id="' + id + '"]');
        await htmlHost.hover({ position: { x: 110, y: 140 } });
        for (let attempt = 0; attempt < 12; attempt++) {
            const dy = await node.evaluate((n, fraction) => {
                const offset = parseFloat(getComputedStyle(document.body).getPropertyValue("--mineru-preview-offset")) || 24;
                return n.getBoundingClientRect().top - offset - fraction * (innerHeight - offset) + 8;
            }, viewportFraction);
            if (Math.abs(dy) < 4) return;
            await page.mouse.wheel(0, Math.round(dy));
            await settle();
        }
        throw new Error("滚轮没有到达目标正文 " + id);
    };
    // 直接观察预览容器位置，页码相同也能发现同页被滚回页首的问题。
    const pdfTop = () => pdf.locator("#viewer-container").evaluate(node => node.scrollTop);
    // 页码和 HTML 位置分别等到预期值，避免以消息送达代替实际布局。
    const waitPage = n => pdf.waitForFunction(n => Number(document.querySelector("#page").value) === n, n);
    const check = (ok, message) => { if (!ok) throw new Error(message); };
    check(await sync.getAttribute("aria-pressed") === "true", "应默认开启");
    await wheelTo(sample.left);
    await waitPage(sample.page);
    await pdfHost.hover({ position: { x: 120, y: 250 } });
    await page.mouse.wheel(0, 45);
    await settle();
    const beforeColumn = await pdfTop();
    await wheelTo(sample.right);
    check(await pdfTop() === beforeColumn && Number(await pageNumber.inputValue()) === sample.page, "同页换栏移动了 PDF");
    // 记录 PDF 视口实际经过的页码，捕获跨页后的反向回跳。
    await pdf.evaluate(() => {
        const input = document.querySelector("#page");
        const pages = [Number(input.value)];
        const controller = new AbortController();
        document.querySelector("#viewer-container").addEventListener("scroll", () => {
            const value = Number(input.value);
            if (pages.at(-1) !== value) pages.push(value);
        }, { signal: controller.signal });
        window.__pageSyncCheck = { pages, controller };
    });
    // 新页首块进入下半屏时仍读旧页，越过可读区域中线后就应翻页。
    await wheelTo(sample.next, 0.64);
    check(Number(await pageNumber.inputValue()) === sample.page, "页首尚在下半屏就提前翻页");
    await wheelTo(sample.next, 0.44);
    await waitPage(sample.page + 1);
    await settle();
    const crossed = await pdf.evaluate(() => {
        const { pages, controller } = window.__pageSyncCheck;
        controller.abort();
        delete window.__pageSyncCheck;
        return pages;
    });
    check(crossed.join(",") === [sample.page, sample.page + 1].join(","), "跨页出现额外跳转 " + crossed);
    const htmlPosition = await html.evaluate(() => scrollY);
    await pdf.locator("#zoom-in").click();
    await settle();
    check(await html.evaluate(() => scrollY) === htmlPosition, "缩放改变了正文阅读位置");
    // 在左侧真正滚回上一页，检查连续滚动也通过工具栏当前页驱动右栏。
    await pdfHost.hover({ position: { x: 120, y: 250 } });
    const distance = await pdf.locator('.page[data-page-number="' + sample.page + '"]').evaluate(n =>
        n.getBoundingClientRect().top - document.querySelector("#viewer-container").getBoundingClientRect().top);
    await page.mouse.wheel(0, Math.round(distance));
    await waitPage(sample.page);
    // 核对右栏阅读线上的真实正文，快速翻页后也要落在最后请求的页。
    const waitFirstBlock = number => html.waitForFunction(pageIdx => {
        const first = [...document.querySelectorAll("[data-docvortex-sources]")].find(n =>
            n.getBoundingClientRect().height && JSON.parse(n.dataset.docvortexSources).some(s => s.page_idx === pageIdx));
        const offset = parseFloat(getComputedStyle(document.body).getPropertyValue("--mineru-preview-offset")) || 24;
        return first && Math.abs(first.getBoundingClientRect().top - offset) < 3;
    }, number - 1);
    await waitFirstBlock(sample.page);
    await pdf.locator("#next").click();
    await pdf.locator("#previous").click();
    await pdf.locator("#next").click();
    await waitPage(sample.page + 1);
    await waitFirstBlock(sample.page + 1);
    await settle();
    check(Number(await pageNumber.inputValue()) === sample.page + 1, "连续翻页发生回跳");
    // 在开关两种状态下激活同一个真实目录项，核对 PDF 最终落到标题框。
    const heading = html.locator("[data-mineru-outline-label]").first();
    const headingId = await heading.evaluate(n => n.closest("[data-docvortex-trace-id]").dataset.docvortexTraceId);
    const link = html.locator('.mineru-outline-link[href="#' + await heading.getAttribute("id") + '"]');
    const toc = html.locator(".mineru-outline-toggle");
    // 目录在窄窗跳转后会自动关闭。
    const navigate = async () => {
        if (await toc.getAttribute("aria-expanded") === "false") await toc.click();
        await link.click();
        await settle();
    };
    await sync.click();
    await navigate();
    check(Number(await pageNumber.inputValue()) === sample.page + 1, "关闭联动后目录移动了 PDF");
    await sync.click();
    await navigate();
    await pdf.waitForFunction(id => [...document.querySelectorAll("button.selected")].some(n => n.traceIds.includes(id)), headingId);
    return { ...sample, columnScrollTop: beforeColumn, crossed, outline: headingId };
}
