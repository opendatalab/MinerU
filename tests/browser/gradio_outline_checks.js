// 使用 playwright-cli run-code --filename 执行，页面须已通过 WebUI 转换带多个章节的文档。
async (page) => {
    // 在正常转换完成的 iframe 上执行真实浏览器交互和布局回归。
    const check = (condition, message) => {
        if (!condition) throw new Error(message);
    };
    const frameElement = page.locator('iframe.mineru-rendered-html-frame');
    const frame = await (await frameElement.elementHandle()).contentFrame();
    const headings = frame.locator('[data-mineru-outline-label]');
    const count = await headings.count();
    check(count > 1, 'need multiple real parsed headings');
    const toggle = frame.locator('.mineru-outline-toggle');
    const pdf = page.frameLocator('iframe.mineru-pdf-frame').locator('input[type=number]').first();
    const pdfPage = await pdf.count() ? await pdf.inputValue() : null;
    // 按稳定标签读取实际标题坐标，覆盖中英文及格式化标题。
    const headingTop = (nodes, label) =>
        nodes.find(node => node.dataset.mineruOutlineLabel === label).getBoundingClientRect().top;
    const results = [];
    for (const width of [360, 600, 1000, 600]) {
        await frameElement.evaluate((node, size) => {
            node.style.width = `${size}px`;
            node.style.maxWidth = 'none';
        }, width);
        await frame.waitForFunction((size) => innerWidth === size, width);
        await page.waitForTimeout(150);
        const initial = await toggle.getAttribute('aria-expanded');
        check(initial === String(width >= 900), `default width=${width} actual=${initial}`);
        if (initial === 'false') await toggle.click();
        const branch = frame.locator('.mineru-outline-branch:visible').first();
        if (await branch.count()) {
            const expanded = await branch.getAttribute('aria-expanded');
            await branch.press('Space');
            check(await branch.getAttribute('aria-expanded') !== expanded, 'keyboard collapse');
            await branch.press('Space');
        }
        const visibleLinks = frame.locator('.mineru-outline-link:visible');
        const link = visibleLinks.nth(Math.min(2, await visibleLinks.count() - 1));
        const label = await link.textContent();
        await link.click();
        await page.waitForTimeout(150);
        const position = await headings.evaluateAll(headingTop, label);
        check(Math.abs(position - 52) < 3, `heading offset ${position} at ${width}`);
        check(await frame.locator('.mineru-outline-link[aria-current]').textContent() === label, 'active heading mismatch');
        check(await toggle.getAttribute('aria-expanded') === String(width >= 900), 'close after navigation');
        // 模拟图片或公式占位变高，验证显式跳转的布局校正。
        await frame.evaluate(() => {
            const spacer = document.createElement('div');
            spacer.dataset.outlineTest = '';
            spacer.style.height = '143px';
            document.querySelector('article').prepend(spacer);
        });
        await page.waitForTimeout(150);
        const shifted = await headings.evaluateAll(headingTop, label);
        check(Math.abs(shifted - 52) < 3, `async layout offset ${shifted}`);
        if (width < 900) {
            await toggle.click();
            await toggle.press('Escape');
            check(await toggle.getAttribute('aria-expanded') === 'false', 'Escape closes');
            await toggle.click();
            // 避开 WebKit 的原生滚动条，点击真实可见的遮罩区域。
            await frame.locator('.mineru-outline-backdrop').click({ position: { x: width - 24, y: 150 } });
            check(await toggle.getAttribute('aria-expanded') === 'false', 'outside click closes');
        }
        results.push({ width, count, label, position, shifted });
    }
    if (pdfPage !== null) check(await pdf.inputValue() === pdfPage, 'TOC moved PDF');
    await frame.evaluate(() => {
        dispatchEvent(new WheelEvent('wheel'));
        scrollTo(0, document.documentElement.scrollHeight);
    });
    await page.waitForTimeout(150);
    const lastLabel = await headings.last().getAttribute('data-mineru-outline-label');
    check(await frame.locator('.mineru-outline-link[aria-current]').textContent() === lastLabel, 'last chapter highlight');
    await frame.evaluate(() => {
        document.querySelectorAll('[data-outline-test]').forEach(node => node.remove());
    });
    await frameElement.evaluate(node => {
        node.style.width = '';
        node.style.maxWidth = '';
    });
    return results;
}
