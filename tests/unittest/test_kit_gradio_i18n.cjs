const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const messages = JSON.parse(fs.readFileSync(0, "utf8"));
// 加载实际产品脚本，语言用独立浏览器环境提供，不复制翻译或状态算法。
const resource = (name) => fs.readFileSync(path.join(__dirname, "../../mineru/resources/", name), "utf8");
for (const [languages, language, expected] of [
    [["zh-CN"], "zh-CN", "zh"], [["zh-TW"], "zh-TW", "zh"], [["zh-HK"], "zh-HK", "zh"],
    [["en-US"], "en-US", "en"], [["ja-JP", "zh-CN"], "ja-JP", "en"], [[], "", "en"], [[], "zh-CN", "zh"],
]) {
    const context = vm.createContext({ navigator: { languages, language }, window: {} });
    const i18n = vm.runInContext(resource("gradio_i18n.js"), context)(messages);
    context.window.__mineruI18n = i18n;
    assert.equal(i18n.resolveLocale(), expected);
    assert.equal(i18n.text("force_ocr_info"), messages.force_ocr_info[expected === "zh" ? 1 : 0]);
    assert.ok(i18n.text("failed", { error: "$& 文件.pdf" }).includes("$& 文件.pdf"));
    const range = vm.runInContext(resource("gradio_page_range.js"), context);
    const out = range(["flash", "standard"], ["csv"], 20, "/test.pdf", 1,
        JSON.stringify({ path: "/test.pdf", page_count: 10 }), "{}", 1, 1,
        JSON.stringify({ tier: "standard", locked: false }));
    assert.equal(out[0].label, i18n.text("start_page"));
    assert.equal(out[3], "1-10");
    assert.ok(out[2].includes(i18n.text("page_limit", { count: 20 })));
    assert.equal(out[8], i18n.text("tier_value", { tier: i18n.text("tier_standard"), notice: "" }));
    assert.ok(!out[2].includes("<strong>"));
    assert.ok(!out[2].includes(i18n.text("start_page")));
    assert.ok(!out[2].includes(i18n.text("end_page")));
    for (const [position, tier] of ["flash", "basic", "standard", "advanced"].entries()) {
        const labels = expected === "zh"
            ? ["解析等级：flash（极速）", "解析等级：basic（中）", "解析等级：standard（高）", "解析等级：advanced（极高）"]
            : ["Parsing tier: flash", "Parsing tier: basic (medium)", "Parsing tier: standard (high)", "Parsing tier: advanced (xhigh)"];
        const result = range(["flash", "basic", "standard", "advanced"], ["csv"], null, null, position,
            "{}", "{}", 1, 1, JSON.stringify({ tier, locked: false }));
        assert.equal(result[8], labels[position]);
    }
    const download = vm.runInContext(resource("gradio_download.js"), context);
    const formats = [["html", "HTML"]];
    download("activate", formats, "", "", "run");
    const [token, busy] = download("begin", formats, "html", "HTML", "run");
    assert.ok(busy.value.includes(i18n.text("preparing_download")));
    const [, failure] = download("complete", formats, "html", "HTML", null,
        JSON.stringify({ request: token, error: '<script>external error</script>' }), "run");
    assert.ok(failure.includes("&lt;script&gt;external error&lt;/script&gt;"));
    assert.ok(!failure.includes("<script>"));
    assert.ok(failure.includes(expected === "zh" ? "下载失败" : "download failed"));
    assert.equal(download("reset", formats)[4].value, "HTML");
}
