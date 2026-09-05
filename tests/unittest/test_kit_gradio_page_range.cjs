const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const reduce = vm.runInThisContext(fs.readFileSync(path.join(__dirname, "../../mineru/resources/gradio_page_range.js"), "utf8"));
const tiers = ["flash", "basic", "standard", "advanced"];
// 使用 Python 测试传入的统一格式集合，测试和产品代码均不维护另一份后缀清单。
const flashOnlyExtensions = JSON.parse(process.argv[2]);
const tieredExtensions = JSON.parse(process.argv[3]);

// 模拟原生 Gradio 组件值与纯前端输出更新，不复制实际联动算法。
function createUi(limit = 20, availableTiers = tiers) {
    const defaultTier = availableTiers.includes("standard") ? "standard" : availableTiers.at(-1);
    const state = {
        file: null, position: availableTiers.indexOf(defaultTier), metadata: {}, previous: {}, a: 1, b: 1,
        tierSelection: { tier: defaultTier, locked: false },
    };
    return {
        state,
        step(action, patch = {}) {
            // 应用本次输入事件，并保留 Gradio update 的跳过语义。
            Object.assign(state, patch);
            if (action === "clear") state.file = null;
            const result = reduce(
                availableTiers, flashOnlyExtensions, limit, state.file, state.position,
                JSON.stringify(state.metadata), JSON.stringify(state.previous), state.a, state.b,
                JSON.stringify(state.tierSelection)
            );
            if ("value" in result[0]) state.a = result[0].value;
            if ("value" in result[1]) state.b = result[1].value;
            if (result[4].__type__ !== "update") state.previous = JSON.parse(result[4]);
            if ("value" in result[7]) state.position = result[7].value;
            state.tierSelection = JSON.parse(result[9]);
            return result;
        },
        upload(count, filename = "/report.pdf") {
            // 上传和元数据完成分两次触发，以检查未完成阶段的提交限制。
            this.step("file", { file: { path: filename } });
            return this.step("metadata", { metadata: { path: filename, page_count: count, error: "" } });
        },
    };
}

// 所有轻量格式（含大写后缀）锁定 Flash；连续上传和残留输入不能覆盖原先选择。
for (const extension of flashOnlyExtensions.flatMap((extension) => [extension, extension.toUpperCase()])) {
    const ui = createUi();
    ui.step("tier", { position: 3 });
    const out = ui.step("file", { file: { path: `/source.${extension}` } });
    assert.equal(out[7].value, 0);
    assert.equal(out[7].interactive, false);
    assert.equal(out[8], "解析 tier：flash");
    assert.equal(out[5].interactive, true);
    assert.equal(out[3], "");
    assert.match(out[2], /data-range-visible="false"/);
    assert.deepEqual(ui.state.tierSelection, { tier: "advanced", locked: true });
    // 即使模拟一个已经排队的高档位 input，也不能改变锁定值或恢复偏好。
    assert.equal(ui.step("tier", { position: 1 })[7].value, 0);
    ui.step("file", { file: `/another.${extension}` });
    assert.equal(ui.state.tierSelection.tier, "advanced");
    const cleared = ui.step("clear");
    assert.equal(cleared[7].value, 3);
    assert.equal(cleared[7].interactive, true);
    assert.equal(cleared[8], "解析 tier：advanced");
    assert.equal(cleared[5].interactive, false);
}

// 切回任意 PDF/图片后在同一次事件恢复档位，PDF 等待页数、图片可立即转换。
for (const extension of tieredExtensions.flatMap((extension) => [extension, extension.toUpperCase()])) {
    const ui = createUi();
    ui.step("file", { file: "/source.csv" });
    const out = ui.step("file", { file: { path: `/source.${extension}` } });
    assert.equal(out[7].value, 2);
    assert.equal(out[7].interactive, true);
    assert.equal(out[8], "解析 tier：standard");
    assert.equal(out[5].interactive, extension.toLowerCase() !== "pdf");
    assert.deepEqual(ui.state.tierSelection, { tier: "standard", locked: false });
}

// 缺少 Flash 不伪造滑杆位置，始终阻止提交；恢复可分档文件后沿用原先能力和选择。
for (const availableTiers of [["basic", "standard", "advanced"], ["advanced"], ["flash"], ["flash", "advanced"]]) {
    const ui = createUi(20, availableTiers);
    const originalPosition = ui.state.position;
    const available = availableTiers.includes("flash");
    const out = ui.step("file", { file: "/source.csv" });
    assert.equal(out[7].interactive, false);
    assert.equal(out[7].value, available ? availableTiers.indexOf("flash") : originalPosition);
    assert.equal(out[5].interactive, available);
    assert.equal(out[6].visible, !available);
    if (!available) {
        assert.equal(out[6].value, "该格式仅支持 Flash，当前服务不可用");
        assert.match(out[8], /flash.*当前服务不可用/);
    }
    const restored = ui.step("file", { file: "/source.png" });
    assert.equal(restored[7].value, originalPosition);
    assert.equal(restored[7].interactive, availableTiers.length > 1);
    assert.equal(restored[5].interactive, true);
    assert.equal(restored[6].visible, false);
}

// 文件连续切换与旧 PDF 元数据晚到时，锁定状态和偏好只跟随当前文件。
{
    const ui = createUi();
    ui.upload(100);
    ui.step("tier", { position: 3 });
    ui.step("file", { file: "/source.csv" });
    ui.step("metadata", { metadata: { path: "/report.pdf", page_count: 100, error: "" } });
    assert.equal(ui.state.position, 0);
    assert.equal(ui.state.tierSelection.tier, "advanced");
    let out = ui.step("file", { file: "/new.pdf" });
    assert.equal(out[7].value, 3);
    assert.equal(out[5].interactive, false);
    assert.match(out[6].value, /正在读取/);
    out = ui.step("metadata", { metadata: { path: "/report.pdf", page_count: 100, error: "" } });
    assert.equal(out[5].interactive, false);
    out = ui.step("metadata", { metadata: { path: "/new.pdf", page_count: 12, error: "" } });
    assert.equal(out[5].interactive, true);
    assert.equal(out[3], "1-12");
    ui.step("tier", { position: 1 });
    ui.step("file", { file: "/second.csv" });
    assert.equal(ui.step("clear")[7].value, 1);
    // 会话互不共享偏好。
    assert.equal(createUi().state.tierSelection.tier, "standard");
}

// 同时检查组件身份、语义角色、提交范围及无障碍标签，不能只验证排序后的数字。
function assertSelection(result, a, b, startHandle) {
    const state = JSON.parse(result[4]);
    const start = Math.min(a, b);
    const end = Math.max(a, b);
    assert.equal(result[0].value, a);
    assert.equal(result[1].value, b);
    assert.equal(state.handle_a, a);
    assert.equal(state.handle_b, b);
    assert.equal(state.start_handle, startHandle);
    assert.equal(result[0].label, startHandle === "a" ? "起始页" : "结束页");
    assert.equal(result[1].label, startHandle === "b" ? "起始页" : "结束页");
    assert.equal(result[3], start === end ? String(start) : `${start}-${end}`);
    assert(result[2].includes(`[${start}-${end}] · ${end - start + 1} 页`));
    assert(result[2].includes(`data-start-handle="${startHandle}"`));
}

let ui = createUi();
assert.match(ui.step("file")[2], /data-range-visible="false"/);
let out = ui.step("file", { file: { path: "/report.pdf" } });
assert.equal(out[5].interactive, false);
assert.match(out[6].value, /正在读取/);
out = ui.upload(100);
assert.equal(out[0].minimum, 1);
assert.equal(out[1].maximum, 100);
assert.equal(out[3], "1-20");
for (const [action, value, expected] of [
    ["b", 40, "21-40"], ["a", 15, "15-34"], ["a", 20, "20-34"], ["b", 35, "20-35"]
]) {
    assert.equal(ui.step(action, { [action]: value })[3], expected);
}

// 原左滑块保持 A 身份连续穿越：20→40→60→30，超限时推动 B 而不是交换物理值。
assertSelection(ui.step("a", { a: 40 }), 40, 35, "b");
assertSelection(ui.step("a", { a: 60 }), 60, 41, "b");
const crossedState = { ...ui.state.previous };
assert.equal(ui.step("tier", { position: 0 })[3], "");
assert.match(ui.step("tier", { position: 0 })[2], /data-range-visible="false"/);
assert.deepEqual(ui.state.previous, crossedState);
assertSelection(ui.step("tier", { position: 3 }), 60, 41, "b");
assertSelection(ui.step("a", { a: 30 }), 30, 41, "a");

// 两侧接近相等时保留旧角色，只有严格越过才交换；反复穿越也不改变组件身份。
assertSelection(ui.step("a", { a: 41 }), 41, 41, "a");
assertSelection(ui.step("a", { a: 42 }), 42, 41, "b");
assertSelection(ui.step("a", { a: 41 }), 41, 41, "b");
assertSelection(ui.step("a", { a: 40 }), 40, 41, "a");
assertSelection(ui.step("b", { b: 40 }), 40, 40, "a");
assertSelection(ui.step("b", { b: 39 }), 40, 39, "b");
assertSelection(ui.step("b", { b: 40 }), 40, 40, "b");

// 原右滑块保持 B 身份连续穿越：55→30→10，左边的 B 超限时把 A 向左推。
ui = createUi();
ui.upload(100);
ui.step("b", { b: 55 });
assertSelection(ui.step("a", { a: 40 }), 40, 55, "a");
assertSelection(ui.step("b", { b: 30 }), 40, 30, "b");
assertSelection(ui.step("b", { b: 10 }), 29, 10, "b");
assertSelection(ui.step("a", { a: -500 }), 1, 10, "a");
assertSelection(ui.step("b", { b: 500 }), 81, 100, "a");

// 换文件后旧页数响应不能覆盖新状态，已缓存页数不受晚到响应污染。
ui.step("file", { file: { path: "/new.pdf" } });
out = ui.step("metadata", { metadata: { path: "/report.pdf", page_count: 100 } });
assert.equal(out[5].interactive, false);
assert.equal(JSON.parse(out[4]).page_count, 0);
out = ui.upload(12, "/new.pdf");
assert.equal(out[3], "1-12");
assertSelection(out, 1, 12, "a");
ui.step("metadata", { metadata: { path: "/report.pdf", page_count: 100 } });
assert.equal(ui.step("tier", { position: 2 })[3], "1-12");
out = ui.step("file", { file: { path: "/sample.csv" } });
assert.match(out[2], /data-range-visible="false"/);
assert.equal(out[3], "");
assert.equal(out[5].interactive, true);
out = ui.step("clear");
assert.equal(out[5].interactive, false);
assert.equal(JSON.parse(out[4]).path, "");
assert.equal(JSON.parse(out[4]).start_handle, "a");

// 不限页数时跨越不能移动对端；上限为 1 时两个实体一起移动并保留重合角色。
ui = createUi(null);
ui.upload(100);
ui.step("b", { b: 35 });
assertSelection(ui.step("a", { a: 100 }), 100, 35, "b");
assertSelection(ui.step("b", { b: 1 }), 100, 1, "b");
ui = createUi(1);
assertSelection(ui.upload(100), 1, 1, "a");
assertSelection(ui.step("a", { a: 50 }), 50, 50, "a");
assertSelection(ui.step("b", { b: 10 }), 10, 10, "a");

// 文件错误只阻止非 Flash；错误消息须转义，不能成为 HTML 注入入口。
ui = createUi();
ui.step("file", { file: { path: "/bad.pdf" } });
out = ui.step("metadata", { metadata: { path: "/bad.pdf", page_count: 0, error: "<bad>" } });
assert.equal(out[5].interactive, false);
assert.equal(out[6].value, "&lt;bad&gt;");
out = ui.step("tier", { position: 0 });
assert.equal(out[5].interactive, true);
assert.equal(out[6].visible, false);

// 确定性遍历总页数和上限：物理值允许倒序，排序后的范围、角色与上限必须一致。
for (const count of [1, 12, 20, 100]) {
    for (const limit of [null, 1, 20, 250]) {
        ui = createUi(limit);
        out = ui.upload(count);
        assert.equal(ui.state.b, Math.min(count, limit ?? count));
        assert.equal(out[1].interactive, count > 1);
        let seed = 7;
        for (let step = 0; step < 1000; step++) {
            seed = (seed * 16807) % 2147483647;
            const action = step % 2 ? "a" : "b";
            const value = seed % (count + 40) - 20;
            const other = action === "a" ? "b" : "a";
            const previousOther = ui.state[other];
            const previousRole = ui.state.previous.start_handle;
            const wanted = Math.min(count, Math.max(1, value));
            out = ui.step(action, { [action]: value });
            const { a, b } = ui.state;
            assert(a >= 1 && a <= count && b >= 1 && b <= count);
            assert.equal(ui.state[action], wanted);
            assert(limit === null || Math.abs(a - b) + 1 <= limit);
            if (limit === null || Math.abs(wanted - previousOther) + 1 <= limit) {
                assert.equal(ui.state[other], previousOther);
            }
            assertSelection(out, a, b, a < b ? "a" : b < a ? "b" : previousRole);
        }
    }
}
console.log("Page range frontend cases passed (including 16000 boundary transitions).");
