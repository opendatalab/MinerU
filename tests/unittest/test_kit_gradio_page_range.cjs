const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
// 使用产品词典和真实语言解析器；默认中文以保留既有行为断言。
Object.defineProperty(globalThis, "navigator", { value: { languages: ["zh-CN"] }, configurable: true });
global.window = { __mineruI18n: vm.runInThisContext(fs.readFileSync(
    path.join(__dirname, "../../mineru/resources/gradio_i18n.js"), "utf8"
))(JSON.parse(fs.readFileSync(0, "utf8"))) };
const modelSource = fs.readFileSync(path.join(__dirname, "../../mineru/resources/gradio_page_range_model.js"), "utf8");
const model = vm.runInThisContext(modelSource)();
const reduce = vm.runInThisContext(fs.readFileSync(path.join(__dirname, "../../mineru/resources/gradio_page_range.js"), "utf8")
    .replace("__MINERU_PAGE_RANGE_MODEL__", modelSource));
const tiers = ["flash", "basic", "standard", "advanced"];
// 使用 Python 测试传入的统一格式集合，测试和产品代码均不维护另一份后缀清单。
const flashOnlyExtensions = JSON.parse(process.argv[2]);
const tieredExtensions = JSON.parse(process.argv[3]);

// 模拟原生 Gradio 组件值与纯前端输出更新，不复制实际联动算法。
function createUi(limit = 20, availableTiers = tiers) {
    const defaultTier = availableTiers.includes("standard") ? "standard" : availableTiers.at(-1);
    const state = {
        file: null, position: availableTiers.indexOf(defaultTier), metadata: {}, previous: {}, a: 1, b: 1, start: 1, end: 1,
        tierSelection: { tier: defaultTier, locked: false },
    };
    return {
        state,
        step(action, patch = {}) {
            // 应用本次输入事件，并保留 Gradio update 的跳过语义。
            if (action === "a" || action === "b") {
                const [a, b] = model.move(state.a, state.b, action, patch[action], state.previous.page_count, limit);
                patch = { ...patch, a, b };
            }
            Object.assign(state, patch);
            if (action === "clear") state.file = null;
            const result = reduce(
                availableTiers, flashOnlyExtensions, limit, state.file, state.position,
                JSON.stringify(state.metadata), JSON.stringify(state.previous),
                JSON.stringify({ ...state.previous, handle_a: state.a, handle_b: state.b }),
                JSON.stringify(state.tierSelection), (typeof state.file === "string" ? state.file : state.file?.path) || "",
                state.start, state.end, action === "preset"
            );
            state.a = JSON.parse(result[0].value).handle_a;
            state.b = JSON.parse(result[0].value).handle_b;
            if (result[3].__type__ !== "update") state.previous = JSON.parse(result[3]);
            if ("value" in result[6]) state.position = result[6].value;
            state.tierSelection = JSON.parse(result[8]);
            state.start = result[9].value;
            state.end = result[10].value;
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
    assert.equal(out[6].value, 3);
    assert.equal(out[7].props.locked, true);
    assert.equal(out[6].visible, undefined);
    assert.equal(out[7].props.selected_tier, "flash");
    assert.equal(out[4].interactive, true);
    assert.equal(out[2], "");
    assert.match(out[1], /data-range-visible="false"/);
    assert.deepEqual(ui.state.tierSelection, { tier: "advanced", locked: true });
    // 即使模拟一个已经排队的高档位 input，也不能改变锁定值或恢复偏好。
    assert.equal(ui.step("tier", { position: 1 })[6].value, 3);
    ui.step("file", { file: `/another.${extension}` });
    assert.equal(ui.state.tierSelection.tier, "advanced");
    const cleared = ui.step("clear");
    assert.equal(cleared[6].value, 3);
    assert.equal(cleared[7].props.locked, false);
    assert.equal(cleared[7].props.selected_tier, "advanced");
    assert.equal(cleared[4].interactive, false);
}

// 切回任意 PDF/图片后在同一次事件恢复档位，PDF 等待页数、图片可立即转换。
for (const extension of tieredExtensions.flatMap((extension) => [extension, extension.toUpperCase()])) {
    const ui = createUi();
    ui.step("file", { file: "/source.csv" });
    const out = ui.step("file", { file: { path: `/source.${extension}` } });
    assert.equal(out[6].value, 2);
    assert.equal(out[7].props.locked, false);
    assert.equal(out[7].props.selected_tier, "standard");
    assert.equal(out[4].interactive, extension.toLowerCase() !== "pdf");
    assert.deepEqual(ui.state.tierSelection, { tier: "standard", locked: false });
}

// 原生文档独立于 PDF 档位集合，隐藏滑杆但允许提交；切回后恢复合法选择。
for (const availableTiers of [["standard"], ["basic", "standard", "advanced"], ["advanced"], ["flash"], ["flash", "advanced"]]) {
    const ui = createUi(20, availableTiers);
    const originalPosition = ui.state.position;
    const out = ui.step("file", { file: "/source.csv" });
    assert.equal(out[7].props.locked, true);
    assert.equal(out[6].value, originalPosition);
    assert.equal(out[6].visible, undefined);
    assert.equal(out[4].interactive, true);
    assert.equal(out[5].visible, false);
    assert.equal(out[7].props.selected_tier, "flash");
    const restored = ui.step("file", { file: "/source.png" });
    assert.equal(restored[6].value, originalPosition);
    assert.equal(restored[7].props.locked, false);
    assert.equal(restored[6].visible, undefined);
    assert.equal(restored[4].interactive, true);
    assert.equal(restored[5].visible, false);
}

// 文件连续切换与旧 PDF 元数据晚到时，锁定状态和偏好只跟随当前文件。
{
    const ui = createUi();
    ui.upload(100);
    ui.step("tier", { position: 3 });
    ui.step("file", { file: "/source.csv" });
    ui.step("metadata", { metadata: { path: "/report.pdf", page_count: 100, error: "" } });
    assert.equal(ui.state.position, 3);
    assert.equal(ui.state.tierSelection.tier, "advanced");
    let out = ui.step("file", { file: "/new.pdf" });
    assert.equal(out[6].value, 3);
    assert.equal(out[4].interactive, false);
    assert.match(out[1], /data-range-visible="true"/);
    assert.match(out[1], /正在读取/);
    assert.equal(out[2], "");
    assert.equal(out[5].visible, false);
    out = ui.step("metadata", { metadata: { path: "/report.pdf", page_count: 100, error: "" } });
    assert.equal(out[4].interactive, false);
    out = ui.step("metadata", { metadata: { path: "/new.pdf", page_count: 12, error: "" } });
    assert.equal(out[4].interactive, true);
    assert.equal(out[2], "1-12");
    ui.step("tier", { position: 1 });
    ui.step("file", { file: "/second.csv" });
    assert.equal(ui.step("clear")[6].value, 1);
    // 会话互不共享偏好。
    assert.equal(createUi().state.tierSelection.tier, "standard");
}

// 同时检查组件身份、语义角色、提交范围及无障碍标签，不能只验证排序后的数字。
function assertSelection(result, a, b, startHandle) {
    const state = JSON.parse(result[3]);
    const start = Math.min(a, b);
    const end = Math.max(a, b);
    assert.equal(JSON.parse(result[0].value).handle_a, a);
    assert.equal(JSON.parse(result[0].value).handle_b, b);
    assert.equal(state.handle_a, a);
    assert.equal(state.handle_b, b);
    assert.equal(state.start_handle, startHandle);
    assert.equal(JSON.parse(result[0].value).start_handle, startHandle);
    assert.equal(result[2], start === end ? String(start) : `${start}-${end}`);
    assert(result[1].includes(`>${end - start + 1}</span> / ${state.page_count} 页`));
    assert(result[1].includes(start === end ? `第 ${start} 页` : `第 ${start}–${end} 页`));
    assert(result[1].includes(`data-start-handle="${startHandle}"`));
    assert.equal(result[9].value, start);
    assert.equal(result[10].value, end);
}

let ui = createUi();
assert.match(ui.step("file")[1], /data-range-visible="false"/);
let out = ui.step("file", { file: { path: "/report.pdf" } });
assert.equal(out[4].interactive, false);
assert.match(out[1], /data-range-visible="true"/);
assert.match(out[1], /正在读取/);
assert.equal(out[5].visible, false);
assert.equal(JSON.parse(out[0].value).page_count, 0);
out = ui.upload(100);
assert.equal(JSON.parse(out[0].value).page_count, 100);
assert.equal(out[2], "1-20");
for (const [action, value, expected] of [
    ["b", 40, "21-40"], ["a", 15, "15-34"], ["a", 20, "20-34"], ["b", 35, "20-35"]
]) {
    assert.equal(ui.step(action, { [action]: value })[2], expected);
}

// 原左滑块保持 A 身份连续穿越：20→40→60→30，超限时推动 B 而不是交换物理值。
assertSelection(ui.step("a", { a: 40 }), 40, 35, "b");
assertSelection(ui.step("a", { a: 60 }), 60, 41, "b");
const crossedState = { ...ui.state.previous };
// Flash 同样显示滑块，选区与角色原样保留，切回高档位不重置。
assertSelection(ui.step("tier", { position: 0 }), 60, 41, "b");
assert.match(ui.step("tier", { position: 0 })[1], /data-range-visible="true"/);
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
assert.equal(out[4].interactive, false);
assert.equal(JSON.parse(out[3]).page_count, 0);
out = ui.upload(12, "/new.pdf");
assert.equal(out[2], "1-12");
assertSelection(out, 1, 12, "a");
ui.step("metadata", { metadata: { path: "/report.pdf", page_count: 100 } });
assert.equal(ui.step("tier", { position: 2 })[2], "1-12");
out = ui.step("file", { file: { path: "/sample.csv" } });
assert.match(out[1], /data-range-visible="false"/);
assert.equal(out[2], "");
assert.equal(out[4].interactive, true);
out = ui.step("clear");
assert.equal(out[4].interactive, false);
assert.equal(JSON.parse(out[3]).path, "");
assert.equal(JSON.parse(out[3]).start_handle, "a");

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

// 文件错误阻止所有档位提交；错误消息须转义，不能成为 HTML 注入入口。
ui = createUi();
ui.step("file", { file: { path: "/bad.pdf" } });
out = ui.step("metadata", { metadata: { path: "/bad.pdf", page_count: 0, error: "<bad>" } });
assert.equal(out[4].interactive, false);
assert.match(out[1], /无法读取页数/);
assert.equal(out[5].value, "&lt;bad&gt;");
out = ui.step("tier", { position: 0 });
assert.equal(out[4].interactive, false);
assert.equal(out[5].visible, true);
assert.equal(out[5].value, "&lt;bad&gt;");

// 精确输入与滑块共用物理端点；快捷选择一次恢复两端，包含已交叉和单页上限。
ui = createUi();
out = ui.upload(48);
assert.equal(out[11].value, "前 20 页");
assertSelection(ui.step("end", { end: 30 }), 11, 30, "a");
assertSelection(ui.step("start", { start: 5 }), 5, 24, "a");
assertSelection(ui.step("start", { start: 10 }), 10, 24, "a");
assertSelection(ui.step("start", { start: 40 }), 40, 24, "b");
assertSelection(ui.step("end", { end: 48 }), 48, 29, "b");
assertSelection(ui.step("start", { start: null }), 48, 29, "b");
assertSelection(ui.step("end", { end: NaN }), 48, 29, "b");
assertSelection(ui.step("preset"), 1, 20, "a");
assertSelection(ui.step("end", { end: 999 }), 29, 48, "a");
assertSelection(ui.step("start", { start: 0 }), 1, 20, "a");
ui = createUi(null);
assert.equal(ui.upload(48)[11].value, "全部");
ui.step("end", { end: 20 });
assertSelection(ui.step("preset"), 1, 48, "a");
ui = createUi(20);
out = ui.upload(12);
assert.equal(out[11].value, "全部");
out = ui.upload(1, "/single.pdf");
assert.equal(out[9].interactive, false);
assert.equal(out[10].maximum, 1);
assert.equal(out[11].interactive, false);
ui = createUi(1);
ui.upload(48);
assertSelection(ui.step("end", { end: 40 }), 40, 40, "a");
assertSelection(ui.step("preset"), 1, 1, "a");

// 一次提交的两端必须同时生效，保持平移长度和交叉后的物理身份。
ui = createUi();
ui.upload(48);
assertSelection(ui.step("pair", { a: 21, b: 40 }), 21, 40, "a");
assertSelection(ui.step("pair", { a: 48, b: 29 }), 48, 29, "b");
assertSelection(ui.step("pair", { a: 25, b: 6 }), 25, 6, "b");
// 外部异常端点仍须经过边界和上限校验。
assertSelection(ui.step("pair", { a: 999, b: -100 }), 48, 29, "b");

// 确定性遍历总页数和上限：物理值允许倒序，排序后的范围、角色与上限必须一致。
for (const count of [1, 12, 20, 100]) {
    for (const limit of [null, 1, 20, 250]) {
        ui = createUi(limit);
        out = ui.upload(count);
        assert.equal(ui.state.b, Math.min(count, limit ?? count));
        assert.equal(JSON.parse(out[0].value).page_count, count);
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
