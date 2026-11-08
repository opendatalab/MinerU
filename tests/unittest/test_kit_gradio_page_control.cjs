// 执行真实手势控制器，检查提交次数、捕获生命周期和连续位置，不复制生产算法。
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
// 从资源文件加载同一份控制器及共享约束。
const resource = name => fs.readFileSync(path.join(__dirname, "../../mineru/resources", name), "utf8");
const script = resource("gradio_page_control.js").replace("__MINERU_PAGE_RANGE_MODEL__", resource("gradio_page_range_model.js"));

// 最小 DOM 仅记录浏览器 API 的输入输出，事件回调由生产控制器注册。
function node() {
    return {
        dataset: {}, events: {}, attrs: {}, value: "1", disabled: false,
        style: { values: {},
            // 记录连续位置，检查拖动时不会提前吸附整数。
            setProperty(key, value) { this.values[key] = value; } },
        // 保留无障碍名称以验证交叉后的语义。
        setAttribute(key, value) { this.attrs[key] = value; },
        // 读取旧名称，确保未变化时不产生多余 DOM 属性更新。
        getAttribute(key) { return this.attrs[key]; },
        // 同一元素只为每种测试事件安装一个控制器回调。
        addEventListener(name, callback) { this.events[name] = callback; },
        // 轨道有效宽度 470px，每页正好对应 10px。
        getBoundingClientRect() { return { left: 0, width: 498 }; },
        // 记录并检查捕获归属，不模拟原生拖动算法。
        setPointerCapture(id) { this.capture = id; },
        // 只允许当前捕获指针释放控制权。
        hasPointerCapture(id) { return this.capture === id; },
        // 记录已释放，供取消和换文件断言。
        releasePointerCapture() { this.capture = null; },
        // 测试焦点不触发额外的 Gradio 提交。
        focus() {},
    };
}

// 每个实例保存独立 props、回执和输入节点，便于重放文件切换竞态。
function setup(a = 1, b = 20, count = 48, limit = 20) {
    const rail = node();
    const inputs = [node(), node()];
    inputs.forEach(input => { input.parentElement = node(); });
    const leaves = Object.fromEntries([".mineru-page-start input", ".mineru-page-end input", ".mineru-page-selected-count",
        ".mineru-page-interval"].map(key => [key, node()]));
    const props = { value: JSON.stringify({ path: "/test.pdf", page_count: count, handle_a: a, handle_b: b,
        start_handle: a <= b ? "a" : "b" }) };
    const submissions = [];
    const window = { events: {}, __mineruI18n: { text: (key, values) => key + JSON.stringify(values || {}) },
        // 窗口失焦必须恢复正式值。
        addEventListener(name, callback) { this.events[name] = callback; } };
    let watcher;
    vm.runInNewContext(`(${script})`, { window, document: {} })({
        element: {
            // 数字和摘要位于选页组件共同的父容器。
            closest() { return { querySelector: key => leaves[key] }; },
            querySelector: selector => selector === ".mineru-page-sliders" ? rail
                : inputs[selector.includes("-a ") ? 0 : 1],
        },
        props, maxPages: limit,
        trigger: event => submissions.push({ event, state: JSON.parse(props.value) }),
        watch: (_keys, callback) => { watcher = callback; },
    });
    return { rail, inputs, props, submissions, leaves, window,
        // 按页坐标重放真实指针事件，可额外指定取消与多指属性。
        pointer(type, page, patch = {}) {
            rail.events[type]({ button: 0, pointerId: 7, isPrimary: true, clientX: 14 + (page - 1) * 10,
                preventDefault() {}, ...patch });
        },
        // 输入回执沿用同一实例，确认相同状态不会打断拖动。
        update(patch) { props.value = JSON.stringify({ ...JSON.parse(props.value), ...patch }); watcher(); },
        // 原生键盘入口沿用相同的边界与提交逻辑。
        key(index, key) { inputs[index].events.keydown({ key, preventDefault() {} }); },
        // 返回当前可见整数值，连续位置另由 CSS 变量验证。
        values() { return inputs.map(input => Number(input.value)); },
    };
}

let ui = setup();
ui.pointer("pointerdown", 20);
ui.pointer("pointermove", 30.4);
assert.deepEqual(ui.values(), [11, 30]);
assert.equal(ui.submissions.length, 0);
assert.equal(JSON.parse(ui.props.value).handle_b, 20);
assert(Math.abs(Number(ui.inputs[1].parentElement.style.values["--page-position"]) - 29.4 / 47) < 1e-10);
ui.update({});
assert.deepEqual(ui.values(), [11, 30]);
ui.pointer("pointerup", 30.4);
assert.equal(ui.submissions.length, 1);
assert.deepEqual([ui.submissions[0].state.handle_a, ui.submissions[0].state.handle_b], [11, 30]);
assert.equal(ui.rail.capture, null);

// 中段原子平移，正反方向在边界处保持 20 页；越过边界后回移仍以按下点为基准。
ui.pointer("pointerdown", 20);
assert.equal(ui.rail.dataset.active, "pan");
ui.pointer("pointermove", 30);
assert.deepEqual(ui.values(), [21, 40]);
ui.pointer("pointermove", 90);
assert.deepEqual(ui.values(), [29, 48]);
ui.pointer("pointermove", -90);
assert.deepEqual(ui.values(), [1, 20]);
ui.pointer("pointerup", 30);
assert.deepEqual(ui.values(), [21, 40]);
assert.equal(ui.submissions.length, 2);
ui = setup(30, 11);
ui.pointer("pointerdown", 20);
ui.pointer("pointerup", 30);
assert.deepEqual(ui.values(), [40, 21]);
assert.equal(ui.submissions[0].state.start_handle, "b");

// 半页附近的浮点舍入误差不能让平移临时多选一页。
ui = setup();
ui.pointer("pointerdown", 10);
ui.pointer("pointermove", 10.5 - Number.EPSILON * 4);
assert.equal(ui.values()[1] - ui.values()[0] + 1, 20);
ui.pointer("pointerup", 10.5 - Number.EPSILON * 4);
const snapped = JSON.parse(ui.props.value);
assert.equal(snapped.handle_b - snapped.handle_a + 1, 20);

// 透明热区按下不跳动；交叉后继续控制同一个物理端点。
ui = setup();
ui.pointer("pointerdown", 2.8);
assert.deepEqual(ui.values(), [1, 20]);
ui.pointer("pointermove", 31.8);
assert.deepEqual(ui.values(), [30, 20]);
ui.pointer("pointerup", 41.8);
assert.deepEqual(ui.values(), [40, 21]);
assert.equal(ui.submissions[0].state.start_handle, "b");
assert.equal(ui.inputs[0].attrs["aria-label"], "end_page{}");

// 重合端点向左选起点、向右选终点；不足 2px 的点击不改变范围。
for (const [target, expected] of [[10, [10, 20]], [30, [20, 30]], [20.1, [20, 20]]]) {
    ui = setup(20, 20);
    ui.pointer("pointerdown", 20);
    ui.pointer("pointerup", target);
    assert.deepEqual(ui.values(), expected);
}

// 点击未选中的轨道移动最近端点；松手一次提交，多指不会抢占当前手势。
ui = setup();
ui.pointer("pointerdown", 35);
assert.deepEqual(ui.values(), [16, 35]);
ui.pointer("pointerdown", 48, { pointerId: 8, isPrimary: false });
ui.pointer("pointermove", 48, { pointerId: 8 });
ui.pointer("pointerup", 48, { pointerId: 8 });
assert.deepEqual(ui.values(), [16, 35]);
ui.pointer("pointerup", 35);
assert.equal(ui.submissions.length, 1);

// 取消、失去捕获、Escape 和窗口失焦都回滚本地预览，不产生提交。
for (const reason of ["pointercancel", "lostpointercapture", "Escape", "blur"]) {
    ui = setup();
    ui.pointer("pointerdown", 20);
    ui.pointer("pointermove", 35);
    if (reason === "Escape") ui.key(1, reason);
    else if (reason === "blur") ui.window.events.blur();
    else ui.pointer(reason, 35);
    ui.pointer("pointerup", 35);
    assert.deepEqual(ui.values(), [1, 20]);
    assert.equal(ui.submissions.length, 0);
}

// 换文件会取消旧捕获，迟到的松手不能把旧范围写入新文档。
ui = setup();
ui.pointer("pointerdown", 20);
ui.pointer("pointermove", 35);
ui.update({ path: "/new.pdf", page_count: 12, handle_a: 1, handle_b: 12 });
ui.pointer("pointerup", 35);
assert.deepEqual(ui.values(), [1, 12]);
assert.equal(ui.submissions.length, 0);
ui.key(1, "ArrowLeft");
assert.deepEqual(ui.values(), [1, 11]);
assert.equal(ui.submissions.length, 1);
ui.key(1, "Home");
assert.deepEqual(ui.values(), [1, 1]);
ui.key(0, "End");
assert.deepEqual(ui.values(), [12, 1]);

// 单页禁用；单页上限仍可移动；不设上限时拖动端点不推动另一端。
ui = setup(1, 1, 1);
ui.pointer("pointerdown", 20);
ui.pointer("pointerup", 30);
ui.key(0, "End");
assert(ui.inputs.every(input => input.disabled));
assert.equal(ui.submissions.length, 0);
ui = setup(1, 1, 48, 1);
ui.pointer("pointerdown", 1);
ui.pointer("pointerup", 40);
assert.deepEqual(ui.values(), [40, 40]);
ui = setup(1, 48, 48, null);
ui.pointer("pointerdown", 1);
ui.pointer("pointerup", 40);
assert.deepEqual(ui.values(), [40, 48]);
console.log("Page range gestures passed: preview, atomic commit, pan, crossing, capture, cancel and keyboard.");
