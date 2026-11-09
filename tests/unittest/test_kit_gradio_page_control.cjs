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
        dataset: {}, events: {}, attrs: {}, children: [], value: "1", disabled: false,
        style: { values: {},
            // 记录连续位置，检查拖动时不会提前吸附整数。
            setProperty(key, value) { this.values[key] = value; } },
        // 保留无障碍名称以验证交叉后的语义。
        setAttribute(key, value) { this.attrs[key] = value; },
        // 读取旧名称，确保未变化时不产生多余 DOM 属性更新。
        getAttribute(key) { return this.attrs[key]; },
        // 同一元素只为每种测试事件安装一个控制器回调。
        addEventListener(name, callback) { this.events[name] = callback; },
        // 手势结束后移除页面级监听，不能影响后续上传与预览操作。
        removeEventListener(name, callback) { if (this.events[name] === callback) delete this.events[name]; },
        // 透明层挂载到模拟文档，验证每个退出路径都能清理。
        appendChild(child) { this.children.push(child); child.parent = this; },
        // 移除当前透明层，不影响其他节点。
        remove() { if (this.parent) this.parent.children = this.parent.children.filter(child => child !== this); },
        // 轨道有效宽度 470px，每页正好对应 10px。
        getBoundingClientRect() { return { left: 0, width: 498 }; },
        // 记录并检查捕获归属，不模拟原生拖动算法。
        setPointerCapture(id) { this.capture = id; },
        // 只允许当前捕获指针释放控制权。
        hasPointerCapture(id) { return this.capture === id; },
        // 记录已释放，供取消和换文件断言。
        releasePointerCapture(id) {
            this.capture = null;
            this.events.lostpointercapture?.({ pointerId: id, buttons: 1 });
        },
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
        // 窗口失焦结束当前手势，保留最后可见范围。
        addEventListener(name, callback) { this.events[name] = callback; } };
    let watcher;
    const ownerDocument = { ...node(), body: node(), createElement: node };
    vm.runInNewContext(`(${script})`, { window, document: ownerDocument })({
        element: {
            ownerDocument,
            // 数字和摘要位于选页组件共同的父容器。
            closest() { return { querySelector: key => leaves[key] }; },
            querySelector: selector => selector === ".mineru-page-sliders" ? rail
                : inputs[selector.includes("-a ") ? 0 : 1],
        },
        props, maxPages: limit,
        trigger: event => submissions.push({ event, state: JSON.parse(props.value) }),
        watch: (_keys, callback) => { watcher = callback; },
    });
    return { rail, inputs, props, submissions, leaves, window, document: ownerDocument,
        // 按页坐标重放真实指针事件，可额外指定取消与多指属性。
        pointer(type, page, patch = {}) {
            let prevented = false;
            const event = { button: 0, buttons: type === "pointerup" ? 0 : 1, pointerId: 7,
                isPrimary: true, clientX: 14 + (page - 1) * 10,
                // 记录原生拖放是否被阻止，避免只验证生产回调的输出值。
                preventDefault() { prevented = true; }, ...patch };
            ownerDocument.events[type]?.(event);
            rail.events[type]?.(event);
            return prevented;
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

// 中段按下立即定位最近端点，连续拖动只控制该端，松手前不提交。
ui = setup(1, 48, 48, null);
ui.pointer("pointerdown", 18);
assert.equal(ui.rail.dataset.active, "a");
assert.deepEqual(ui.values(), [18, 48]);
assert.equal(ui.submissions.length, 0);
ui.pointer("pointermove", 22.4);
assert.deepEqual(ui.values(), [22, 48]);
assert(Math.abs(Number(ui.inputs[0].parentElement.style.values["--page-position"]) - 21.4 / 47) < 1e-10);
ui.pointer("pointerup", 25);
assert.deepEqual(ui.values(), [25, 48]);
assert.equal(ui.submissions.length, 1);
ui.pointer("pointerdown", 40);
assert.equal(ui.rail.dataset.active, "b");
assert.deepEqual(ui.values(), [25, 40]);
ui.pointer("pointermove", 36.2);
ui.pointer("pointerup", 36.2);
assert.deepEqual(ui.values(), [25, 36]);
assert.equal(ui.submissions.length, 2);

// 已交叉的物理端点仍按几何距离选择；等距时优先语义起点。
ui = setup(40, 10, 48, null);
ui.pointer("pointerdown", 20);
assert.equal(ui.rail.dataset.active, "b");
assert.deepEqual(ui.values(), [40, 20]);
ui.pointer("pointerup", 25);
assert.deepEqual(ui.values(), [40, 25]);
assert.equal(ui.submissions[0].state.start_handle, "b");
for (const [a, b, expected, active] of [[10, 30, [20, 30], "a"], [30, 10, [30, 20], "b"]]) {
    ui = setup(a, b, 48, null);
    ui.pointer("pointerdown", 20);
    assert.equal(ui.rail.dataset.active, active);
    assert.deepEqual(ui.values(), expected);
    ui.pointer("pointerup", 20);
    assert.equal(ui.submissions.length, 1);
}

// 可见圆点内按下保留偏移；交叉后继续控制同一个物理端点。
ui = setup();
ui.pointer("pointerdown", 1.8);
assert.deepEqual(ui.values(), [1, 20]);
ui.pointer("pointermove", 30.8);
assert.deepEqual(ui.values(), [30, 20]);
ui.pointer("pointerup", 40.8);
assert.deepEqual(ui.values(), [40, 21]);
assert.equal(ui.submissions[0].state.start_handle, "b");
assert.equal(ui.inputs[0].attrs["aria-label"], "end_page{}");

// 圆点外但距离很近的中段也属于轨道点击，不能再被旧的透明热区吞掉。
ui = setup();
ui.pointer("pointerdown", 2.8);
assert.deepEqual(ui.values(), [3, 20]);
ui.pointer("pointerup", 2.8);
assert.deepEqual(ui.values(), [3, 20]);

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

// 明确取消和 Escape 回滚预览，透明层和页面级监听必须一起回收。
for (const reason of ["pointercancel", "Escape"]) {
    ui = setup();
    ui.pointer("pointerdown", 20);
    ui.pointer("pointermove", 35);
    if (reason === "Escape") ui.key(1, reason);
    else ui.pointer(reason, 35);
    ui.pointer("pointerup", 35);
    assert.deepEqual(ui.values(), [1, 20]);
    assert.equal(ui.submissions.length, 0);
    assert.equal(ui.document.body.children.length, 0);
    assert.deepEqual(Object.keys(ui.document.events), []);
}

// 按住时丢失捕获仍通过页面事件拖动，在胶囊外松手提交最终范围。
ui = setup();
ui.pointer("pointerdown", 20);
ui.pointer("pointermove", 30);
assert(ui.pointer("dragstart", 30));
ui.rail.releasePointerCapture(7);
assert.deepEqual(ui.values(), [11, 30]);
assert.equal(ui.document.body.children.length, 1);
ui.pointer("pointermove", 35);
ui.pointer("pointerup", 40);
assert.deepEqual(ui.values(), [21, 40]);
assert.equal(ui.submissions.length, 1);
assert.equal(ui.document.body.children.length, 0);
assert.deepEqual(Object.keys(ui.document.events), []);

// 窗口失焦或漏收松手时保留最后范围，不能回弹或跟随重入鼠标再次跳动。
for (const reason of ["blur", "released-capture", "missed-up"]) {
    ui = setup();
    ui.pointer("pointerdown", 20);
    ui.pointer("pointermove", 35);
    if (reason === "blur") ui.window.events.blur();
    else if (reason === "released-capture") {
        ui.rail.capture = null;
        ui.pointer("lostpointercapture", 1, { buttons: 0 });
        // 捕获丢失的 buttons=0 不可靠，必须等真实移动或松手确认。
        assert.equal(ui.submissions.length, 0);
        ui.pointer("pointermove", 1, { buttons: 0 });
    } else ui.pointer("pointermove", 1, { buttons: 0 });
    ui.pointer("pointerup", 1);
    assert.deepEqual(ui.values(), [16, 35]);
    assert.equal(ui.submissions.length, 1);
    assert.equal(ui.document.body.children.length, 0);
    assert.deepEqual(Object.keys(ui.document.events), []);
}

// 换文件会取消旧捕获，迟到的松手不能把旧范围写入新文档。
ui = setup();
ui.pointer("pointerdown", 20);
ui.pointer("pointermove", 35);
ui.update({ path: "/new.pdf", page_count: 12, handle_a: 1, handle_b: 12 });
ui.pointer("pointerup", 35);
assert.deepEqual(ui.values(), [1, 12]);
assert.equal(ui.submissions.length, 0);
assert.equal(ui.document.body.children.length, 0);
assert.deepEqual(Object.keys(ui.document.events), []);
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
console.log("Page range gestures passed: nearest endpoint, preview, atomic commit, crossing, capture recovery and keyboard.");
