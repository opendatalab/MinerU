// 执行真实胶囊脚本，验证连续拖动不会提前提交、锁档不会污染偏好。
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const script = fs.readFileSync(path.join(__dirname, "../../mineru/resources/gradio_tier.js"), "utf8");

// 最小 DOM 保留输入值、属性和事件；不替代生产状态转换算法。
function node() {
    const classes = new Set();
    return {
        children: [], dataset: {}, attributes: {}, events: {}, value: "0", disabled: false,
        style: {
            // 保存连续视觉位置，不能只检查最终整数档位。
            setProperty(name, value) { this[name] = value; },
        },
        classList: {
            // 记录跟手状态，验证回执和捕获丢失不错误启用吸附动画。
            add(name) { classes.add(name); },
            // 结束手势时移除拖动态。
            remove(name) { classes.delete(name); },
            // 尺寸观察器只更新是否显示重置按钮。
            toggle(name, enabled) { if (enabled) classes.add(name); else classes.delete(name); },
            // 允许断言当前拖动态。
            contains(name) { return classes.has(name); },
        },
        // 创建刻度时保留它们的真实顺序。
        appendChild(child) { this.children.push(child); child.parent = this; },
        // 透明拖动层必须随手势结束移除，不能阻断后续页面交互。
        remove() { if (this.parent) this.parent.children = this.parent.children.filter(child => child !== this); },
        // 记录无障碍标签和当前选中状态。
        setAttribute(name, value) { this.attributes[name] = value; },
        // 生产代码仅在属性变化时写入 DOM。
        getAttribute(name) { return this.attributes[name]; },
        // 判断点击是否来自说明按钮或其装饰图标。
        contains(target) { return target === this || this.children.includes(target); },
        // 重放浏览器事件时调用生产回调。
        addEventListener(name, callback) { this.events[name] = callback; },
        // 页面级监听只在手势期间存在，重复清理保持幂等。
        removeEventListener(name, callback) { if (this.events[name] === callback) delete this.events[name]; },
        // 指针捕获在此只记录请求，不模拟浏览器拖动算法。
        setPointerCapture(id) { this.capture = id; },
        // 捕获只属于当前指针。
        hasPointerCapture(id) { return this.capture === id; },
        // 同步派发捕获丢失，检查释放期间的重入也不会回滚或重复提交。
        releasePointerCapture(id) {
            assert.equal(this.capture, id);
            this.capture = null;
            this.events.lostpointercapture?.({ pointerId: id });
        },
        // 聚焦不模拟浏览器原生拖动。
        focus() {},
        // 单元测试只提供初始宽度，实际尺寸变化由真实浏览器覆盖。
        getBoundingClientRect() { return { left: 100, width: 300 }; },
    };
}

// 每个控件独立持有 props 和提交记录，模拟 Gradio watch 的输出回执。
function setup(tiers, position = 0, defaultIndex = position) {
    const nodes = Object.fromEntries(["capsule", "rail", "thumb", "range", "reset", "dots", "title", "current", "options", "announcement",
        "label", "device", "device-tooltip"]
        .map(name => [name, node()]));
    nodes["device-tooltip"].hidden = true;
    const props = { value: position, selected_tier: tiers[position], locked: false };
    const submissions = [];
    // 保留目标档位参数，验证动态按钮说明没有使用当前档位替代默认值。
    const window = { __mineruI18n: { text: (key, values = {}) => values.tier ? `${key}: ${values.tier}` : key },
        events: {},
        // 窗口失焦和输入失焦分别覆盖，不能把取消当成提交。
        addEventListener(name, callback) { this.events[name] = callback; },
    };
    // 将 CSS 连续位置转换为可见圆点，允许单独模拟尚未完成的吸附动画。
    nodes.thumb.getBoundingClientRect = () => ({ width: 30,
        left: nodes.thumb.animatedLeft ?? 100 + Number(nodes.capsule.style["--tier-position"] || 0) * 270 });
    let watcher;
    let receiveRuntime, rejectRuntime;
    let runtimeReads = 0;
    const server = {
        // 延迟设备响应，用于覆盖拖动期间返回和每页只查询一次的约束。
        read_runtime_device() {
            runtimeReads++;
            return new Promise((resolve, reject) => { receiveRuntime = resolve; rejectRuntime = reject; });
        },
    };
    const ownerDocument = { ...node(), body: node(), createElement: node };
    // 不模拟浏览器布局，只提供生产脚本所需的尺寸观察接口。
    class ResizeObserver {
        // 控件在本测试中不会改变布局，无需触发异步尺寸回调。
        observe() {}
    }
    vm.runInNewContext(`(${script})`, { window, document: ownerDocument, ResizeObserver })({
        element: { ownerDocument, querySelector: selector => nodes[selector.replace(".mineru-tier-", "")] },
        props, tiers, defaultIndex, server,
        trigger: event => { submissions.push([event, props.value]); },
        watch: (_keys, callback) => { watcher = callback; },
    });
    return { nodes, props, submissions, window, document: ownerDocument,
        // 排空设备响应微任务后检查真实前端处理结果。
        async runtime(value) { receiveRuntime(value); await Promise.resolve(); },
        // 失败时等待 catch 完成，确保未知状态不影响其余控件。
        async failRuntime() { rejectRuntime(new Error("offline")); await Promise.resolve(); await Promise.resolve(); },
        // 读取网络调用次数，用户交互不应触发重新查询。
        runtimeReads() { return runtimeReads; },
        // 回执到来时同步现有实例，不能重建控件掩盖拖动竞态。
        update(patch) { Object.assign(props, patch); watcher(); },
        // 键盘默认行为必须被阻止，否则细步长会额外改变值。
        key(key) { let prevented = false; nodes.range.events.keydown({ key, preventDefault() { prevented = true; } }); assert(prevented); },
        // 用真实指针坐标驱动生产事件，不再手动替原生 range 写入拖动值。
        pointer(type, clientX, pointerId = 7, extra = {}) {
            let prevented = false;
            const handler = nodes.rail.events[type] || ownerDocument.events[type];
            handler?.({ clientX, pointerId, button: 0, buttons: type === "pointerup" ? 0 : 1, isPrimary: true,
                // 验证按下时明确阻止浏览器原生拖动。
                preventDefault() { prevented = true; }, ...extra });
            return prevented;
        },
        // 测试坐标由固定轨道几何换算，包含圆点两端的半径留白。
        x(value) { return 115 + value / Math.max(1, tiers.length - 1) * 270; },
        // 小数只存在于外观，不进入 Gradio 或无障碍输入的整数值。
        visual() { return Number(nodes.capsule.style["--tier-position"]) * (tiers.length - 1); },
    };
}

const ui = setup(["flash", "basic", "standard", "advanced"], 1);
assert(ui.pointer("pointerdown", ui.x(1)));
ui.pointer("pointermove", ui.x(1.67));
assert.equal(ui.nodes.current.textContent, "tier_compact_standard");
assert.equal(ui.props.value, 1);
assert.deepEqual(ui.submissions, []);
assert.equal(ui.nodes.range.value, "2");
assert(Math.abs(ui.visual() - 1.67) < 1e-9);
ui.pointer("pointerup", ui.x(2.49));
ui.nodes.range.events.change();
assert.equal(ui.nodes.range.value, "2");
assert.deepEqual(ui.submissions, [["input", 2]]);
ui.key("End");
ui.key("ArrowRight");
assert.equal(ui.props.value, 3);
assert.equal(ui.submissions.length, 2);
assert.equal(ui.nodes.reset.disabled, false);
assert.equal(ui.nodes.reset.attributes["aria-label"], "tier_reset: tier_compact_basic");

// 文件切换中途锁到 Flash，即使迟到的 pointerup/change 也不能写回临时档位。
ui.pointer("pointerdown", ui.x(3), 8);
ui.pointer("pointermove", ui.x(0.4), 8);
ui.update({ locked: true, selected_tier: "flash" });
assert.equal(ui.nodes.rail.capture, null);
ui.pointer("pointerup", ui.x(0.4), 8);
ui.nodes.range.events.change();
assert.equal(ui.nodes.options.hidden, true);
assert.equal(ui.nodes.current.textContent, "tier_compact_flash");
assert.equal(ui.props.value, 3);
assert.equal(ui.submissions.length, 2);
assert.equal(ui.nodes.reset.disabled, true);
assert.equal(ui.nodes.reset.attributes.title, "tier_locked");
ui.nodes.reset.events.click();
assert.equal(ui.props.value, 3);
assert.equal(ui.submissions.length, 2);
ui.update({ locked: false, selected_tier: "advanced" });
assert.equal(ui.nodes.range.value, "3");
assert.equal(ui.nodes.options.hidden, false);
assert.equal(ui.nodes.reset.disabled, false);

// 页数回执不能打断进行中的拖动；取消指针则恢复已提交值。
ui.pointer("pointerdown", ui.x(3), 9);
ui.pointer("pointermove", ui.x(1.2), 9);
ui.update({ value: 3 });
assert(Math.abs(ui.visual() - 1.2) < 1e-9);
assert(ui.nodes.capsule.classList.contains("is-dragging"));
ui.pointer("pointercancel", ui.x(1.2), 9);
assert.equal(ui.nodes.range.value, "3");
assert.equal(ui.submissions.length, 2);
ui.nodes.reset.events.click();
assert.equal(ui.props.value, 1);
assert.equal(ui.nodes.reset.disabled, true);
assert.equal(ui.nodes.reset.attributes.title, "tier_at_default: tier_compact_basic");
assert.deepEqual(ui.submissions.at(-1), ["input", 1]);
ui.nodes.reset.events.click();
assert.equal(ui.submissions.length, 3);

// 即使没有最后一次 pointermove/input，pointerup 也必须使用松手坐标。
const fast = setup(["flash", "basic", "standard", "advanced"], 3);
fast.pointer("pointerdown", fast.x(3));
fast.pointer("pointerup", fast.x(0));
assert.equal(fast.props.value, 0);
assert.deepEqual(fast.submissions, [["input", 0]]);
fast.pointer("lostpointercapture", fast.x(0));
assert.equal(fast.props.value, 0);
assert.equal(fast.nodes.rail.capture, null);

// 轨道点击直接定位；从圆点边缘拖动保留抓取偏移。
fast.pointer("pointerdown", fast.x(1.3));
fast.pointer("pointerup", fast.x(1.3));
assert.equal(fast.props.value, 1);
fast.pointer("pointerdown", fast.x(1) + 10);
assert.equal(fast.visual(), 1);
fast.pointer("pointermove", fast.x(2) + 10);
assert.equal(fast.visual(), 2);
fast.pointer("pointerup", fast.x(2) + 10);
assert.equal(fast.props.value, 2);

// 吸附动画尚未到达目标时，重新按住可见圆点不能跳到目标值。
fast.nodes.thumb.animatedLeft = fast.x(1.6) - 15;
fast.pointer("pointerdown", fast.x(1.6) + 5);
assert(Math.abs(fast.visual() - 1.6) < 1e-9);
delete fast.nodes.thumb.animatedLeft;
fast.pointer("pointerup", fast.x(0) + 5);
assert.equal(fast.props.value, 0);

// 其他指针、原生事件和较早的状态回执不能覆盖进行中的连续位置。
fast.pointer("pointerdown", fast.x(0), 11);
fast.pointer("pointermove", fast.x(1.7), 11);
// 重新捕获同一鼠标后才送达的旧捕获丢失通知应被忽略。
fast.pointer("lostpointercapture", fast.x(0), 11);
fast.pointer("pointerdown", fast.x(3), 12);
fast.pointer("pointermove", fast.x(3), 12);
fast.pointer("pointerup", fast.x(3), 12);
fast.pointer("pointercancel", fast.x(3), 12);
fast.pointer("lostpointercapture", fast.x(3), 12);
fast.nodes.range.events.input();
fast.nodes.range.events.change();
fast.update({ value: 1, selected_tier: "basic" });
assert(Math.abs(fast.visual() - 1.7) < 1e-9);
assert.equal(fast.nodes.rail.capture, 11);
const beforeRelease = fast.submissions.length;
fast.pointer("pointerup", fast.x(2.2), 11);
assert.equal(fast.props.value, 2);
assert.equal(fast.submissions.length, beforeRelease + 1);

// Escape 和实际 pointercancel 保持取消语义，迟到的松手不得再次提交。
for (const reason of ["escape", "pointercancel"]) {
    const count = fast.submissions.length;
    fast.pointer("pointerdown", fast.x(2));
    fast.pointer("pointermove", fast.x(0.2));
    if (reason === "escape") fast.key("Escape");
    else fast.pointer("pointercancel", fast.x(0.2));
    fast.pointer("pointerup", fast.x(0));
    assert.equal(fast.visual(), 2);
    assert.equal(fast.props.value, 2);
    assert.equal(fast.submissions.length, count);
    assert.equal(fast.nodes.rail.capture, null);
    assert.equal(fast.document.body.children.length, 0);
    assert.deepEqual(Object.keys(fast.document.events).sort(), ["keydown", "pointerdown"]);
}

// 按住时失去捕获仍由页面事件跟踪，在胶囊外松手也只提交一次最终位置。
const outside = setup(["flash", "basic", "standard", "advanced"], 3);
outside.pointer("pointerdown", outside.x(3));
outside.pointer("pointermove", outside.x(1.5));
assert.equal(outside.document.body.children.length, 1);
assert(outside.pointer("dragstart", outside.x(1.5)));
outside.nodes.rail.releasePointerCapture(7);
assert.equal(outside.visual(), 1.5);
outside.pointer("pointermove", outside.x(0.8));
outside.pointer("pointerup", outside.x(1.1));
assert.equal(outside.props.value, 1);
assert.deepEqual(outside.submissions, [["input", 1]]);
assert.equal(outside.document.body.children.length, 0);
assert.deepEqual(Object.keys(outside.document.events).sort(), ["keydown", "pointerdown"]);

// 已松手的捕获丢失、输入或窗口失焦均保留最后位置，不能退回起点。
for (const reason of ["released-capture", "input-blur", "window-blur", "missed-up"]) {
    const interrupted = setup(["flash", "basic", "standard", "advanced"], 3);
    interrupted.pointer("pointerdown", interrupted.x(3));
    interrupted.pointer("pointermove", interrupted.x(1.2));
    if (reason === "released-capture") {
        interrupted.nodes.rail.capture = null;
        interrupted.pointer("lostpointercapture", interrupted.x(0), 7, { buttons: 0 });
        // WebKit 可能在仍按住时报告 buttons=0，不能以捕获丢失事件提前结束。
        assert.equal(interrupted.submissions.length, 0);
        interrupted.pointer("pointermove", interrupted.x(0), 7, { buttons: 0 });
    } else if (reason === "input-blur") interrupted.nodes.range.events.blur();
    else if (reason === "window-blur") interrupted.window.events.blur();
    else interrupted.pointer("pointermove", interrupted.x(3), 7, { buttons: 0 });
    interrupted.pointer("pointerup", interrupted.x(0));
    assert.equal(interrupted.props.value, 1);
    assert.deepEqual(interrupted.submissions, [["input", 1]]);
    assert.equal(interrupted.document.body.children.length, 0);
    assert.deepEqual(Object.keys(interrupted.document.events).sort(), ["keydown", "pointerdown"]);
}

// 辅助技术的 input/change 仍生效，重复通知不能产生两次业务提交。
const beforeNative = fast.submissions.length;
fast.nodes.range.value = "1";
fast.nodes.range.events.input();
fast.nodes.range.events.change();
assert.equal(fast.props.value, 1);
assert.equal(fast.submissions.length, beforeNative + 1);
assert.equal(fast.pointer("pointerdown", fast.x(3), 7, { button: 2 }), false);
assert.equal(fast.pointer("pointerdown", fast.x(3), 8, { isPrimary: false }), false);
assert.equal(fast.props.value, 1);

// 缺档和单档只展示实际能力；原生文档的 Flash 不要求存在于公开 PDF 档位中。
for (const tiers of [["standard"], ["basic", "advanced"], ["advanced"]]) {
    const subset = setup(tiers);
    assert.equal(subset.nodes.dots.children.length, tiers.length);
    assert.equal(subset.nodes.options.hidden, tiers.length === 1);
    assert.equal(subset.nodes.range.max, String(Math.max(1, tiers.length - 1)));
    assert.equal(subset.nodes.reset.disabled, true);
    subset.update({locked: true, selected_tier: "flash"});
    assert.equal(subset.nodes.current.textContent, "tier_compact_flash");
    subset.update({locked: false, selected_tier: tiers[0]});
    assert.equal(subset.nodes.current.textContent, `tier_compact_${tiers[0]}`);
}

// 重置目标使用服务注入的默认位置，不能写死 Standard 或第一个可用档位。
for (const [tiers, defaultIndex, selectedIndex] of [
    [["flash", "basic", "standard", "advanced"], 2, 3],
    [["flash", "basic", "standard", "advanced"], 1, 3],
    [["basic", "standard", "advanced"], 1, 0],
    [["basic", "advanced"], 0, 1],
    [["flash", "advanced"], 0, 1],
]) {
    const resetUi = setup(tiers, selectedIndex, defaultIndex);
    resetUi.nodes.reset.events.click();
    assert.equal(resetUi.props.value, defaultIndex);
    assert.equal(resetUi.nodes.range.value, String(defaultIndex));
    assert.equal(resetUi.nodes.current.textContent, `tier_compact_${tiers[defaultIndex]}`);
    assert.deepEqual(resetUi.submissions, [["input", defaultIndex]]);
    // 无关回执和重复点击不能重复派发，也不能改变服务默认位置。
    resetUi.update({ value: defaultIndex });
    resetUi.nodes.reset.events.click();
    assert.equal(resetUi.submissions.length, 1);
}

// 覆盖设备状态与四个档位的组合，以及查询只发生在首次挂载的约束。
async function testRuntimeDevice() {
    for (const device of ["gpu", "cpu", null, "mps"]) {
        for (const [position, tier] of ["flash", "basic", "standard", "advanced"].entries()) {
            const runtimeUi = setup(["flash", "basic", "standard", "advanced"], position);
            assert.equal(runtimeUi.runtimeReads(), 1);
            await runtimeUi.runtime(device);
            const expectedDevice = ["cpu", "gpu"].includes(device) ? device : "unknown";
            const expectedEffect = device === "gpu" ? "fast" : tier === "advanced" ? "slow" : "none";
            assert.equal(runtimeUi.nodes.capsule.dataset.runtime, expectedDevice);
            assert.equal(runtimeUi.nodes.capsule.dataset.stars, expectedEffect);
            assert.equal(runtimeUi.nodes.device.attributes["aria-label"], `runtime_device_${expectedDevice}`);
            assert.equal(runtimeUi.nodes["device-tooltip"].hidden, true);
            runtimeUi.nodes.device.events.pointerenter();
            assert.equal(runtimeUi.nodes["device-tooltip"].hidden, false);
            runtimeUi.nodes.device.events.pointerleave();
            assert.equal(runtimeUi.nodes["device-tooltip"].hidden, true);
            runtimeUi.nodes.device.events.click();
            assert.equal(runtimeUi.nodes["device-tooltip"].textContent, `runtime_device_${expectedDevice}`);
            runtimeUi.document.events.keydown({ key: "Escape" });
            assert.equal(runtimeUi.nodes["device-tooltip"].hidden, true);
            runtimeUi.nodes.device.events.focus();
            runtimeUi.nodes.device.events.blur();
            assert.equal(runtimeUi.nodes["device-tooltip"].hidden, true);
            runtimeUi.nodes.device.events.click();
            runtimeUi.document.events.pointerdown({ target: runtimeUi.nodes.range });
            assert.equal(runtimeUi.nodes["device-tooltip"].hidden, true);
            assert.equal(runtimeUi.props.value, position);
            assert.deepEqual(runtimeUi.submissions, []);
            runtimeUi.update({ locked: true, selected_tier: "flash" });
            assert.equal(runtimeUi.nodes.capsule.dataset.runtime, expectedDevice);
            assert.equal(runtimeUi.nodes.capsule.dataset.stars, device === "gpu" ? "fast" : "none");
            runtimeUi.update({ locked: false, selected_tier: tier });
            assert.equal(runtimeUi.nodes.capsule.dataset.stars, expectedEffect);
            // 语言变化只重绘提示，设备快照不能被清空或重新读取。
            runtimeUi.window.__mineruI18n.text = key => `en:${key}`;
            runtimeUi.window.__mineruTier.localize();
            assert.equal(runtimeUi.nodes["device-tooltip"].textContent, `en:runtime_device_${expectedDevice}`);
            assert.equal(runtimeUi.runtimeReads(), 1);
        }
    }
    const late = setup(["flash", "basic", "standard", "advanced"], 1);
    late.pointer("pointerdown", late.x(1));
    late.pointer("pointermove", late.x(1.8));
    await late.runtime("gpu");
    assert(Math.abs(late.visual() - 1.8) < 1e-9);
    assert(late.nodes.capsule.classList.contains("is-dragging"));
    assert.equal(late.props.value, 1);
    assert.deepEqual(late.submissions, []);
    late.pointer("pointerup", late.x(2));
    assert.equal(late.props.value, 2);
    assert.equal(late.runtimeReads(), 1);
    const failed = setup(["advanced"]);
    await failed.failRuntime();
    assert.equal(failed.nodes.capsule.dataset.runtime, "unknown");
    assert.equal(failed.nodes.capsule.dataset.stars, "slow");
    assert.equal(failed.nodes.options.hidden, true);
}
// 异步断言失败必须使 Node 及调用它的 pytest 返回失败。
testRuntimeDevice().catch(error => { console.error(error); process.exitCode = 1; });
