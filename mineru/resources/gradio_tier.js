// 胶囊只负责呈现与用户输入；文件锁档、可用档位和会话偏好仍由共享联动事件决定。
({ element, props, trigger, watch, server, tiers, defaultIndex }) => {
    const ownerDocument = element.ownerDocument;
    const capsule = element.querySelector(".mineru-tier-capsule");
    const rail = element.querySelector(".mineru-tier-rail");
    const thumb = element.querySelector(".mineru-tier-thumb");
    const range = element.querySelector(".mineru-tier-range");
    const reset = element.querySelector(".mineru-tier-reset");
    const dots = element.querySelector(".mineru-tier-dots");
    const label = element.querySelector(".mineru-tier-label");
    const current = element.querySelector(".mineru-tier-current");
    const deviceButton = element.querySelector(".mineru-tier-device");
    const deviceTooltip = element.querySelector(".mineru-tier-device-tooltip");
    let runtimeDevice = null;
    let drag = null;
    let committed = Number(props.value);
    let visual = committed;
    let locked = props.locked;
    const maximum = tiers.length - 1;
    // 词典可能晚于组件挂载；初始化完成后由应用入口统一补齐本地化。
    const text = (key, values = {}) => window.__mineruI18n?.text(key, values) || (key === "tier" ? "解析等级" : "");
    // 所有提交值都夹紧为当前服务实际开放的整数档位。
    const clamp = value => Math.max(0, Math.min(maximum, Math.round(value)));
    // 高频拖动只改变化的文字和属性，避免反复触发整页本地化观察器。
    const setText = (selector, value) => {
        const node = element.querySelector(selector);
        if (node.textContent === value) return false;
        node.textContent = value;
        return true;
    };
    // 同值属性不重复写入 DOM；无障碍说明仍在档位或语言变化时更新。
    const setAttribute = (node, name, value) => {
        if (node.getAttribute(name) !== value) node.setAttribute(name, value);
    };

    // 按本地化文字的实际宽度给闪电和重置留空间，窄栏继续隐藏重置按钮。
    const resize = () => {
        const minimumWidth = Math.max(340, label.getBoundingClientRect().width * 2
            + current.getBoundingClientRect().width + 54);
        capsule.classList.toggle("is-wide", capsule.getBoundingClientRect().width >= minimumWidth);
    };
    const resizeObserver = new ResizeObserver(resize);
    resizeObserver.observe(capsule);
    resize();

    // 设备信息独立于档位偏好；提示、颜色和动效只消费当前页面的只读快照。
    const renderDevice = () => {
        capsule.dataset.runtime = runtimeDevice || "unknown";
        capsule.dataset.stars = runtimeDevice === "gpu" ? "fast" : capsule.dataset.tier === "advanced" ? "slow" : "none";
        const message = text(`runtime_device_${runtimeDevice || "unknown"}`) || "设备状态未知";
        setText(".mineru-tier-device-tooltip", message);
        setAttribute(deviceButton, "aria-label", message);
    };
    // 点击和悬浮只展开相同提示，不触发任何设备选型或网络查询。
    const showDeviceInfo = () => { deviceTooltip.hidden = false; };
    // Escape、失焦和外部点击均可主动关闭提示，避免焦点样式使提示重新出现。
    const hideDeviceInfo = () => { deviceTooltip.hidden = true; };
    deviceButton.addEventListener("pointerenter", showDeviceInfo);
    // 点击后保留焦点提示；仅悬浮时随鼠标移出关闭。
    deviceButton.addEventListener("pointerleave", () => {
        if (ownerDocument.activeElement !== deviceButton) hideDeviceInfo();
    });
    deviceButton.addEventListener("focus", showDeviceInfo);
    deviceButton.addEventListener("click", showDeviceInfo);
    deviceButton.addEventListener("blur", hideDeviceInfo);
    // 页面其他控件不会改变设备快照，只负责关闭已展开的说明。
    ownerDocument.addEventListener("pointerdown", event => {
        if (!deviceButton.contains(event.target)) hideDeviceInfo();
    });
    // 不阻断滑块自身的 Escape 取消语义。
    ownerDocument.addEventListener("keydown", event => {
        if (event.key === "Escape") hideDeviceInfo();
    });

    for (let index = 0; index < tiers.length; index++) {
        const dot = document.createElement("span");
        dot.className = "mineru-tier-dot";
        dots.appendChild(dot);
    }
    range.max = String(Math.max(1, maximum));

    // 连续位置只绘制外观，文字按最近档位显示，不向 Gradio 写入小数。
    const render = (value, announce = false) => {
        visual = Math.max(0, Math.min(maximum, value));
        const index = clamp(visual);
        const tier = locked ? "flash" : tiers[index];
        const disabled = locked || maximum === 0;
        capsule.dataset.tier = tier;
        capsule.style.setProperty("--tier-position", String(maximum ? visual / maximum : 0));
        range.value = String(index);
        const titleChanged = setText(".mineru-tier-title", text("tier"));
        const currentChanged = setText(".mineru-tier-current", text(`tier_compact_${tier}`) || tier);
        if (titleChanged || currentChanged) resize();
        renderDevice();
        element.querySelector(".mineru-tier-options").hidden = disabled;
        range.disabled = disabled;
        setAttribute(range, "aria-label", text("tier"));
        setAttribute(range, "aria-valuetext", text(`tier_${tier}`) || tier);
        const defaultLabel = text(`tier_compact_${tiers[defaultIndex]}`) || tiers[defaultIndex];
        const resetLabel = text("tier_reset", { tier: defaultLabel }) || `恢复默认等级：${defaultLabel}`;
        reset.disabled = disabled || index === defaultIndex;
        setAttribute(reset, "aria-label", resetLabel);
        setAttribute(reset, "title", locked ? text("tier_locked") : index === defaultIndex
            ? text("tier_at_default", { tier: defaultLabel }) : resetLabel);
        for (const [i, dot] of [...dots.children].entries()) {
            dot.dataset.passed = String(i <= index);
        }
        if (announce) setText(".mineru-tier-announcement", text(`tier_${tier}`) || tier);
    };

    // 先结束当前手势再释放捕获，正常松手产生的 lostpointercapture 不得回滚提交。
    const release = () => {
        const previous = drag;
        drag = null;
        capsule.classList.remove("is-dragging");
        ownerDocument.removeEventListener("pointermove", pointerMove, true);
        ownerDocument.removeEventListener("pointerup", pointerUp, true);
        ownerDocument.removeEventListener("pointercancel", pointerCancel, true);
        ownerDocument.removeEventListener("dragstart", preventNativeDrag, true);
        previous?.surface.remove();
        if (previous && rail.hasPointerCapture(previous.id)) rail.releasePointerCapture(previous.id);
    };
    // Escape、触屏滚动等明确取消才恢复已确认档位。
    const cancel = () => {
        if (!drag) return;
        release();
        render(committed);
    };
    // 文件锁档立即中止拖动；无关页码回执不打断当前跟手位置。
    const sync = () => {
        const lockChanged = locked !== props.locked;
        committed = Number(props.value);
        locked = props.locked;
        if (lockChanged) release();
        render(drag ? visual : committed);
    };

    // 松手、重置和键盘都走同一提交入口，只派发一次用户 input。
    const commit = value => {
        release();
        if (locked || maximum === 0) { sync(); return; }
        const index = clamp(value);
        render(index, true);
        if (index === committed) return;
        committed = index;
        props.value = index;
        trigger("input");
    };

    // 指针坐标是拖动的唯一输入；松手时也重算一次，避免快速移动漏掉最后一帧。
    const preview = event => {
        if (!drag || event.pointerId !== drag.id) return;
        render((event.clientX - drag.left - drag.offset) / drag.width * maximum);
    };
    // 捕获意外丢失后仍跟踪页面事件；漏收松手时保留最后位置，不追随重新进入的鼠标。
    const pointerMove = event => {
        if (!drag || event.pointerId !== drag.id) return;
        if (event.buttons === 0) { commit(visual); return; }
        preview(event);
    };
    // 页面级松手先补算最终坐标，再吸附提交，支持胶囊外部及预览区域。
    const pointerUp = event => {
        if (!drag || event.pointerId !== drag.id) return;
        preview(event);
        commit(visual);
    };
    // 非当前指针的取消不影响手势，触屏滚动等实际取消仍保持原有语义。
    const pointerCancel = event => { if (drag?.id === event.pointerId) cancel(); };
    // 窗口或输入意外失焦时结束当前选择，不把已拖到的位置退回起点。
    const finishInterrupted = () => { if (drag) commit(visual); };
    // 原生 HTML 拖放会取消指针流并显示禁拖光标，滑块手势期间明确阻止它接管。
    const preventNativeDrag = event => { if (drag) event.preventDefault(); };
    // 轨道自行处理拖动，阻止原生 range 与手动指针捕获争用浏览器内部滑块。
    rail.addEventListener("pointerdown", event => {
        if (range.disabled || event.button !== 0 || event.isPrimary === false || drag) return;
        event.preventDefault();
        rail.dataset.keyboard = "false";
        const bounds = rail.getBoundingClientRect();
        const knob = thumb.getBoundingClientRect();
        const center = knob.left + knob.width / 2;
        // 临时透明层隔离 PDF/Office iframe，保证跨预览拖动的松手仍属于当前文档。
        // 挂到 body 避免容器裁切；内联定位不受 Gradio 的 CSS 前缀改写影响。
        const surface = ownerDocument.createElement("div");
        surface.className = "mineru-tier-drag-surface";
        surface.setAttribute("aria-hidden", "true");
        Object.assign(surface.style, { position: "fixed", inset: "0", zIndex: "2147483647", cursor: "grabbing" });
        drag = { id: event.pointerId, surface, left: bounds.left + knob.width / 2,
            width: Math.max(1, bounds.width - knob.width),
            // 抓住圆点边缘不跳动；动画尚未结束时从实际可见位置接续拖动。
            offset: Math.abs(event.clientX - center) <= knob.width / 2 ? event.clientX - center : 0 };
        capsule.classList.add("is-dragging");
        ownerDocument.body.appendChild(surface);
        ownerDocument.addEventListener("pointermove", pointerMove, true);
        ownerDocument.addEventListener("pointerup", pointerUp, true);
        ownerDocument.addEventListener("pointercancel", pointerCancel, true);
        ownerDocument.addEventListener("dragstart", preventNativeDrag, true);
        rail.setPointerCapture(event.pointerId);
        range.focus({ preventScroll: true });
        preview(event);
    });
    // 捕获丢失事件的 buttons 在浏览器间不一致；仅由页面移动、松手或失焦结束手势。
    // 辅助技术直接修改原生值仍能提交；指针拖动期间不接收原生滑块事件。
    const nativeInput = () => {
        if (!drag && !range.disabled) commit(Number(range.value));
    };
    range.addEventListener("input", nativeInput);
    range.addEventListener("change", nativeInput);
    range.addEventListener("blur", finishInterrupted);
    window.addEventListener("blur", finishInterrupted);
    // 指针聚焦不显示整条焦点环，Tab 和辅助技术聚焦仍保留可见提示。
    range.addEventListener("focus", () => { rail.dataset.keyboard = String(!drag); });
    // 键盘维持整数档位语义，Escape 可取消尚未提交的指针手势。
    range.addEventListener("keydown", event => {
        if (event.key === "Escape") {
            if (drag) { event.preventDefault(); cancel(); }
            return;
        }
        if (range.disabled || drag) return;
        const index = clamp(visual);
        const values = { ArrowRight: index + 1, ArrowUp: index + 1, ArrowLeft: index - 1, ArrowDown: index - 1,
            Home: 0, End: maximum, PageUp: index + 1, PageDown: index - 1 };
        if (!Object.hasOwn(values, event.key)) return;
        event.preventDefault();
        rail.dataset.keyboard = "true";
        commit(values[event.key]);
    });
    // 默认档位由服务端共享选择器注入；锁档时不改写会话的未锁定偏好。
    reset.addEventListener("click", () => {
        if (!reset.disabled) commit(defaultIndex);
    });
    watch(["value", "selected_tier", "locked"], sync);
    window.__mineruTier = { localize: sync };
    sync();
    // 每次页面挂载只发起一次 health 查询；迟到的响应不重绘滑块或打断当前拖动。
    const receiveDevice = device => {
        runtimeDevice = device === "cpu" || device === "gpu" ? device : null;
        renderDevice();
    };
    server.read_runtime_device().then(receiveDevice)
        // 状态查询失败不影响解析，保持未知并等待用户下次重新打开页面。
        .catch(() => receiveDevice(null));
}
