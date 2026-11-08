// 初始化 Header 的响应式按钮布局与资源菜单交互。
({ element }) => {
    const header = element.querySelector(".mineru-demo-header");
    const menus = [...header.querySelectorAll(".mineru-header-menu")];
    let activeMenu = null;
    let pinned = false;

    // 以 Header 实际宽度切换四列与两列，并限制窄屏弹层的最大宽度。
    const resize = () => {
        const width = header.getBoundingClientRect().width;
        header.classList.toggle("is-wide", width >= 560);
        header.style.setProperty("--mineru-header-popover-max-width", `${Math.max(0, width - 32)}px`);
    };
    // 下一帧再写入布局，避免列数变化同时改变高度时触发 ResizeObserver 循环。
    const observer = new ResizeObserver(() => requestAnimationFrame(resize));
    resize();
    observer.observe(header);

    // 单次只展开一个菜单，同时同步可访问状态和隐藏链接的键盘可达性。
    const openMenu = (menu, pin = false) => {
        activeMenu = menu;
        pinned = pin;
        menus.forEach((item) => {
            const open = item === menu;
            item.classList.toggle("is-open", open);
            item.querySelector("button").setAttribute("aria-expanded", String(open));
            item.querySelector(".mineru-header-popover").hidden = !open;
        });
    };

    // 每个资源菜单分别处理悬停与点击，共用上面的展开状态。
    menus.forEach((menu) => {
        // 鼠标保留悬停展开；触屏只响应点击，防止轻触后的模拟悬停干扰收起。
        menu.addEventListener("pointerenter", (event) => {
            if (event.pointerType === "mouse" && activeMenu !== menu) openMenu(menu);
        });
        menu.addEventListener("pointerleave", () => {
            if (activeMenu === menu && !pinned && !menu.contains(document.activeElement)) openMenu(null);
        });
        // 首次点击固定悬停菜单，再次点击同一按钮收起，触屏和键盘沿用同一入口。
        menu.querySelector("button").addEventListener("click", () => {
            openMenu(activeMenu === menu && pinned ? null : menu, true);
        });
    });

    // 离开整个菜单后关闭弹层，避免 Tab 导航继续经过已经离开的菜单项。
    header.addEventListener("focusout", (event) => {
        if (activeMenu?.contains(event.target) && !activeMenu.contains(event.relatedTarget)) openMenu(null);
    });
    // Escape 收起并归还焦点；向下键直接进入当前菜单的第一个链接。
    header.addEventListener("keydown", (event) => {
        if (event.key === "Escape" && activeMenu) {
            const trigger = activeMenu.querySelector("button");
            openMenu(null);
            trigger.focus();
            event.preventDefault();
        } else if (event.key === "ArrowDown" && event.target.matches(".mineru-header-trigger")) {
            const menu = event.target.closest(".mineru-header-menu");
            openMenu(menu, true);
            menu.querySelector(".mineru-header-popover-link").focus();
            event.preventDefault();
        }
    });
    // 点击菜单外立即收起，也覆盖不接收键盘焦点的 Header 空白区域。
    document.addEventListener("pointerdown", (event) => {
        if (activeMenu && !activeMenu.contains(event.target)) openMenu(null);
    });
}
