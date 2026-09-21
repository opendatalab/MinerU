// 可信阅读器脚本独立于书籍内容；章节 iframe 始终禁止脚本执行。
(async () => {
    const params = new URLSearchParams(window.location.search);
    const messages = JSON.parse(params.get("messages") || "{}");
    const status = document.getElementById("status");
    const viewer = document.getElementById("viewer");
    const toc = document.getElementById("toc");
    const pageInput = document.getElementById("page");
    const pagesOutput = document.getElementById("pages");
    const previous = document.getElementById("previous");
    const next = document.getElementById("next");
    let book;
    let rendition;
    let failed = false;
    let disposed = false;
    let updateFrame;
    let scrollContainer = viewer;
    let resizeObserver;
    let observedScrollContainer;
    let activeSpineIndex = null;
    let navigationSpineIndex = null;
    const SPINE_SWITCH_RATIO = 0.5;

    const fail = (error) => {
        failed = true;
        status.textContent = messages.epub_preview_failed || "Could not load EPUB preview.";
        status.setAttribute("role", "alert");
        pageInput.disabled = previous.disabled = next.disabled = toc.disabled = true;
        console.warn("EPUB preview failed", error);
    };

    previous.title = previous.ariaLabel = messages.epub_previous || "Previous page";
    next.title = next.ariaLabel = messages.epub_next || "Next page";
    pageInput.setAttribute("aria-label", messages.epub_spine || messages.epub_page || "Spine");
    toc.ariaLabel = messages.epub_contents || "Contents";
    status.textContent = messages.epub_loading || "Loading EPUB…";
    pageInput.disabled = previous.disabled = next.disabled = true;

    const isExternalLink = (href) => /^(?:[a-z][a-z\d+.-]*:|\/\/)/i.test(href);

    // 把章节内相对链接归一化为 EPUB 包内路径，保证 footnote 和跨章节链接使用同一入口。
    const resolveInternalTarget = (href, sourceHref = "") => {
        const raw = String(href || "").trim();
        if (!raw || isExternalLink(raw)) return null;
        const base = new URL(sourceHref || "/", "https://mineru.invalid/");
        const resolved = new URL(raw, base);
        return `${resolved.pathname.replace(/^\/+/, "")}${resolved.search}${resolved.hash}`;
    };

    // 根据当前视口内各个 spine 的可见比例更新顶部编号，避免长章节滚动时误切页。
    const updateActiveSpine = () => {
        if (!book || !rendition || !viewer) return;
        const bounds = viewer.getBoundingClientRect();
        const candidates = [];
        const views = rendition.views?.();
        if (views && typeof views.forEach === "function") {
            views.forEach((view) => {
                const index = Number(view?.section?.index ?? view?.index);
                const element = view?.element;
                if (!Number.isInteger(index) || !element?.getBoundingClientRect) return;
                const rect = element.getBoundingClientRect();
                const visibleHeight = Math.max(0, Math.min(rect.bottom, bounds.bottom) - Math.max(rect.top, bounds.top));
                if (visibleHeight > 0 && rect.height > 0) {
                    candidates.push({index, ratio: visibleHeight / Math.max(1, bounds.height)});
                }
            });
        }
        if (!candidates.length) return;
        candidates.sort((left, right) => right.ratio - left.ratio);
        const best = candidates[0];
        const active = candidates.find((candidate) => candidate.index === activeSpineIndex);
        if (active && best.index !== activeSpineIndex && best.ratio <= SPINE_SWITCH_RATIO) return;
        activeSpineIndex = best.index;
    };

    // 按预览窗口高度设置连续管理器的前后加载范围，只保留当前及相邻 spine 附近内容。
    const updateLoadWindow = () => {
        const manager = rendition?.manager;
        if (!manager?.settings) return;
        manager.settings.offset = Math.max(400, Math.round(viewer.clientHeight || 400));
    };

    // 等待连续管理器完成相邻 spine 的挂载后，把目标 spine 直接对齐到阅读窗口顶部。
    const focusDisplayedSection = async (section, target = "") => {
        if (!section || !scrollContainer) return;
        for (let attempt = 0; attempt < 12; attempt += 1) {
            await new Promise((resolve) => requestAnimationFrame(resolve));
            let targetView;
            const views = rendition?.views?.();
            views?.forEach?.((view) => {
                if (view?.section?.index === section.index || view?.index === section.index) targetView = view;
            });
            if (!targetView?.element) continue;
            let top = targetView.element.offsetTop;
            if (target.includes("#") && typeof targetView.locationOf === "function") {
                const location = targetView.locationOf(target);
                if (Number.isFinite(location?.top)) top += Math.max(0, location.top);
            }
            scrollContainer.scrollTo({top: Math.max(0, top), behavior: "auto"});
        }
    };

    const displayTarget = async (href, sourceHref = "") => {
        const rawTarget = resolveInternalTarget(href, sourceHref);
        if (!rawTarget || failed || disposed || !rendition) return;
        const [targetPath, targetHash = ""] = rawTarget.split("#", 2);
        const section = book?.spine?.get(targetPath) || book?.spine?.get(rawTarget);
        // 无法映射到 OPF spine 的内部链接只在当前视图中降级，不能让整个阅读器进入失败状态。
        if (!section) {
            console.warn("EPUB internal target not found", {href, sourceHref, target: rawTarget});
            navigationSpineIndex = null;
            return;
        }
        const target = `${section.href}${targetHash ? `#${targetHash}` : ""}`;
        navigationSpineIndex = Number.isInteger(section?.index) ? section.index : null;
        try {
            await rendition.display(target);
            await focusDisplayedSection(section, target);
            if (navigationSpineIndex !== null) activeSpineIndex = navigationSpineIndex;
            navigationSpineIndex = null;
            schedulePageUpdate();
        } catch (error) {
            navigationSpineIndex = null;
            fail(error);
        }
    };

    // 直接显示相邻 spine 的起点，上一页/下一页不再模拟窗口分页。
    const goToSpine = async (value) => {
        if (disposed || failed || !book || !rendition) return;
        const index = Number(value) - 1;
        const count = Number(book.spine?.length || 0);
        if (!Number.isInteger(index) || index < 0 || index >= count) {
            schedulePageUpdate();
            return;
        }
        const section = book.spine.get(index);
        if (!section) return;
        navigationSpineIndex = index;
        try {
            await rendition.display(section.href);
            await focusDisplayedSection(section, section.href);
            activeSpineIndex = index;
            navigationSpineIndex = null;
            schedulePageUpdate();
        } catch (error) {
            navigationSpineIndex = null;
            fail(error);
        }
    };
    const updatePageControls = () => {
        updateFrame = undefined;
        if (disposed || failed) return;
        updateActiveSpine();
        const spineCount = Math.max(1, Number(book?.spine?.length || 1));
        const currentSpine = Math.min(spineCount, Math.max(1, (activeSpineIndex ?? 0) + 1));
        pageInput.disabled = previous.disabled = next.disabled = false;
        pageInput.max = String(spineCount);
        pageInput.value = String(currentSpine);
        pagesOutput.textContent = String(spineCount);
        previous.disabled = currentSpine <= 1;
        next.disabled = currentSpine >= spineCount;
        status.textContent = `${currentSpine} / ${spineCount}`;
    };

    const schedulePageUpdate = () => {
        if (updateFrame === undefined) updateFrame = requestAnimationFrame(updatePageControls);
    };

    // 在未挂载的章节 DOM 中注入 CSP，并绑定统一的 EPUB 内部链接导航。
    const sanitizeDocument = (contents) => {
        const candidates = [contents?.document, contents?.ownerDocument, contents];
        const doc = candidates.find((candidate) => candidate && typeof candidate.querySelectorAll === "function");
        if (!doc || typeof doc.querySelectorAll !== "function") throw new Error("Invalid EPUB chapter document");
        doc.querySelectorAll("script, iframe, object, embed, form, video, audio, source, meta[http-equiv]")
            .forEach((node) => node.remove());
        doc.querySelectorAll("*").forEach((node) => {
            [...node.attributes].forEach((attribute) => {
                const name = attribute.name.toLowerCase();
                const value = attribute.value.trim();
                if (name.startsWith("on") || ["srcdoc", "ping", "target"].includes(name)) {
                    node.removeAttribute(attribute.name);
                } else if (["src", "href", "xlink:href", "poster", "action"].includes(name)
                    && /^(?:[a-z][a-z\d+.-]*:|\/\/)/i.test(value)) {
                    const localName = node.localName?.toLowerCase();
                    const imageResource = (localName === "img" && name === "src")
                        || (localName === "image" && ["href", "xlink:href"].includes(name));
                    const stylesheetResource = localName === "link" && name === "href"
                        && /(?:^|\s)stylesheet(?:\s|$)/i.test(node.getAttribute("rel") || "");
                    const embeddedResource = /^(?:blob:|data:)/i.test(value) && (imageResource || stylesheetResource);
                    if (!embeddedResource) {
                        node.removeAttribute(attribute.name);
                    }
                }
            });
        });
        const head = doc.querySelector("head");
        if (!head) throw new Error("Invalid EPUB chapter: missing head");
        const policy = doc.createElementNS("http://www.w3.org/1999/xhtml", "meta");
        policy.setAttribute("http-equiv", "Content-Security-Policy");
        policy.setAttribute("content", "default-src 'none'; script-src 'none'; style-src 'unsafe-inline' blob: data:; " +
            "img-src blob: data:; font-src blob: data:; frame-src 'none'; object-src 'none'; form-action 'none'");
        head.prepend(policy);
    };

    try {
        const url = new URL(params.get("book"));
        if (url.origin !== location.origin || !url.pathname.includes("/gradio_api/file=")) {
            throw new Error("Invalid EPUB file URL");
        }
        const response = await fetch(url, {credentials: "same-origin", signal: AbortSignal.timeout(30000)});
        if (!response.ok) throw new Error(`EPUB fetch failed: ${response.status}`);
        const payload = await response.arrayBuffer();
        if (payload.byteLength > 128 * 1024 * 1024) throw new Error("EPUB exceeds preview size limit");
        const archive = await JSZip.loadAsync(payload);
        if (archive.file("META-INF/encryption.xml")) throw new Error("Encrypted EPUB preview is unsupported");
        book = ePub();
        rendition = book.renderTo(viewer, {
            manager: "continuous",
            flow: "scrolled-continuous",
            width: "100%",
            height: "100%",
            spread: "none",
            allowScriptedContent: false,
            allowPopups: false,
            // 初始只加载当前 spine 附近内容，具体范围会在 viewport 建立后动态更新。
            offset: 800,
        });
        // 只保留 iframe 内的安全清理；EPUB.js 的原始 spine hook 依赖 XMLDocument.createElement，
        // 在浏览器沙箱的 XMLDocument 实现中会阻断后续章节加载。
        book.spine.hooks.content.clear();
        // Safari 会阻止父页面向只有 allow-same-origin 的 srcdoc iframe 派发点击事件；
        // 这里仅为事件兼容补上 allow-scripts，章节脚本仍由清理逻辑和 CSP 的 script-src 'none' 禁止。
        const enableChapterIframeEvents = () => {
            viewer.querySelectorAll(".epub-view > iframe").forEach((frame) => {
                const sandbox = frame.getAttribute("sandbox") || "";
                if (!/\ballow-scripts\b/.test(sandbox)) {
                    frame.setAttribute("sandbox", `${sandbox} allow-scripts`.trim());
                }
            });
        };
        const chapterFrameObserver = typeof MutationObserver === "function"
            ? new MutationObserver(enableChapterIframeEvents)
            : null;
        chapterFrameObserver?.observe(viewer, {childList: true, subtree: true});
        enableChapterIframeEvents();
        const bindScrollContainer = () => {
            const nextScrollContainer = viewer.querySelector(".epub-container") || viewer;
            if (nextScrollContainer === observedScrollContainer) return;
            observedScrollContainer?.removeEventListener("scroll", schedulePageUpdate);
            observedScrollContainer = nextScrollContainer;
            scrollContainer = nextScrollContainer;
            scrollContainer.addEventListener("scroll", schedulePageUpdate, {passive: true});
            resizeObserver?.observe(scrollContainer);
            updateLoadWindow();
            schedulePageUpdate();
        };
        rendition.hooks.content.register((contents) => {
            sanitizeDocument(contents.document);
            // EPUB.js Contents 使用 sectionIndex 标识所属 spine；不能使用不存在的 contents.index。
            const section = Number.isInteger(contents?.sectionIndex) ? book.spine.get(contents.sectionIndex) : null;
            const sectionHref = section?.href || contents.section?.href || contents.href || "";
            if (contents.document?.defaultView) {
                contents.document.defaultView.__mineruEpubSectionHref = sectionHref;
            }
            contents.document?.addEventListener("click", (event) => {
                const anchor = event.target?.closest?.("a[href]");
                if (!anchor) return;
                const href = anchor.getAttribute("href") || "";
                event.preventDefault();
                event.stopImmediatePropagation();
                if (!isExternalLink(href)) void displayTarget(href, sectionHref);
            }, true);
            if (contents.document) {
                contents.document.documentElement?.setAttribute("data-mineru-links-bound", "true");
            }
            bindScrollContainer();
            updateLoadWindow();
            schedulePageUpdate();
        });
        window.addEventListener("pagehide", () => {
            disposed = true;
            cancelAnimationFrame(updateFrame);
            chapterFrameObserver?.disconnect();
            book.destroy();
        }, {once: true});
        resizeObserver = new ResizeObserver(() => {
            bindScrollContainer();
            updateLoadWindow();
            schedulePageUpdate();
        });
        resizeObserver.observe(viewer);
        bindScrollContainer();
        previous.addEventListener("click", () => goToSpine(Number(pageInput.value) - 1));
        next.addEventListener("click", () => goToSpine(Number(pageInput.value) + 1));
        pageInput.addEventListener("change", () => goToSpine(pageInput.value));
        pageInput.addEventListener("keydown", (event) => { if (event.key === "Enter") goToSpine(pageInput.value); });
        rendition.on("displayerror", fail);
        rendition.on("relocated", schedulePageUpdate);
        const appendContents = (items, depth = 0) => {
            for (const item of items) {
                const href = item.href || "";
                if (href && !isExternalLink(href)) {
                    const option = document.createElement("option");
                    option.value = href;
                    option.textContent = `${"\u00a0\u00a0".repeat(Math.min(depth, 10))}${item.label || href}`;
                    toc.appendChild(option);
                }
                appendContents(item.subitems || [], depth + 1);
            }
        };
        toc.addEventListener("change", () => displayTarget(toc.value));
        await book.open(payload, "binary");
        // 导航目录在 book.open() 后才由 EPUB 包的 navigation 文档填充，不能提前读取。
        await book.loaded.navigation;
        appendContents(book.navigation?.toc || []);
        toc.hidden = toc.options.length === 0;
        await displayTarget(book.spine.first()?.href || "");
        await new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));
        updateLoadWindow();
        updateActiveSpine();
        schedulePageUpdate();
    } catch (error) {
        fail(error);
    }
})();
