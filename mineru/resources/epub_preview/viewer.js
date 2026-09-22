// 可信阅读器脚本独立于书籍内容；章节 iframe 始终禁止脚本执行。
(async () => {
    const params = new URLSearchParams(window.location.search);
    const messages = JSON.parse(params.get("messages") || "{}");
    const status = document.getElementById("status");
    const viewer = document.getElementById("viewer");
    const toc = document.getElementById("toc");
    const sectionInput = document.getElementById("section");
    const sectionsOutput = document.getElementById("sections");
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
    let pendingSpineIndex = null;
    let navigationSequence = 0;
    let navigationQueue = Promise.resolve();
    const SPINE_SWITCH_RATIO = 0.5;
    const MAX_EPUB_BYTES = 128 * 1024 * 1024;
    const SANITIZED_MARKER = "data-mineru-sanitized";
    const SANITIZED_TOKEN = Array.from(crypto.getRandomValues(new Uint32Array(4)), (value) => value.toString(16)).join("-");
    const FONT_OBFUSCATION_ALGORITHMS = new Set([
        "http://www.idpf.org/2008/embedding",
        "http://ns.adobe.com/pdf/enc#RC",
    ]);

    const fail = (error) => {
        failed = true;
        status.textContent = messages.epub_preview_failed || "Could not load EPUB preview.";
        status.setAttribute("role", "alert");
        sectionInput.disabled = previous.disabled = next.disabled = toc.disabled = true;
        console.warn("EPUB preview failed", error);
    };

    previous.title = previous.ariaLabel = messages.epub_previous || "Previous section";
    next.title = next.ariaLabel = messages.epub_next || "Next section";
    sectionInput.setAttribute("aria-label", messages.epub_spine || "Section");
    toc.ariaLabel = messages.epub_contents || "Contents";
    status.textContent = messages.epub_loading || "Loading EPUB…";
    sectionInput.disabled = previous.disabled = next.disabled = true;

    const isExternalLink = (href) => /^(?:[a-z][a-z\d+.-]*:|\/\/)/i.test(href);

    // 把章节内相对链接归一化为 EPUB 包内路径，保证 footnote 和跨章节链接使用同一入口。
    const resolveInternalTarget = (href, sourceHref = "") => {
        const raw = String(href || "").trim();
        if (!raw || isExternalLink(raw)) return null;
        const base = new URL(sourceHref || "/", "https://mineru.invalid/");
        const resolved = new URL(raw, base);
        return `${resolved.pathname.replace(/^\/+/, "")}${resolved.search}${resolved.hash}`;
    };

    // 根据当前视口内各个 spine 的可见比例更新顶部编号，避免长章节滚动时误切章节。
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

    // 等待连续管理器完成目标 spine 挂载；最终只滚动一次，锚点暂不可定位时回退到章节顶部。
    const focusDisplayedSection = async (section, target = "", navigationId = null) => {
        if (!section || !scrollContainer) return false;
        let fallbackTop = null;
        let resolvedTop = null;
        for (let attempt = 0; attempt < 12; attempt += 1) {
            await new Promise((resolve) => requestAnimationFrame(resolve));
            if (navigationId !== null && navigationId !== navigationSequence) return false;
            let targetView;
            const views = rendition?.views?.();
            views?.forEach?.((view) => {
                if (view?.section?.index === section.index || view?.index === section.index) targetView = view;
            });
            if (!targetView?.element) continue;
            fallbackTop = targetView.element.offsetTop;
            let top = fallbackTop;
            if (target.includes("#") && typeof targetView.locationOf === "function") {
                const location = targetView.locationOf(target);
                if (!Number.isFinite(location?.top)) continue;
                top += Math.max(0, location.top);
            }
            resolvedTop = top;
            break;
        }
        const top = resolvedTop ?? fallbackTop;
        if (!Number.isFinite(top)) return false;
        scrollContainer.scrollTo({top: Math.max(0, top), behavior: "auto"});
        return true;
    };

    // 串行执行章节导航，并用递增序号淘汰旧请求，避免快速点击时旧 display 结果覆盖最后一次操作。
    const enqueueNavigation = (section, target) => {
        if (!section || failed || disposed || !rendition) return Promise.resolve();
        const navigationId = ++navigationSequence;
        pendingSpineIndex = Number.isInteger(section.index) ? section.index : null;
        scheduleSectionUpdate();
        navigationQueue = navigationQueue.catch(() => undefined).then(async () => {
            if (navigationId !== navigationSequence || failed || disposed) return;
            try {
                await rendition.display(target);
                if (navigationId !== navigationSequence || failed || disposed) return;
                await focusDisplayedSection(section, target, navigationId);
                if (navigationId !== navigationSequence || failed || disposed) return;
                activeSpineIndex = Number.isInteger(section.index) ? section.index : activeSpineIndex;
                pendingSpineIndex = null;
                scheduleSectionUpdate();
            } catch (error) {
                if (navigationId !== navigationSequence || disposed) return;
                pendingSpineIndex = null;
                fail(error);
            }
        });
        return navigationQueue;
    };

    // 将目录、脚注等包内链接统一映射为 spine，再交给串行导航入口处理。
    const displayTarget = async (href, sourceHref = "") => {
        const rawTarget = resolveInternalTarget(href, sourceHref);
        if (!rawTarget || failed || disposed || !rendition) return;
        const [targetPath, targetHash = ""] = rawTarget.split("#", 2);
        const section = book?.spine?.get(targetPath) || book?.spine?.get(rawTarget);
        // 无法映射到 OPF spine 的内部链接只在当前视图中降级，不能让整个阅读器进入失败状态。
        if (!section) {
            console.warn("EPUB internal target not found", {href, sourceHref, target: rawTarget});
            return;
        }
        const target = `${section.href}${targetHash ? `#${targetHash}` : ""}`;
        await enqueueNavigation(section, target);
    };

    // 直接显示相邻 spine 的起点，上一节/下一节不再模拟窗口分页。
    const goToSpine = async (value) => {
        if (disposed || failed || !book || !rendition) return;
        const index = Number(value) - 1;
        const count = Number(book.spine?.length || 0);
        if (!Number.isInteger(index) || index < 0 || index >= count) {
            scheduleSectionUpdate();
            return;
        }
        const section = book.spine.get(index);
        if (!section) return;
        await enqueueNavigation(section, section.href);
    };

    // 根据当前可见或待导航 spine 刷新章节编号和按钮状态。
    const updateSectionControls = () => {
        updateFrame = undefined;
        if (disposed || failed) return;
        if (pendingSpineIndex === null) updateActiveSpine();
        const spineCount = Math.max(1, Number(book?.spine?.length || 1));
        const currentIndex = pendingSpineIndex ?? activeSpineIndex ?? 0;
        const currentSpine = Math.min(spineCount, Math.max(1, currentIndex + 1));
        sectionInput.disabled = previous.disabled = next.disabled = false;
        sectionInput.max = String(spineCount);
        sectionInput.value = String(currentSpine);
        sectionsOutput.textContent = String(spineCount);
        previous.disabled = currentSpine <= 1;
        next.disabled = currentSpine >= spineCount;
        status.textContent = `${messages.epub_spine || "Section"} ${currentSpine} / ${spineCount}`;
    };

    const scheduleSectionUpdate = () => {
        if (updateFrame === undefined) updateFrame = requestAnimationFrame(updateSectionControls);
    };

    // 在章节序列化进 iframe 前移除主动内容和远程资源，并注入严格 CSP。
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
                    if (!embeddedResource) node.removeAttribute(attribute.name);
                }
            });
        });
        const head = doc.querySelector("head");
        if (!head) throw new Error("Invalid EPUB chapter: missing head");
        const namespace = doc.documentElement?.namespaceURI || "http://www.w3.org/1999/xhtml";
        const policy = doc.createElementNS(namespace, "meta");
        policy.setAttribute("http-equiv", "Content-Security-Policy");
        policy.setAttribute("content", "default-src 'none'; script-src 'none'; style-src 'unsafe-inline' blob: data:; " +
            "img-src blob: data:; font-src blob: data:; frame-src 'none'; object-src 'none'; form-action 'none'");
        head.prepend(policy);
        doc.documentElement?.setAttribute(SANITIZED_MARKER, SANITIZED_TOKEN);
        return doc;
    };

    // 用 namespace-safe 的实现恢复 EPUB.js 默认 base/canonical/identifier 元数据，避免 XMLDocument.createElement 的兼容问题。
    const restoreSpineMetadata = (doc, section) => {
        const head = doc.querySelector("head");
        if (!head) throw new Error("Invalid EPUB chapter: missing head");
        const namespace = doc.documentElement?.namespaceURI || "http://www.w3.org/1999/xhtml";
        const createElement = (name) => doc.createElementNS(namespace, name);

        let base = head.querySelector("base");
        if (!base) {
            base = createElement("base");
            head.appendChild(base);
        }
        let sectionUrl = String(section?.url || "");
        if (sectionUrl && !sectionUrl.includes("://")) sectionUrl = new URL(sectionUrl, location.origin).href;
        if (sectionUrl) base.setAttribute("href", sectionUrl);

        let canonical = head.querySelector('link[rel="canonical"]');
        if (!canonical) {
            canonical = createElement("link");
            canonical.setAttribute("rel", "canonical");
            head.appendChild(canonical);
        }
        if (section?.canonical) canonical.setAttribute("href", String(section.canonical));

        let identifier = head.querySelector('meta[name="dc.identifier"], meta[property="dc.identifier"], link[property="dc.identifier"]');
        if (!identifier) {
            identifier = createElement("meta");
            identifier.setAttribute("name", "dc.identifier");
            head.appendChild(identifier);
        }
        if (section?.idref) identifier.setAttribute("content", String(section.idref));
    };

    // 替换 EPUB.js 依赖 XMLDocument.createElement 的默认 spine hooks，同时保留其元数据语义并确保安全清理发生在 iframe 创建前。
    const installSafeSpineContentHook = (currentBook) => {
        currentBook.spine.hooks.content.clear();
        currentBook.spine.hooks.content.register((document, section) => {
            const doc = sanitizeDocument(document);
            restoreSpineMetadata(doc, section);
        });
    };

    // 读取响应时实时限制字节数，避免超大 EPUB 在完整 arrayBuffer 分配后才被拒绝。
    const readResponseWithLimit = async (response, maxBytes) => {
        const lengthHeader = response.headers.get("content-length");
        const declaredSize = lengthHeader ? Number(lengthHeader) : Number.NaN;
        if (Number.isFinite(declaredSize) && declaredSize > maxBytes) {
            await response.body?.cancel?.();
            throw new Error("EPUB exceeds preview size limit");
        }
        const reader = response.body?.getReader?.();
        if (!reader) {
            const payload = await response.arrayBuffer();
            if (payload.byteLength > maxBytes) throw new Error("EPUB exceeds preview size limit");
            return payload;
        }
        const chunks = [];
        let total = 0;
        while (true) {
            const {done, value} = await reader.read();
            if (done) break;
            total += value.byteLength;
            if (total > maxBytes) {
                await reader.cancel();
                throw new Error("EPUB exceeds preview size limit");
            }
            chunks.push(value);
        }
        const bytes = new Uint8Array(total);
        let offset = 0;
        for (const chunk of chunks) {
            bytes.set(chunk, offset);
            offset += chunk.byteLength;
        }
        return bytes.buffer;
    };

    // encryption.xml 也用于 EPUB 标准字体混淆；仅拒绝其中出现的未知/真实加密算法。
    const ensureSupportedEncryption = async (currentBook) => {
        const encryption = await currentBook.archive?.getText?.("/META-INF/encryption.xml");
        if (!encryption) return;
        const document = new DOMParser().parseFromString(encryption, "application/xml");
        const elements = Array.from(document.getElementsByTagName("*"));
        if (elements.some((element) => element.localName === "parsererror")) {
            throw new Error("Invalid EPUB encryption metadata");
        }
        const encryptedItems = elements.filter((element) => element.localName === "EncryptedData");
        const unsupported = encryptedItems.map((item) => {
            const method = Array.from(item.getElementsByTagName("*"))
                .find((element) => element.localName === "EncryptionMethod");
            return method?.getAttribute("Algorithm")?.trim() || "";
        }).filter((algorithm) => !FONT_OBFUSCATION_ALGORITHMS.has(algorithm));
        if (unsupported.length) throw new Error("Encrypted EPUB preview is unsupported");
    };

    try {
        const url = new URL(params.get("book"));
        if (url.origin !== location.origin || !url.pathname.includes("/gradio_api/file=")) {
            throw new Error("Invalid EPUB file URL");
        }
        const response = await fetch(url, {credentials: "same-origin", signal: AbortSignal.timeout(30000)});
        if (!response.ok) throw new Error(`EPUB fetch failed: ${response.status}`);
        const payload = await readResponseWithLimit(response, MAX_EPUB_BYTES);
        book = ePub();
        await book.open(payload, "binary");
        await ensureSupportedEncryption(book);
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
        installSafeSpineContentHook(book);
        // Safari 的章节点击兼容仅对已经在序列化前清理且带严格 CSP 的 srcdoc 开放脚本 sandbox 能力。
        const enableChapterIframeEvents = () => {
            viewer.querySelectorAll(".epub-view > iframe").forEach((frame) => {
                const srcdoc = frame.getAttribute("srcdoc") || "";
                if (!srcdoc.includes(`${SANITIZED_MARKER}="${SANITIZED_TOKEN}"`) || !srcdoc.includes("script-src 'none'")) return;
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
            observedScrollContainer?.removeEventListener("scroll", scheduleSectionUpdate);
            observedScrollContainer = nextScrollContainer;
            scrollContainer = nextScrollContainer;
            scrollContainer.addEventListener("scroll", scheduleSectionUpdate, {passive: true});
            resizeObserver?.observe(scrollContainer);
            updateLoadWindow();
            scheduleSectionUpdate();
        };
        rendition.hooks.content.register((contents) => {
            if (contents.document?.documentElement?.getAttribute(SANITIZED_MARKER) !== SANITIZED_TOKEN) {
                throw new Error("EPUB chapter reached iframe without sanitization");
            }
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
            scheduleSectionUpdate();
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
            scheduleSectionUpdate();
        });
        resizeObserver.observe(viewer);
        bindScrollContainer();
        previous.addEventListener("click", () => goToSpine(Number(sectionInput.value) - 1));
        next.addEventListener("click", () => goToSpine(Number(sectionInput.value) + 1));
        sectionInput.addEventListener("change", () => goToSpine(sectionInput.value));
        sectionInput.addEventListener("keydown", (event) => { if (event.key === "Enter") goToSpine(sectionInput.value); });
        rendition.on("displayerror", fail);
        rendition.on("relocated", scheduleSectionUpdate);
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
        // 导航目录在 book.open() 后才由 EPUB 包的 navigation 文档填充，不能提前读取。
        await book.loaded.navigation;
        appendContents(book.navigation?.toc || []);
        toc.hidden = toc.options.length === 0;
        await displayTarget(book.spine.first()?.href || "");
        await new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));
        updateLoadWindow();
        updateActiveSpine();
        scheduleSectionUpdate();
    } catch (error) {
        fail(error);
    }
})();
