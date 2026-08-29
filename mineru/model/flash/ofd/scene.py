# Copyright (c) Opendatalab. All rights reserved.
"""把 OFD 文档、模板、图层与资源组合成逐页场景。"""

from __future__ import annotations

from dataclasses import replace

from loguru import logger
from lxml import etree  # type: ignore[reportMissingImports]

from ....types import BBox
from .constants import MAX_OBJECT_RECURSION, MAX_PAGE_COUNT, OFD_NAMESPACES
from .errors import OfdParseError, OfdResourceLimitError
from .geometry import Affine, bbox_intersection, parse_affine, parse_st_box, transform_bbox
from .images import build_image_item
from .models import OfdPageScene, PageBuildContext, PageRef, ResourceRegistry, TemplateRef
from .package import OfdPackage, element_text, first_child, first_descendant, local_name, namespace_name, parse_int
from .path import OfdPathBudget, build_axis_lines
from .resources import merge_registries, parse_resource_part, resolve_draw_param
from .text import FontMetricResolver, OfdTextBudget, build_text_lines


class OfdSceneBuilder:
    """按文档页树构造 OFD 页面场景。"""

    def __init__(self, package: OfdPackage) -> None:
        """绑定当前包和跨页共享预算。"""
        self.package = package
        self.text_budget = OfdTextBudget()
        self.path_budget = OfdPathBudget()
        self.font_metrics = FontMetricResolver(package)
        self._paint_order = 0
        self._page_count = 0

    def build(self) -> list[OfdPageScene]:
        """按 DocBody 和 Pages 声明顺序构造全部页面。"""
        scenes: list[OfdPageScene] = []
        try:
            for document_ref in self.package.document_refs():
                document_root = self._required_ofd_xml(document_ref.document_part)
                scenes.extend(self._build_document(document_ref.document_part, document_root, len(scenes)))
            return scenes
        finally:
            self.font_metrics.close()

    def _required_ofd_xml(self, part_name: str) -> etree._Element:
        """读取必需核心 XML 并限制为已知 OFD 命名空间。"""
        root = self.package.xml_part(part_name, required=True)
        assert root is not None
        namespace = namespace_name(root.tag)
        if namespace not in OFD_NAMESPACES:
            raise OfdParseError(f"Malformed OFD package: unsupported namespace {namespace!r} in {part_name!r}")
        return root

    def _resolve_part(self, base_part: str, element: etree._Element | None, *, required: bool = False) -> str | None:
        """解析一个 ST_Loc 元素并按需要求结果存在。"""
        part = self.package.resolve_reference(base_part, element_text(element)) if element is not None else None
        if required and (part is None or not self.package.has_part(part)):
            raise OfdParseError(f"Malformed OFD package: invalid required location from {base_part!r}")
        return part

    def _document_resources(self, document_part: str, document_root: etree._Element) -> ResourceRegistry:
        """解析文档级 PublicRes 与 DocumentRes。"""
        common_data = first_descendant(document_root, "CommonData")
        public_part = self._resolve_part(document_part, first_child(common_data, "PublicRes"))
        document_resource_part = self._resolve_part(document_part, first_child(common_data, "DocumentRes"))
        return merge_registries(
            parse_resource_part(self.package, public_part),
            parse_resource_part(self.package, document_resource_part),
        )

    def _page_area(self, root: etree._Element, fallback: BBox | None, name: str) -> BBox | None:
        """读取页面区域，缺失或非法时返回文档级回退。"""
        area = first_descendant(root, "Area")
        if area is None:
            area = first_descendant(root, "PageArea")
        value = parse_st_box(element_text(first_descendant(area, name))) if area is not None else None
        return value or fallback

    def _template_refs(self, document_part: str, document_root: etree._Element) -> dict[int, TemplateRef]:
        """读取 CommonData 中的模板页面映射。"""
        common_data = first_descendant(document_root, "CommonData")
        result: dict[int, TemplateRef] = {}
        if common_data is None:
            return result
        for element in common_data.iter():
            if local_name(element.tag) != "TemplatePage":
                continue
            template_id = parse_int(element.get("ID"))
            page_part = self.package.resolve_reference(document_part, element.get("BaseLoc"))
            if template_id is not None and page_part is not None:
                result[template_id] = TemplateRef(template_id=template_id, page_part=page_part)
        return result

    def _page_refs(self, document_part: str, document_root: etree._Element) -> list[PageRef]:
        """按 Document.xml 的 Pages 子节点顺序读取页面引用。"""
        pages = first_descendant(document_root, "Pages")
        if pages is None:
            raise OfdParseError(f"Malformed OFD package: {document_part!r} has no Pages")
        result: list[PageRef] = []
        for element in pages:
            if local_name(element.tag) != "Page":
                continue
            self._page_count += 1
            if self._page_count > MAX_PAGE_COUNT:
                raise OfdResourceLimitError(f"OFD resource limit exceeded: max_page_count={MAX_PAGE_COUNT}")
            page_part = self.package.resolve_reference(document_part, element.get("BaseLoc"))
            if page_part is None or not self.package.has_part(page_part):
                raise OfdParseError(f"Malformed OFD package: invalid page BaseLoc in {document_part!r}")
            result.append(PageRef(page_id=parse_int(element.get("ID")), page_part=page_part))
        return result

    def _build_document(
        self,
        document_part: str,
        document_root: etree._Element,
        page_index_offset: int,
    ) -> list[OfdPageScene]:
        """构造单个 Document.xml 声明的全部页面。"""
        common_data = first_descendant(document_root, "CommonData")
        default_physical = self._page_area(common_data, None, "PhysicalBox") if common_data is not None else None
        default_content = self._page_area(common_data, None, "ContentBox") if common_data is not None else None
        document_resources = self._document_resources(document_part, document_root)
        templates = self._template_refs(document_part, document_root)
        scenes: list[OfdPageScene] = []
        for local_page_idx, page_ref in enumerate(self._page_refs(document_part, document_root)):
            page_root = self._required_ofd_xml(page_ref.page_part)
            physical_box = self._page_area(page_root, default_physical, "PhysicalBox")
            if physical_box is None:
                raise OfdParseError(f"Malformed OFD package: page {page_ref.page_part!r} has no PhysicalBox")
            content_box = self._page_area(page_root, default_content, "ContentBox")
            page_resource_part = self._resolve_part(page_ref.page_part, first_descendant(page_root, "PageRes"))
            resources = merge_registries(document_resources, parse_resource_part(self.package, page_resource_part))
            scene = OfdPageScene(
                page_idx=page_index_offset + local_page_idx,
                physical_box=physical_box,
                content_box=content_box,
            )
            base_context = PageBuildContext(
                transform=Affine(),
                clip_bbox=physical_box,
                layer_type="Body",
                template_id=None,
                draw_style={},
            )
            template_uses = [element for element in page_root.iter() if local_name(element.tag) == "Template"]
            for template_use in template_uses:
                if (template_use.get("ZOrder") or "Background").casefold() == "foreground":
                    continue
                self._append_template(template_use, templates, resources, scene, base_context)
            content = first_descendant(page_root, "Content")
            if content is not None:
                self._walk(content, base_context, resources, scene, depth=0, composite_stack=frozenset())
            for template_use in template_uses:
                if (template_use.get("ZOrder") or "Background").casefold() != "foreground":
                    continue
                self._append_template(template_use, templates, resources, scene, base_context)
            scenes.append(scene)
        return scenes

    def _append_template(
        self,
        template_use: etree._Element,
        templates: dict[int, TemplateRef],
        resources: ResourceRegistry,
        scene: OfdPageScene,
        base_context: PageBuildContext,
    ) -> None:
        """把一个页面引用的模板内容合并到当前场景。"""
        template_id = parse_int(template_use.get("TemplateID"))
        template_ref = templates.get(template_id) if template_id is not None else None
        if template_ref is None:
            logger.warning(f"OFD_TEMPLATE_MISSING: template_id={template_id}")
            return
        template_root = self._required_ofd_xml(template_ref.page_part)
        template_resource_part = self._resolve_part(template_ref.page_part, first_descendant(template_root, "PageRes"))
        template_resources = merge_registries(resources, parse_resource_part(self.package, template_resource_part))
        content = first_descendant(template_root, "Content")
        if content is None:
            return
        context = replace(base_context, template_id=template_id)
        self._walk(content, context, template_resources, scene, depth=0, composite_stack=frozenset())

    def _child_context(self, element: etree._Element, context: PageBuildContext) -> PageBuildContext | None:
        """为 PageBlock 或 CompositeObject 计算子级变换和裁剪。"""
        boundary = parse_st_box(element.get("Boundary"))
        if boundary is None:
            return context
        boundary_page = transform_bbox(boundary, context.transform)
        if boundary_page is None:
            return None
        clip = bbox_intersection(context.clip_bbox, boundary_page)
        if clip is None:
            return None
        transform = context.transform.compose(Affine.translation(boundary[0], boundary[1])).compose(
            parse_affine(element.get("CTM"))
        )
        return replace(context, transform=transform, clip_bbox=clip)

    def _resolved_style(
        self,
        element: etree._Element,
        context: PageBuildContext,
        resources: ResourceRegistry,
    ) -> dict[str, str]:
        """按父上下文、DrawParam 和对象直接属性解析最终样式。"""
        style = dict(context.draw_style)
        try:
            style.update(resolve_draw_param(resources, parse_int(element.get("DrawParam"))))
        except OfdResourceLimitError:
            raise
        except ValueError as exc:
            raise OfdParseError(str(exc)) from exc
        style.update({str(key): str(value) for key, value in element.attrib.items()})
        return style

    def _walk(
        self,
        element: etree._Element,
        context: PageBuildContext,
        resources: ResourceRegistry,
        scene: OfdPageScene,
        *,
        depth: int,
        composite_stack: frozenset[int],
    ) -> None:
        """按绘制顺序递归展开页面、模板和复合图元对象。"""
        if depth > MAX_OBJECT_RECURSION:
            raise OfdResourceLimitError(f"OFD resource limit exceeded: max_object_recursion={MAX_OBJECT_RECURSION}")
        for child in element:
            if not isinstance(child.tag, str):
                continue
            name = local_name(child.tag)
            child_context = context
            if name == "Layer":
                child_context = replace(
                    context,
                    layer_type=(child.get("Type") or "Body"),
                    draw_style=self._resolved_style(child, context, resources),
                )
            elif name == "PageBlock":
                resolved_context = self._child_context(child, context)
                if resolved_context is None:
                    continue
                child_context = replace(
                    resolved_context,
                    draw_style=self._resolved_style(child, context, resources),
                )
            if name == "TextObject":
                lines = build_text_lines(
                    child,
                    parent_transform=context.transform,
                    parent_clip=context.clip_bbox,
                    resources=resources,
                    package=self.package,
                    font_metrics=self.font_metrics,
                    budget=self.text_budget,
                    paint_order=self._paint_order,
                    layer_type=context.layer_type,
                    template_id=context.template_id,
                    resolved_style=self._resolved_style(child, context, resources),
                )
                scene.text_lines.extend(lines)
                self._paint_order += max(1, len(lines))
                continue
            if name == "PathObject":
                scene.axis_lines.extend(
                    build_axis_lines(
                        child,
                        parent_transform=context.transform,
                        parent_clip=context.clip_bbox,
                        paint_order=self._paint_order,
                        template_id=context.template_id,
                        budget=self.path_budget,
                        resolved_style=self._resolved_style(child, context, resources),
                    )
                )
                self._paint_order += 1
                continue
            if name == "ImageObject":
                image = build_image_item(
                    child,
                    parent_transform=context.transform,
                    parent_clip=context.clip_bbox,
                    resources=resources,
                    package=self.package,
                    paint_order=self._paint_order,
                    layer_type=context.layer_type,
                    template_id=context.template_id,
                    resolved_style=self._resolved_style(child, context, resources),
                )
                if image is not None:
                    scene.images.append(image)
                self._paint_order += 1
                continue
            if name == "CompositeObject":
                resource_id = parse_int(child.get("ResourceID"))
                composite = resources.composites.get(resource_id) if resource_id is not None else None
                if composite is None:
                    logger.warning(f"OFD_COMPOSITE_MISSING: resource_id={resource_id}")
                    continue
                if resource_id in composite_stack:
                    raise OfdParseError(f"OFD CompositeObject cycle detected at id={resource_id}")
                resolved_context = self._child_context(child, context)
                if resolved_context is None:
                    continue
                resolved_context = replace(
                    resolved_context,
                    draw_style=self._resolved_style(child, context, resources),
                )
                content = first_descendant(composite.element, "Content")
                if content is not None:
                    self._walk(
                        content,
                        resolved_context,
                        resources,
                        scene,
                        depth=depth + 1,
                        composite_stack=composite_stack | {resource_id},
                    )
                continue
            self._walk(
                child,
                child_context,
                resources,
                scene,
                depth=depth + 1,
                composite_stack=composite_stack,
            )


__all__ = ["OfdSceneBuilder"]
