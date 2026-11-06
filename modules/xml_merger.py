"""
任务6：XML合并模块

功能：
    - 收集各个模块生成的XML片段
    - 按层级和位置排序
    - 合并成完整的DrawIO XML文件

重构说明：
    - 每个子模块负责生成自己的mxCell XML代码
    - XMLMerger不再负责生成样式，只负责：
        1. 收集各模块的XML片段
        2. 按层级（layer_level）排序
        3. 同层级内按面积降序排列（大的在下）
        4. 重新分配ID避免冲突
        5. 生成完整的DrawIO XML文件

作者：[你的名字]
负责任务：任务6 - 合并XML

使用示例：
    from modules import XMLMerger, ProcessingContext, XMLFragment
    
    merger = XMLMerger()
    context = ProcessingContext(image_path="test.png")
    context.canvas_width = 800
    context.canvas_height = 600
    
    # 各模块的XML片段
    context.xml_fragments = [
        XMLFragment(element_id=0, xml_content="<mxCell .../>", layer_level=1, ...),
        XMLFragment(element_id=1, xml_content="<mxCell .../>", layer_level=2, ...),
    ]
    
    result = merger.process(context)
"""

import math
import os
import re
import xml.etree.ElementTree as ET
from xml.dom import minidom
from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path

from .base import BaseProcessor, ProcessingContext
from .data_types import ElementInfo, XMLFragment, ProcessingResult, BoundingBox, LayerLevel


class XMLMerger(BaseProcessor):
    """
    XML合并模块
    
    职责：
        1. 收集各模块生成的XML片段
        2. 按层级排序（layer_level升序，小的在底层）
        3. 同层级内按面积降序排列（大的在下，避免遮挡）
        4. 重新分配mxCell的id，避免冲突
        5. 生成完整的DrawIO XML文件
    
    不负责：
        - 生成元素的样式（由各子模块负责）
        - 决定元素的颜色、描边等属性（由各子模块负责）
    
    层级规则（从底到顶）：
        0. BACKGROUND  - section_panel, title_bar等大面积容器
        1. BASIC_SHAPE - rectangle, ellipse等基本图形
        2. IMAGE       - icon, picture等图片
        3. ARROW       - 箭头/连接线
        4. TEXT        - 文字（最上层）
        5. OTHER       - 其他
    """
    
    def __init__(self, config=None):
        super().__init__(config)
    
    def process(self, context: ProcessingContext) -> ProcessingResult:
        """
        处理入口 - 合并所有XML片段
        
        Args:
            context: 处理上下文，需要包含:
                - canvas_width, canvas_height: 画布尺寸
                - xml_fragments: List[XMLFragment] 或
                - elements: List[ElementInfo]（每个元素需要有xml_fragment）
            
        Returns:
            ProcessingResult: 包含生成的XML路径
        """
        # 获取 upscale_factor，用于将画布尺寸缩放回原始尺寸
        upscale_factor = 1.0
        if hasattr(context, 'intermediate_results') and context.intermediate_results:
            upscale_factor = context.intermediate_results.get('upscale_factor', 1.0)
        
        # 计算原始画布尺寸
        canvas_width = context.canvas_width
        canvas_height = context.canvas_height
        if upscale_factor != 1.0:
            canvas_width = int(context.canvas_width / upscale_factor)
            canvas_height = int(context.canvas_height / upscale_factor)
            self._log(f"画布尺寸缩放: {context.canvas_width}x{context.canvas_height} → {canvas_width}x{canvas_height}")
        
        # 收集所有XML片段
        fragments = self._collect_fragments(context)
        
        self._log(f"开始合并XML: 共{len(fragments)}个片段")
        
        if not fragments:
            self._log("警告: 没有XML片段需要合并")
            return ProcessingResult(
                success=True,
                canvas_width=canvas_width,
                canvas_height=canvas_height,
                metadata={'output_path': None, 'fragment_count': 0}
            )
        
        # 排序片段
        sorted_fragments = self._sort_fragments(fragments)
        
        # 构建XML结构（使用原始画布尺寸）
        xml_root = self._build_xml_structure(
            canvas_width,
            canvas_height,
            sorted_fragments
        )
        
        # 格式化并保存
        output_dir = context.output_dir or "./output"
        os.makedirs(output_dir, exist_ok=True)
        
        stem = Path(context.image_path).stem if context.image_path else "merged"
        output_path = os.path.join(output_dir, f"{stem}_merged.drawio.xml")
        
        xml_content = self._prettify_xml(xml_root)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(xml_content)
        
        self._log(f"XML已保存: {output_path}")
        
        return ProcessingResult(
            success=True,
            elements=context.elements,
            xml_fragments=sorted_fragments,
            canvas_width=canvas_width,  # 使用原始画布尺寸
            canvas_height=canvas_height,  # 使用原始画布尺寸
            metadata={
                'output_path': output_path,
                'xml_content': xml_content,
                'fragment_count': len(sorted_fragments)
            }
        )
    
    def _collect_fragments(self, context: ProcessingContext) -> List[XMLFragment]:
        """
        收集所有XML片段
        
        收集顺序：
        1. 从 context.xml_fragments 获取
        2. 从 context.elements 中提取有 xml_fragment 的元素（坐标需要缩放回原始尺寸）
        3. 从 context.intermediate_results['text_xml'] 提取文字 XML（坐标保持不变）
        
        坐标缩放说明：
        - 文字处理使用原始图像，坐标基于原始尺寸
        - SAM3/图形处理使用放大后的图像，坐标需要缩放回原始尺寸
        """
        fragments = []
        
        # 获取 upscale_factor，用于将非文字元素坐标缩放回原始尺寸
        upscale_factor = 1.0
        if hasattr(context, 'intermediate_results') and context.intermediate_results:
            upscale_factor = context.intermediate_results.get('upscale_factor', 1.0)
        
        # 方式1: 直接从 xml_fragments 获取（需要缩放）
        if hasattr(context, 'xml_fragments') and context.xml_fragments:
            for frag in context.xml_fragments:
                if upscale_factor != 1.0:
                    scaled_frag = self._scale_fragment_coordinates(frag, 1.0 / upscale_factor)
                    fragments.append(scaled_frag)
                else:
                    fragments.append(frag)
        
        # 方式2: 从 elements 中提取（需要缩放）
        if context.elements:
            for elem in context.elements:
                if elem.has_xml():
                    # 检查是否已经添加过（通过element_id判断）
                    existing_ids = {f.element_id for f in fragments}
                    if elem.id not in existing_ids:
                        # 缩放 bbox
                        scaled_bbox = elem.bbox
                        if upscale_factor != 1.0:
                            scaled_bbox = BoundingBox(
                                x1=int(elem.bbox.x1 / upscale_factor),
                                y1=int(elem.bbox.y1 / upscale_factor),
                                x2=int(elem.bbox.x2 / upscale_factor),
                                y2=int(elem.bbox.y2 / upscale_factor)
                            )
                        
                        # 缩放 xml_content 中的坐标
                        xml_content = elem.xml_fragment
                        if upscale_factor != 1.0:
                            xml_content = self._scale_xml_coordinates(elem.xml_fragment, 1.0 / upscale_factor)
                        
                        fragments.append(XMLFragment(
                            element_id=elem.id,
                            xml_content=xml_content,
                            layer_level=elem.layer_level,
                            bbox=scaled_bbox,
                            element_type=elem.element_type
                        ))
        
        if upscale_factor != 1.0:
            self._log(f"非文字元素坐标已按 1/{upscale_factor:.2f} 缩放回原始尺寸")
        
        # 方式3: 从 text_xml 提取文字片段（坐标保持不变，基于原始图像）
        if hasattr(context, 'intermediate_results') and context.intermediate_results:
            text_xml = context.intermediate_results.get('text_xml')
            if text_xml:
                text_fragments = self._extract_text_fragments_from_xml(text_xml)
                fragments.extend(text_fragments)
                self._log(f"从文字XML中提取了 {len(text_fragments)} 个文字片段")

        # 自动将邻近连线的独立文字绑定为 DrawIO edgeLabel
        fragments = self.bind_text_to_edges(fragments)

        return fragments
    
    def _scale_xml_coordinates(self, xml_content: str, scale: float) -> str:
        """
        缩放 XML 字符串中的坐标
        
        支持两种格式：
        1. 普通图形（vertex）：mxGeometry 的 x, y, width, height
        2. 箭头/边（edge）：mxPoint 的 x, y（sourcePoint, targetPoint, waypoints）
        
        Args:
            xml_content: mxCell XML 字符串
            scale: 缩放因子
        """
        try:
            cell = ET.fromstring(xml_content)
            geometry = cell.find('mxGeometry')
            if geometry is not None:
                # 处理普通图形的坐标（vertex）
                if geometry.get('x') is not None:
                    geometry.set('x', str(int(float(geometry.get('x', 0)) * scale)))
                if geometry.get('y') is not None:
                    geometry.set('y', str(int(float(geometry.get('y', 0)) * scale)))
                if geometry.get('width') is not None:
                    geometry.set('width', str(int(float(geometry.get('width', 0)) * scale)))
                if geometry.get('height') is not None:
                    geometry.set('height', str(int(float(geometry.get('height', 0)) * scale)))
                
                # 处理箭头/边的坐标（edge）- mxPoint 元素
                for mxpoint in geometry.iter('mxPoint'):
                    if mxpoint.get('x') is not None:
                        mxpoint.set('x', str(int(float(mxpoint.get('x', 0)) * scale)))
                    if mxpoint.get('y') is not None:
                        mxpoint.set('y', str(int(float(mxpoint.get('y', 0)) * scale)))
            
            return ET.tostring(cell, encoding='unicode')
        except Exception as e:
            self._log(f"缩放XML坐标失败: {e}")
            return xml_content
    
    def _scale_fragment_coordinates(self, fragment: XMLFragment, scale: float) -> XMLFragment:
        """
        缩放 XMLFragment 的坐标
        """
        scaled_bbox = None
        if fragment.bbox:
            scaled_bbox = BoundingBox(
                x1=int(fragment.bbox.x1 * scale),
                y1=int(fragment.bbox.y1 * scale),
                x2=int(fragment.bbox.x2 * scale),
                y2=int(fragment.bbox.y2 * scale)
            )
        
        scaled_xml = self._scale_xml_coordinates(fragment.xml_content, scale)
        
        return XMLFragment(
            element_id=fragment.element_id,
            xml_content=scaled_xml,
            layer_level=fragment.layer_level,
            bbox=scaled_bbox,
            element_type=fragment.element_type
        )
    
    def _extract_text_fragments_from_xml(self, text_xml: str) -> List[XMLFragment]:
        """
        从 flowchart_text 生成的完整 XML 中提取 mxCell 片段
        
        flowchart_text 生成的是完整的 drawio XML 文件，
        我们需要从中提取 mxCell 元素（跳过 id=0 和 id=1 的基础元素）
        """
        fragments = []
        
        try:
            # 解析 XML
            root = ET.fromstring(text_xml)
            
            # 查找所有 mxCell 元素
            for cell in root.iter('mxCell'):
                cell_id = cell.get('id', '')
                
                # 跳过基础元素（id=0 和 id=1）
                if cell_id in ['0', '1']:
                    continue
                
                # 提取 geometry 信息来创建 bbox
                geometry = cell.find('mxGeometry')
                bbox = None
                if geometry is not None:
                    try:
                        x = float(geometry.get('x', 0))
                        y = float(geometry.get('y', 0))
                        w = float(geometry.get('width', 100))
                        h = float(geometry.get('height', 20))
                        bbox = BoundingBox(x1=int(x), y1=int(y), x2=int(x+w), y2=int(y+h))
                    except (ValueError, TypeError):
                        pass
                
                # 将 mxCell 转为字符串
                xml_content = ET.tostring(cell, encoding='unicode')
                
                # 创建 XMLFragment，文字放在最上层 (TEXT = 4)
                fragments.append(XMLFragment(
                    element_id=f"text_{cell_id}",
                    xml_content=xml_content,
                    layer_level=LayerLevel.TEXT.value,  # 文字在最上层
                    bbox=bbox,
                    element_type="text"
                ))
                
        except ET.ParseError as e:
            self._log(f"解析文字XML失败: {e}")
        except Exception as e:
            self._log(f"提取文字片段失败: {e}")
        
        return fragments

    def bind_text_to_edges(self, fragments: List[XMLFragment], max_dist: float = 60.0) -> List[XMLFragment]:
        """Auto-position standalone text fragments as DrawIO edgeLabels for nearby arrows/connectors."""
        def point_to_segment(px, py, x1, y1, x2, y2):
            dx, dy = x2 - x1, y2 - y1
            if dx == 0 and dy == 0:
                return math.hypot(px - x1, py - y1)
            t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
            proj_x = x1 + t * dx
            proj_y = y1 + t * dy
            return math.hypot(px - proj_x, py - proj_y)

        # 1. Collect edges and shape boxes
        edges = []
        shape_boxes = []
        for f in fragments:
            if 'edge="1"' in f.xml_content:
                try:
                    cell = ET.fromstring(f.xml_content)
                    edge_id = cell.get('id', '')
                    geom = cell.find('mxGeometry')
                    if geom is not None:
                        sp = geom.find('mxPoint[@as="sourcePoint"]')
                        tp = geom.find('mxPoint[@as="targetPoint"]')
                        if sp is not None and tp is not None:
                            sx, sy = float(sp.get('x', 0)), float(sp.get('y', 0))
                            tx, ty = float(tp.get('x', 0)), float(tp.get('y', 0))
                            pts = [(sx, sy)]
                            arr = geom.find('Array')
                            if arr is not None:
                                for mp in arr.iter('mxPoint'):
                                    pts.append((float(mp.get('x', 0)), float(mp.get('y', 0))))
                            pts.append((tx, ty))
                            # Calculate midpoint along path
                            seg_lens = [math.hypot(pts[i+1][0]-pts[i][0], pts[i+1][1]-pts[i][1]) for i in range(len(pts)-1)]
                            total_l = sum(seg_lens)
                            half_l = total_l / 2.0
                            acc = 0.0
                            mid_x, mid_y = pts[-1]
                            for i, slen in enumerate(seg_lens):
                                if acc + slen >= half_l:
                                    t_ratio = (half_l - acc) / max(1e-6, slen)
                                    mid_x = pts[i][0] + t_ratio * (pts[i+1][0] - pts[i][0])
                                    mid_y = pts[i][1] + t_ratio * (pts[i+1][1] - pts[i][1])
                                    break
                                acc += slen
                            edges.append({
                                'edge_id': edge_id,
                                'pts': pts,
                                'mid': (mid_x, mid_y),
                            })
                except Exception:
                    pass
            elif f.layer_level == LayerLevel.BASIC_SHAPE.value and f.bbox:
                shape_boxes.append(f.bbox)

        if not edges:
            return fragments

        # 2. Check each text fragment
        updated = []
        for f in fragments:
            if f.element_type == "text" and f.bbox and 'connectable="0"' not in f.xml_content:
                tcx = (f.bbox.x1 + f.bbox.x2) / 2.0
                tcy = (f.bbox.y1 + f.bbox.y2) / 2.0
                # Check if inside any shape box
                in_shape = any(sb.x1 <= tcx <= sb.x2 and sb.y1 <= tcy <= sb.y2 for sb in shape_boxes)
                if not in_shape:
                    best_edge = None
                    best_dist = 1e9
                    for e in edges:
                        pts = e['pts']
                        d = min(point_to_segment(tcx, tcy, pts[i][0], pts[i][1], pts[i+1][0], pts[i+1][1]) for i in range(len(pts)-1))
                        if d < best_dist:
                            best_dist = d
                            best_edge = e

                    if best_edge and best_dist <= max_dist:
                        ox = int(round(tcx - best_edge['mid'][0]))
                        oy = int(round(tcy - best_edge['mid'][1]))
                        try:
                            cell = ET.fromstring(f.xml_content)
                            cell.set('parent', best_edge['edge_id'])
                            cell.set('connectable', '0')
                            cell.set('vertex', '1')
                            cell.set('style', 'edgeLabel;html=1;align=center;verticalAlign=middle;resizable=0;points=[];')
                            geom = cell.find('mxGeometry')
                            if geom is None:
                                geom = ET.SubElement(cell, 'mxGeometry')
                            geom.attrib.clear()
                            geom.set('relative', '1')
                            geom.set('as', 'geometry')
                            off = ET.SubElement(geom, 'mxPoint')
                            off.set('x', str(ox))
                            off.set('y', str(oy))
                            off.set('as', 'offset')
                            new_xml = ET.tostring(cell, encoding='unicode')
                            f = XMLFragment(
                                element_id=f.element_id,
                                xml_content=new_xml,
                                layer_level=LayerLevel.TEXT.value,
                                bbox=f.bbox,
                                element_type="edge_label"
                            )
                        except Exception:
                            pass
            updated.append(f)
        return updated
    def _sort_fragments(self, fragments: List[XMLFragment]) -> List[XMLFragment]:
        """
        排序XML片段
        
        排序规则：
            1. 首先按 layer_level 升序（小的在底层，先写入XML）
            2. 同层级内按面积降序（大的在下，先写入）
        
        DrawIO的Z轴规则：先写入的在底层，后写入的在顶层
        """
        return sorted(
            fragments,
            key=lambda f: (f.layer_level, -f.area)  # layer升序，area降序
        )
    
    def _build_xml_structure(self, 
                             canvas_width: int,
                             canvas_height: int,
                             sorted_fragments: List[XMLFragment]) -> ET.Element:
        """
        构建完整的DrawIO XML结构
        """
        # 创建基础结构
        mxfile = self._create_base_xml(canvas_width, canvas_height)
        root_elem = mxfile.find(".//root")
        current_id = 2  # 0和1已被基础cell占用
        id_map = {}
        cells = []

        for fragment in sorted_fragments:
            cell = self._parse_and_update_cell(fragment.xml_content, current_id)
            if cell is not None:
                old_id = str(fragment.element_id)
                id_map[old_id] = str(current_id)
                try:
                    orig_cell_id = ET.fromstring(fragment.xml_content).get('id')
                    if orig_cell_id:
                        id_map[str(orig_cell_id)] = str(current_id)
                except Exception:
                    pass
                cells.append(cell)
                current_id += 1

        # 第二遍：重映射 edge 的 source/target 以及 edgeLabel 的 parent 引用
        for cell in cells:
            if cell.get("edge") == "1":
                src = cell.get("source")
                tgt = cell.get("target")
                if src in id_map:
                    cell.set("source", id_map[src])
                if tgt in id_map:
                    cell.set("target", id_map[tgt])
            elif cell.get("connectable") == "0":
                parent_id = cell.get("parent")
                if parent_id in id_map:
                    cell.set("parent", id_map[parent_id])
            root_elem.append(cell)

        return mxfile
    
    def _parse_and_update_cell(self, xml_content: str, new_id: int) -> Optional[ET.Element]:
        """
        解析XML片段并更新ID
        
        Args:
            xml_content: mxCell的XML字符串
            new_id: 新分配的ID
            
        Returns:
            更新后的ET.Element，解析失败返回None
        """
        try:
            # 清理XML字符串
            xml_content = xml_content.strip()
            
            # 如果不是以<mxCell开头，尝试提取mxCell部分
            if not xml_content.startswith('<mxCell'):
                match = re.search(r'<mxCell[^>]*>.*?</mxCell>|<mxCell[^>]*/>', 
                                  xml_content, re.DOTALL)
                if match:
                    xml_content = match.group()
                else:
                    self._log(f"警告: 无法提取mxCell: {xml_content[:100]}...")
                    return None
            
            # 解析XML
            cell = ET.fromstring(xml_content)
            
            # 更新ID
            cell.set("id", str(new_id))
            
            # 确保parent正确（除了特殊cell或edgeLabel外，都应该是"1"）
            if cell.get("connectable") != "0" and cell.get("parent") not in ["0"]:
                cell.set("parent", "1")
            return cell
            
        except ET.ParseError as e:
            self._log(f"XML解析失败: {e}, 内容: {xml_content[:100]}...")
            return None
    
    def _create_base_xml(self, canvas_width: int, canvas_height: int) -> ET.Element:
        """创建DrawIO基础XML结构"""
        mxfile = ET.Element("mxfile", {"host": "app.diagrams.net", "type": "device"})
        diagram = ET.SubElement(mxfile, "diagram", {"id": "merged", "name": "Page-1"})
        
        mx_graph_model = ET.SubElement(diagram, "mxGraphModel", {
            "dx": str(canvas_width),
            "dy": str(canvas_height),
            "grid": "1",
            "gridSize": "10",
            "guides": "1",
            "tooltips": "1",
            "connect": "1",
            "arrows": "1",
            "fold": "1",
            "page": "1",
            "pageScale": "1",
            "pageWidth": str(canvas_width),
            "pageHeight": str(canvas_height),
            "math": "0",
            "shadow": "0",
            "background": "#ffffff"
        })
        
        root = ET.SubElement(mx_graph_model, "root")
        ET.SubElement(root, "mxCell", {"id": "0"})
        ET.SubElement(root, "mxCell", {"id": "1", "parent": "0"})
        
        return mxfile
    
    def _prettify_xml(self, elem: ET.Element) -> str:
        """格式化XML输出（移除版本声明，过滤空行）"""
        rough_string = ET.tostring(elem, "utf-8")
        reparsed = minidom.parseString(rough_string)
        
        lines = reparsed.toprettyxml(indent="  ").split('\n')
        return '\n'.join([
            line for line in lines
            if line.strip() and not line.strip().startswith("<?xml")
        ])
    
    # ======================== 便捷方法 ========================
    
    def merge_xml_files(self, 
                        xml_paths: List[str],
                        output_path: str,
                        canvas_width: int,
                        canvas_height: int) -> str:
        """
        合并多个XML文件
        
        Args:
            xml_paths: XML文件路径列表
            output_path: 输出路径
            canvas_width: 画布宽度
            canvas_height: 画布高度
            
        Returns:
            输出文件路径
        """
        self._log(f"合并{len(xml_paths)}个XML文件")
        
        all_cells = []
        
        for xml_path in xml_paths:
            if not os.path.exists(xml_path):
                self._log(f"警告: 文件不存在 {xml_path}")
                continue
            
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()
                
                # 提取mxCell元素
                root_elem = root.find(".//root")
                if root_elem is not None:
                    for cell in root_elem:
                        cell_id = cell.get("id")
                        if cell_id not in ["0", "1"]:
                            all_cells.append(ET.tostring(cell, encoding='unicode'))
                            
            except Exception as e:
                self._log(f"解析失败 {xml_path}: {e}")
        
        # 转换为XMLFragment
        fragments = []
        for i, cell_xml in enumerate(all_cells):
            fragments.append(XMLFragment(
                element_id=i,
                xml_content=cell_xml,
                layer_level=LayerLevel.OTHER.value  # 无法判断层级时使用OTHER
            ))
        
        # 使用标准流程合并
        context = ProcessingContext(
            image_path="",
            canvas_width=canvas_width,
            canvas_height=canvas_height,
            output_dir=os.path.dirname(output_path) or "."
        )
        context.xml_fragments = fragments
        
        result = self.process(context)
        
        # 如果输出路径不同，重命名
        if result.metadata.get('output_path') != output_path:
            import shutil
            shutil.move(result.metadata['output_path'], output_path)
        
        return output_path
    
    def merge_with_text_xml(self,
                            shape_xml_path: str,
                            text_xml_path: str,
                            output_path: str,
                            image_path: str = None) -> str:
        """
        合并图形XML和文字XML
        
        这是最常用的合并场景：SAM3提取的图形 + OCR提取的文字
        
        Args:
            shape_xml_path: 图形XML路径
            text_xml_path: 文字XML路径
            output_path: 输出路径
            image_path: 原始图片路径（用于获取尺寸）
            
        Returns:
            输出文件路径
        """
        self._log("合并图形和文字XML")
        
        fragments = []
        
        # 解析图形XML
        shape_tree = ET.parse(shape_xml_path)
        shape_root = shape_tree.getroot()
        
        # 获取画布尺寸
        model = shape_root.find(".//mxGraphModel")
        canvas_width = int(model.get("pageWidth", 800))
        canvas_height = int(model.get("pageHeight", 600))
        
        # 提取图形cells并判断层级
        root_elem = shape_root.find(".//root")
        for cell in list(root_elem):
            cell_id = cell.get("id")
            if cell_id in ["0", "1"]:
                continue
            
            cell_xml = ET.tostring(cell, encoding='unicode')
            style = cell.get("style", "")
            
            # 根据style判断层级
            if "image=data:image" in style:
                layer = LayerLevel.IMAGE.value
            else:
                layer = LayerLevel.BASIC_SHAPE.value
            
            # 提取geometry信息用于排序
            geom = cell.find("mxGeometry")
            bbox = None
            if geom is not None:
                try:
                    bbox = BoundingBox(
                        x1=int(float(geom.get("x", 0))),
                        y1=int(float(geom.get("y", 0))),
                        x2=int(float(geom.get("x", 0))) + int(float(geom.get("width", 0))),
                        y2=int(float(geom.get("y", 0))) + int(float(geom.get("height", 0)))
                    )
                except:
                    pass
            
            fragments.append(XMLFragment(
                element_id=len(fragments),
                xml_content=cell_xml,
                layer_level=layer,
                bbox=bbox
            ))
        
        # 解析文字XML
        if text_xml_path and os.path.exists(text_xml_path):
            try:
                text_tree = ET.parse(text_xml_path)
                text_root = text_tree.getroot()
                text_root_elem = text_root.find(".//root")
                
                if text_root_elem is not None:
                    for cell in text_root_elem:
                        if cell.get("id") not in ["0", "1"]:
                            cell_xml = ET.tostring(cell, encoding='unicode')
                            
                            # 文字层级最高
                            fragments.append(XMLFragment(
                                element_id=len(fragments),
                                xml_content=cell_xml,
                                layer_level=LayerLevel.TEXT.value
                            ))
            except Exception as e:
                self._log(f"解析文字XML失败: {e}")
        
        # 使用标准流程合并
        context = ProcessingContext(
            image_path=image_path or "",
            canvas_width=canvas_width,
            canvas_height=canvas_height,
            output_dir=os.path.dirname(output_path) or "."
        )
        context.xml_fragments = fragments
        
        result = self.process(context)
        
        # 重命名到指定输出路径
        actual_output = result.metadata.get('output_path')
        if actual_output and actual_output != output_path:
            import shutil
            shutil.move(actual_output, output_path)
        
        self._log(f"合并完成: {output_path}")
        return output_path


# ======================== 快捷函数 ========================
def merge_fragments(fragments: List[XMLFragment],
                    canvas_width: int,
                    canvas_height: int,
                    output_path: str) -> str:
    """
    快捷函数 - 合并XML片段
    
    Args:
        fragments: XMLFragment列表
        canvas_width: 画布宽度
        canvas_height: 画布高度
        output_path: 输出路径
        
    Returns:
        输出文件路径
        
    使用示例:
        from modules.data_types import XMLFragment, LayerLevel
        
        fragments = [
            XMLFragment(
                element_id=0,
                xml_content='<mxCell id="2" .../>',
                layer_level=LayerLevel.BASIC_SHAPE.value
            ),
            XMLFragment(
                element_id=1,
                xml_content='<mxCell id="3" .../>',
                layer_level=LayerLevel.IMAGE.value
            ),
        ]
        
        output = merge_fragments(fragments, 800, 600, "output.drawio.xml")
    """
    merger = XMLMerger()
    context = ProcessingContext(
        image_path="",
        canvas_width=canvas_width,
        canvas_height=canvas_height,
        output_dir=os.path.dirname(output_path) or "."
    )
    context.xml_fragments = fragments
    
    result = merger.process(context)
    
    actual_output = result.metadata.get('output_path')
    if actual_output and actual_output != output_path:
        import shutil
        shutil.move(actual_output, output_path)
    
    return output_path


def merge_shape_and_text(shape_xml: str, 
                         text_xml: str, 
                         output_path: str) -> str:
    """
    快捷函数 - 合并图形XML和文字XML
    
    Args:
        shape_xml: 图形XML路径
        text_xml: 文字XML路径
        output_path: 输出路径
        
    Returns:
        输出文件路径
        
    使用示例:
        output = merge_shape_and_text("shapes.xml", "text.xml", "merged.xml")
    """
    merger = XMLMerger()
    return merger.merge_with_text_xml(shape_xml, text_xml, output_path)
