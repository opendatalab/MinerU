# Copyright (c) Opendatalab. All rights reserved.
import re

LEFT_PATTERN = re.compile(r'(\\left)(\S*)')
RIGHT_PATTERN = re.compile(r'(\\right)(\S*)')
LEFT_COUNT_PATTERN = re.compile(r'\\left(?![a-zA-Z])')
RIGHT_COUNT_PATTERN = re.compile(r'\\right(?![a-zA-Z])')
LEFT_RIGHT_REMOVE_PATTERN = re.compile(r'\\left\.?|\\right\.?')

def fix_latex_left_right(s, fix_delimiter=True):
    """
    修复LaTeX中的\\left和\\right命令
    1. 确保它们后面跟有效分隔符
    2. 平衡\\left和\\right的数量
    """
    # 白名单分隔符
    valid_delims_list = [r'(', r')', r'[', r']', r'{', r'}', r'/', r'|',
                         r'\{', r'\}', r'\lceil', r'\rceil', r'\lfloor',
                         r'\rfloor', r'\backslash', r'\uparrow', r'\downarrow',
                         r'\Uparrow', r'\Downarrow', r'\|', r'\.']

    # 为\left后缺失有效分隔符的情况添加点
    def fix_delim(match, is_left=True):
        cmd = match.group(1)  # \left 或 \right
        rest = match.group(2) if len(match.groups()) > 1 else ""
        if not rest or rest not in valid_delims_list:
            return cmd + "."
        return match.group(0)

    # 使用更精确的模式匹配\left和\right命令
    # 确保它们是独立的命令，不是其他命令的一部分
    # 使用预编译正则和统一回调函数
    if fix_delimiter:
        s = LEFT_PATTERN.sub(lambda m: fix_delim(m, True), s)
        s = RIGHT_PATTERN.sub(lambda m: fix_delim(m, False), s)

    # 更精确地计算\left和\right的数量
    left_count = len(LEFT_COUNT_PATTERN.findall(s))  # 不匹配\lefteqn等
    right_count = len(RIGHT_COUNT_PATTERN.findall(s))  # 不匹配\rightarrow等

    if left_count == right_count:
        # 如果数量相等，检查是否在同一组
        return fix_left_right_pairs(s)
        # return s
    else:
        # 如果数量不等，移除所有\left和\right
        # logger.debug(f"latex:{s}")
        # logger.warning(f"left_count: {left_count}, right_count: {right_count}")
        return LEFT_RIGHT_REMOVE_PATTERN.sub('', s)


def fix_left_right_pairs(latex_formula):
    """
    检测并修复LaTeX公式中\\left和\\right不在同一组的情况

    Args:
        latex_formula (str): 输入的LaTeX公式

    Returns:
        str: 修复后的LaTeX公式
    """
    # 用于跟踪花括号嵌套层级
    brace_stack = []
    # 用于存储\left信息: (位置, 深度, 分隔符)
    left_stack = []
    # 存储需要调整的\right信息: (开始位置, 结束位置, 目标位置)
    adjustments = []

    i = 0
    while i < len(latex_formula):
        # 检查是否是转义字符
        if i > 0 and latex_formula[i - 1] == '\\':
            backslash_count = 0
            j = i - 1
            while j >= 0 and latex_formula[j] == '\\':
                backslash_count += 1
                j -= 1

            if backslash_count % 2 == 1:
                i += 1
                continue

        # 检测\left命令
        if i + 5 < len(latex_formula) and latex_formula[i:i + 5] == "\\left" and i + 5 < len(latex_formula):
            delimiter = latex_formula[i + 5]
            left_stack.append((i, len(brace_stack), delimiter))
            i += 6  # 跳过\left和分隔符
            continue

        # 检测\right命令
        elif i + 6 < len(latex_formula) and latex_formula[i:i + 6] == "\\right" and i + 6 < len(latex_formula):
            delimiter = latex_formula[i + 6]

            if left_stack:
                left_pos, left_depth, left_delim = left_stack.pop()

                # 如果\left和\right不在同一花括号深度
                if left_depth != len(brace_stack):
                    # 找到\left所在花括号组的结束位置
                    target_pos = find_group_end(latex_formula, left_pos, left_depth)
                    if target_pos != -1:
                        # 记录需要移动的\right
                        adjustments.append((i, i + 7, target_pos))

            i += 7  # 跳过\right和分隔符
            continue

        # 处理花括号
        if latex_formula[i] == '{':
            brace_stack.append(i)
        elif latex_formula[i] == '}':
            if brace_stack:
                brace_stack.pop()

        i += 1

    # 应用调整，从后向前处理以避免索引变化
    if not adjustments:
        return latex_formula

    result = list(latex_formula)
    adjustments.sort(reverse=True, key=lambda x: x[0])

    for start, end, target in adjustments:
        # 提取\right部分
        right_part = result[start:end]
        # 从原位置删除
        del result[start:end]
        # 在目标位置插入
        result.insert(target, ''.join(right_part))

    return ''.join(result)


def find_group_end(text, pos, depth):
    """查找特定深度的花括号组的结束位置"""
    current_depth = depth
    i = pos

    while i < len(text):
        if text[i] == '{' and (i == 0 or not is_escaped(text, i)):
            current_depth += 1
        elif text[i] == '}' and (i == 0 or not is_escaped(text, i)):
            current_depth -= 1
            if current_depth < depth:
                return i
        i += 1

    return -1  # 未找到对应结束位置


def is_escaped(text, pos):
    """检查字符是否被转义"""
    backslash_count = 0
    j = pos - 1
    while j >= 0 and text[j] == '\\':
        backslash_count += 1
        j -= 1

    return backslash_count % 2 == 1


# 常见的在KaTeX/MathJax中可用的数学环境
ENV_TYPES = ['array', 'matrix', 'pmatrix', 'bmatrix', 'vmatrix',
             'Bmatrix', 'Vmatrix', 'cases', 'aligned', 'gathered', 'align', 'align*']
ENV_BEGIN_PATTERNS = {env: re.compile(r'\\begin\{' + env + r'\}') for env in ENV_TYPES}
ENV_END_PATTERNS = {env: re.compile(r'\\end\{' + env + r'\}') for env in ENV_TYPES}
ENV_FORMAT_PATTERNS = {env: re.compile(r'\\begin\{' + env + r'\}\{([^}]*)\}') for env in ENV_TYPES}
MATHRING_FONT_ARGUMENT_PATTERN = re.compile(
    r"\\mathring(?!\s*\{)\s*(?P<argument>\\mathrm\s*\{[^{}]*\})"
)

def fix_latex_environments(s):
    """
    检测LaTeX中环境（如array）的\\begin和\\end是否匹配
    1. 如果缺少\\begin标签则在开头添加
    2. 如果缺少\\end标签则在末尾添加
    """
    for env in ENV_TYPES:
        begin_count = len(ENV_BEGIN_PATTERNS[env].findall(s))
        end_count = len(ENV_END_PATTERNS[env].findall(s))

        if begin_count != end_count:
            if end_count > begin_count:
                format_match = ENV_FORMAT_PATTERNS[env].search(s)
                default_format = '{c}' if env == 'array' else ''
                format_str = '{' + format_match.group(1) + '}' if format_match else default_format

                missing_count = end_count - begin_count
                begin_command = '\\begin{' + env + '}' + format_str + ' '
                s = begin_command * missing_count + s
            else:
                missing_count = begin_count - end_count
                s = s + (' \\end{' + env + '}') * missing_count

    return s


def fix_mathring_font_arguments(s: str) -> str:
    """为 mathring 后未分组的 mathrm 参数补充显式花括号。"""

    def _replace(match: re.Match[str]) -> str:
        """把单个 mathrm 字体参数包装成 mathring 的完整参数。"""
        return "\\mathring{" + match.group("argument") + "}"

    return MATHRING_FONT_ARGUMENT_PATTERN.sub(_replace, s)


def remove_up_commands(s: str):
    """Remove unnecessary up commands from LaTeX code."""
    UP_PATTERN = re.compile(r'\\up([a-zA-Z]+)')
    s = UP_PATTERN.sub(
        lambda m: m.group(0) if m.group(1) in ["arrow", "downarrow", "lus", "silon"] else f"\\{m.group(1)}", s
    )
    return s


def remove_unsupported_commands(s: str):
    """Remove unsupported LaTeX commands."""
    COMMANDS_TO_REMOVE_PATTERN = re.compile(
        r'\\(?:lefteqn|boldmath|ensuremath|centering|textsubscript|sides|textsl|textcent|emph|protect|null)')
    s = COMMANDS_TO_REMOVE_PATTERN.sub('', s)
    return s


def _apply_shared_latex_repairs(s: str, *, fix_delimiter: bool) -> str:
    """执行公式识别所需的通用 LaTeX 修复步骤。"""
    s = fix_latex_left_right(s, fix_delimiter=fix_delimiter)
    s = fix_latex_environments(s)
    s = fix_mathring_font_arguments(s)
    s = remove_up_commands(s)
    return remove_unsupported_commands(s)


def fix_pp_formulanet_latex(s: str) -> str:
    """执行 PP-FormulaNet 专用 LaTeX 后处理，不补写 left/right 分隔符。"""
    return _apply_shared_latex_repairs(s, fix_delimiter=False)


def largest_power_of_two_leq(value: int) -> int:
    if value < 1:
        return 0
    return 2 ** (value.bit_length() - 1)


def get_mfr_effective_batch_size(num_items: int, requested_batch_size: int) -> int:
    return min(
        requested_batch_size,
        largest_power_of_two_leq(max(1, num_items)),
    )


def get_mfr_min_dynamic_batch_size(requested_batch_size: int) -> int:
    return max(16, requested_batch_size // 4)


def finalize_mfr_batch_groups(
    batch_groups: list[list[int]],
    total_count: int,
    requested_batch_size: int,
) -> list[list[int]]:
    """整理 MFR batch 分组，避免尾部大面积公式被重新合并成超大 batch。"""
    if not batch_groups:
        return []

    if len(batch_groups) == 1:
        if total_count <= 1 or requested_batch_size <= total_count:
            return batch_groups

        first_group_size = largest_power_of_two_leq(total_count - 1)
        if first_group_size < 1:
            return batch_groups

        source_group = batch_groups[0]
        first_group = source_group[:first_group_size]
        second_group = source_group[first_group_size:]
        if not first_group or not second_group:
            return batch_groups
        return [first_group, second_group]

    min_dynamic_batch_size = get_mfr_min_dynamic_batch_size(requested_batch_size)
    while len(batch_groups) >= 3 and len(batch_groups[-1]) < len(batch_groups[-2]):
        tail_merge_limit = min(
            requested_batch_size,
            max(len(batch_groups[-2]), min_dynamic_batch_size),
        )
        # 尾部通常聚集最大面积公式，合并后超过安全上限会造成最后 batch 显存峰值过高。
        if len(batch_groups[-2]) + len(batch_groups[-1]) > tail_merge_limit:
            break
        tail_group = batch_groups.pop()
        batch_groups[-1].extend(tail_group)

    return batch_groups


def build_mfr_batch_groups(sorted_areas: list[int], requested_batch_size: int) -> list[list[int]]:
    if not sorted_areas:
        return []

    total_count = len(sorted_areas)
    effective_batch_size = get_mfr_effective_batch_size(
        total_count,
        requested_batch_size,
    )
    if effective_batch_size < 1:
        return []

    min_dynamic_batch_size = get_mfr_min_dynamic_batch_size(requested_batch_size)
    batch_groups = []
    if total_count < min_dynamic_batch_size:
        batch_groups.append(list(range(total_count)))
        return finalize_mfr_batch_groups(
            batch_groups,
            total_count,
            requested_batch_size,
        )

    base_mean_area = sum(sorted_areas[:effective_batch_size]) / effective_batch_size
    cursor = 0

    while cursor < total_count:
        remaining_count = total_count - cursor
        if remaining_count < min_dynamic_batch_size:
            batch_groups.append(list(range(cursor, total_count)))
            break

        probe_size = min(effective_batch_size, remaining_count)
        current_mean_area = sum(sorted_areas[cursor : cursor + probe_size]) / probe_size
        ratio = 1 if base_mean_area <= 0 else current_mean_area / base_mean_area

        candidate_batch_size = effective_batch_size
        threshold = 4
        while (
            ratio >= threshold
            and candidate_batch_size // 2 >= min_dynamic_batch_size
        ):
            candidate_batch_size //= 2
            threshold *= 2

        candidate_batch_size = min(
            candidate_batch_size,
            largest_power_of_two_leq(remaining_count),
        )
        batch_groups.append(list(range(cursor, cursor + candidate_batch_size)))
        cursor += candidate_batch_size

    return finalize_mfr_batch_groups(
        batch_groups,
        total_count,
        requested_batch_size,
    )
