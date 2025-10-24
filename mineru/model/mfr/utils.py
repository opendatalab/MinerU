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


def fix_unbalanced_braces(latex_formula):
    """
    检测LaTeX公式中的花括号是否闭合，并删除无法配对的花括号

    Args:
        latex_formula (str): 输入的LaTeX公式

    Returns:
        str: 删除无法配对的花括号后的LaTeX公式
    """
    stack = []  # 存储左括号的索引
    unmatched = set()  # 存储不匹配括号的索引
    i = 0

    while i < len(latex_formula):
        # 检查是否是转义的花括号
        if latex_formula[i] in ['{', '}']:
            # 计算前面连续的反斜杠数量
            backslash_count = 0
            j = i - 1
            while j >= 0 and latex_formula[j] == '\\':
                backslash_count += 1
                j -= 1

            # 如果前面有奇数个反斜杠，则该花括号是转义的，不参与匹配
            if backslash_count % 2 == 1:
                i += 1
                continue

            # 否则，该花括号参与匹配
            if latex_formula[i] == '{':
                stack.append(i)
            else:  # latex_formula[i] == '}'
                if stack:  # 有对应的左括号
                    stack.pop()
                else:  # 没有对应的左括号
                    unmatched.add(i)

        i += 1

    # 所有未匹配的左括号
    unmatched.update(stack)

    # 构建新字符串，删除不匹配的括号
    return ''.join(char for i, char in enumerate(latex_formula) if i not in unmatched)


def process_latex(input_string):
    """
        处理LaTeX公式中的反斜杠：
        1. 如果\后跟特殊字符(#$%&~_^\\{})或空格，保持不变
        2. 如果\后跟两个小写字母，保持不变
        3. 其他情况，在\后添加空格

        Args:
            input_string (str): 输入的LaTeX公式

        Returns:
            str: 处理后的LaTeX公式
        """

    def replace_func(match):
        # 获取\后面的字符
        next_char = match.group(1)

        # 如果是特殊字符或空格，保持不变
        if next_char in "#$%&~_^|\\{} \t\n\r\v\f":
            return match.group(0)

        # 如果是字母，检查下一个字符
        if 'a' <= next_char <= 'z' or 'A' <= next_char <= 'Z':
            pos = match.start() + 2  # \x后的位置
            if pos < len(input_string) and ('a' <= input_string[pos] <= 'z' or 'A' <= input_string[pos] <= 'Z'):
                # 下一个字符也是字母，保持不变
                return match.group(0)

        # 其他情况，在\后添加空格
        return '\\' + ' ' + next_char

    # 匹配\后面跟一个字符的情况
    pattern = r'\\(.)'

    return re.sub(pattern, replace_func, input_string)

# 常见的在KaTeX/MathJax中可用的数学环境
ENV_TYPES = ['array', 'matrix', 'pmatrix', 'bmatrix', 'vmatrix',
             'Bmatrix', 'Vmatrix', 'cases', 'aligned', 'gathered', 'align', 'align*']
ENV_BEGIN_PATTERNS = {env: re.compile(r'\\begin\{' + env + r'\}') for env in ENV_TYPES}
ENV_END_PATTERNS = {env: re.compile(r'\\end\{' + env + r'\}') for env in ENV_TYPES}
ENV_FORMAT_PATTERNS = {env: re.compile(r'\\begin\{' + env + r'\}\{([^}]*)\}') for env in ENV_TYPES}

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


REPLACEMENTS_PATTERNS = {
    re.compile(r'\\underbar'): r'\\underline',
    re.compile(r'\\Bar'): r'\\hat',
    re.compile(r'\\Hat'): r'\\hat',
    re.compile(r'\\Tilde'): r'\\tilde',
    re.compile(r'\\slash'): r'/',
    re.compile(r'\\textperthousand'): r'‰',
    re.compile(r'\\sun'): r'☉',
    re.compile(r'\\textunderscore'): r'\\_',
    re.compile(r'\\fint'): r'⨏',
    re.compile(r'\\up '): r'\\ ',
    re.compile(r'\\vline = '): r'\\models ',
    re.compile(r'\\vDash '): r'\\models ',
    re.compile(r'\\sq \\sqcup '): r'\\square ',
    re.compile(r'\\copyright'): r'©',
}
QQUAD_PATTERN = re.compile(r'\\qquad(?!\s)')


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


def latex_rm_whitespace(s: str):
    """Remove unnecessary whitespace from LaTeX code."""
    s = fix_unbalanced_braces(s)
    s = fix_latex_left_right(s)
    s = fix_latex_environments(s)

    s = remove_up_commands(s)
    s = remove_unsupported_commands(s)

    # 应用所有替换
    for pattern, replacement in REPLACEMENTS_PATTERNS.items():
        s = pattern.sub(replacement, s)

    # 处理LaTeX中的反斜杠和空格
    s = process_latex(s)

    # \qquad后补空格
    s = QQUAD_PATTERN.sub(r'\\qquad ', s)

    # 如果字符串以反斜杠结尾，去掉最后的反斜杠
    while s.endswith('\\'):
        s = s[:-1]

    return s