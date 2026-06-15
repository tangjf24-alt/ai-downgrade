#!/usr/bin/env python3
"""
AI降重工具 - 后端服务
Flask + OpenAI 兼容 API（默认用 DeepSeek，也可切换其他）
"""

import json, os, re, random, time
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from openai import OpenAI

app = Flask(__name__, static_folder='../frontend', static_url_path='')
CORS(app)

# ── Configuration ──────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
EXAMPLES_FILE = os.path.join(DATA_DIR, 'examples.json')

# Load few-shot examples
with open(EXAMPLES_FILE, 'r', encoding='utf-8') as f:
    EXAMPLES = json.load(f)

# ── Routes ─────────────────────────────────────────────────────

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/api/status', methods=['GET'])
def status():
    return jsonify({
        'ok': True,
        'examples_count': len(EXAMPLES),
        'example_ids': [ex['id'] for ex in EXAMPLES]
    })

@app.route('/api/examples/add', methods=['POST'])
def add_example():
    """添加新的原文-降后文示例到示例库"""
    data = request.get_json(force=True)
    source = (data.get('source') or '').strip()
    result = (data.get('result') or '').strip()
    source_id = (data.get('id') or '').strip()

    if not source:
        return jsonify({'ok': False, 'error': '请填写原文内容'}), 400
    if not result:
        return jsonify({'ok': False, 'error': '请填写降重后内容'}), 400

    # Generate ID if not provided
    if not source_id:
        source_id = str(int(time.time() * 1000))

    # Check duplicate
    for ex in EXAMPLES:
        if ex.get('id') == source_id:
            return jsonify({'ok': False, 'error': f'ID "{source_id}" 已存在，请使用不同的标识'}), 400

    example = {"id": source_id, "source": source, "result": result}
    EXAMPLES.append(example)

    # Persist to file
    save_examples()

    return jsonify({
        'ok': True,
        'id': source_id,
        'examples_count': len(EXAMPLES)
    })


@app.route('/api/examples/delete', methods=['POST'])
def delete_example():
    """删除指定示例"""
    data = request.get_json(force=True)
    target_id = (data.get('id') or '').strip()
    if not target_id:
        return jsonify({'ok': False, 'error': '请提供要删除的示例 ID'}), 400

    global EXAMPLES
    before = len(EXAMPLES)
    EXAMPLES = [ex for ex in EXAMPLES if ex.get('id') != target_id]
    if len(EXAMPLES) == before:
        return jsonify({'ok': False, 'error': f'未找到 ID 为 "{target_id}" 的示例'}), 404

    save_examples()
    return jsonify({'ok': True, 'deleted_id': target_id, 'examples_count': len(EXAMPLES)})


@app.route('/api/examples/list', methods=['GET'])
def list_examples():
    """列出所有示例（含原文和降后文预览）"""
    previews = []
    for ex in EXAMPLES:
        previews.append({
            'id': ex['id'],
            'source_preview': ex['source'][:120] + ('...' if len(ex['source']) > 120 else ''),
            'result_preview': ex['result'][:120] + ('...' if len(ex['result']) > 120 else ''),
            'source_length': len(ex['source']),
            'result_length': len(ex['result'])
        })
    return jsonify({'ok': True, 'examples': previews, 'count': len(previews)})


@app.route('/api/rewrite', methods=['POST'])
def rewrite():
    data = request.get_json(force=True)
    text = (data.get('text') or '').strip()
    if not text:
        return jsonify({'ok': False, 'error': '请输入要降重的文章内容'}), 400

    api_key = data.get('api_key', '')
    api_base = data.get('api_base', 'https://api.deepseek.com')
    model = data.get('model', 'deepseek-chat')
    intensity = int(data.get('intensity', 3))  # 1-5

    if not api_key:
        return jsonify({'ok': False, 'error': '请填写 API Key'}), 400

    try:
        client = OpenAI(api_key=api_key, base_url=api_base)

        # ── Round 1: 正常改写 ──
        prompt1 = build_prompt(text, intensity)
        r1 = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt1}
            ],
            temperature=0.75 + (intensity - 1) * 0.04,
            max_tokens=4096
        )
        pass1 = r1.choices[0].message.content.strip()
        pass1 = re.sub(r'^```[\s\S]*?\n', '', pass1)
        pass1 = re.sub(r'\n```$', '', pass1)
        total_tokens = r1.usage.total_tokens if r1.usage else 0

        # ── Round 2: 质检 + 二次改写 ──
        pass2 = second_pass(client, model, text, pass1, intensity)
        if pass2:
            # Count second pass tokens too
            # (We don't have token count from second pass since it might not be available)
            result_text = pass2
        else:
            result_text = pass1

        return jsonify({
            'ok': True,
            'result': result_text,
            'usage': {
                'model': model,
                'intensity': intensity,
                'tokens': total_tokens
            }
        })
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500


def second_pass(client, model, original, first_pass, intensity):
    """第二轮：挑出骨架没动的句子，强制拆骨重写"""
    # Split into paragraphs
    orig_paras = [p.strip() for p in original.split('\n') if p.strip()]
    pass_paras = [p.strip() for p in first_pass.split('\n') if p.strip()]

    if len(orig_paras) != len(pass_paras) or len(orig_paras) == 0:
        return None  # Paragraph mismatch, skip second pass

    check_prompt = SECOND_PASS_PROMPT.format(
        intensity=intensity,
        pairs="\n\n".join(
            f"## 段落{i}\n原文：{o}\n第一轮改写：{p}"
            for i, (o, p) in enumerate(zip(orig_paras, pass_paras), 1)
        )
    )

    try:
        r = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SECOND_PASS_SYSTEM},
                {"role": "user", "content": check_prompt}
            ],
            temperature=0.55,
            max_tokens=4096
        )
        result = r.choices[0].message.content.strip()
        result = re.sub(r'^```[\s\S]*?\n', '', result)
        result = re.sub(r'\n```$', '', result)
        return result
    except Exception:
        return None  # Second pass failed, keep first pass


# ── Prompt Building ────────────────────────────────────────────

SYSTEM_PROMPT = """你的首要身份不是"改写器"，而是"换个人写这篇文章"。你的目标不是执行一系列替换规则，而是让这篇文章读起来完全像是另一个人独立写出来的——用词习惯不同、组织句子的方式不同、说话的节奏不同。以下规则只是帮助你实现这个目标的工具。

# 核心原则
- 保持原文的意思不变，不增删任何事实信息
- 尽量保持字数与原文一致，不要大幅缩短或拉长
- 保留原文的段落结构和论述逻辑
- 学术/科研文章的严谨性必须守住，不能变成口语聊天

# 学术严谨底线（不可逾越）
- 专业术语不动：p值、相关系数、编码方式、独热编码、回归系数、置信区间等
- 数字、数据、统计量原样保留
- 引用格式、图表编号原样保留
- 绝对不能出现"说白了"、"你懂的"、"差不多就是"、"大概齐"等口语词
- 不能加"其实"、"说真的"等主观感叹词
- 不能用"挺"、"蛮"、"超"等程度副词替代学术表述

# 最高优先级：语感重塑
在执行任何具体操作之前，先在心里问自己："这句话换个写作习惯的人会怎么表达？"

## 拆句重组（最重要的手法，优先执行）
不要只在原句上换词。把每个长句拆成2-3个短句，然后换一种组织方式重新组装。

示例1——拆句后换逻辑顺序：
❌ 原句："通过这种二值化编码，将原本具有语义描述的类别信息转换为机器学习模型易于处理的数值形式。"
🔄 拆成：这种二值化编码，是把原本用语义描述的类别信息，转成了机器学习模型容易处理的数值。
（拆了1个长句→1个短句，"通过…将…转换为"这种典型AI框架消失了）

示例2——因果倒装：
❌ 原句："由于多分类变量包含多个不同类别，若采用简单数值编码可能引入顺序关系，因此本研究采用独热编码。"
🔄 拆成：多分类变量有好几个类别，直接拿数字编码容易让模型误会类别之间有先后顺序。所以本研究用的是独热编码。
（"由于…因此"→"容易…所以"，句式骨架完全不同）

示例3——并列重写：
❌ 原句："该编码策略不仅保留了类别语义，还使模型能够直接计算，避免了表示方式不当带来的偏差。"
🔄 拆成：这个编码策略保留了类别原本的语义，模型的数值计算也能直接用。类别的表示方式如果不合理，本来很容易带来偏差，这样一来就避开了。
（"不仅A还B，避免了C"→"A了，B也能。C本来的问题，解决了"，节奏完全变了）

## 句式骨架替换
以下AI句式骨架必须彻底替换，不要只在上面换词：
- "通过X来实现Y" → 换骨架："X是Y的途径" / "用X来做Y" / "Y的达成，离不开X"
- "由于A因此B" → 换骨架："A的存在，导致B" / "B跟A分不开" / "A带来了B"
- "使得/使…能够" → 换骨架："让…得以" / "…就可以" / 直接删掉"使得"，用逗号断句
- "具有…意义/价值" → 换骨架："对…来说很重要" / "在…方面很关键" / 直接具体化
- "基于…本文…" → 换骨架："本文从…出发" / "以…为框架"
- "不仅是…也是…" → 拆成两句，或者换成"…的同时，还…"

# 具体改写手法（辅助工具，不是流程清单）

1. **同义替换**：
   高频替换："通过"→"依靠/采用/经由"；"避免"→"防止/躲开/杜绝"
   "使用"→"用/运用"；"影响"→"产生…影响/作用"；"存在"→"有着/有"
   "为"（=号含义）→"是"（"相关系数为-0.58"→"相关系数是-0.58"）
   "从…角度"→"就…来说"；"能够"→"可以"；"需要"→"得"
   注意：不要只用固定的替换表，要有变化，同一篇文章里同一个词不要总换同一个。

2. **被动变主动**：
   "数据被转换为二进制特征" → "数据转换成了二进制特征"
   仅在不歧义时转换，被动是唯一准确表达时保留。

3. **连接词直接删除**：
   "同时/此外/与此同时/不仅如此/另外" → 能删就删，不加替代
   "首先/其次/再次/最后；综上所述/总而言之/值得注意的是/需要指出的是" → 必须删除

4. **拆散并列结构**：
   "A、B及C" → "A和B，另外还有C"；"具备X、Y和Z的能力" → "具备X和Y的能力，Z方面也…"

5. **段落首句去总结腔**：
   不要用"总体来看"、"综上所述"、"研究发现"、"结果表明"、"通过…可以看出"、"根据…不难发现"开头
   改成直接叙述。

6. **适度人味小词**（学术范围内严格限制）：
   可以用的："可以说"、"换句话说"、"大概"、"也许"、"这一来"
   每段最多1-2个。绝对不用："说白了"、"其实这事"、"你想想"、"不得不说"

# 什么是失败的改写（反面示例）
以下改写方式算失败，不要这样改：

❌ 只换了几个词，句式骨架完全没动：
原文："这种编码方式不仅保留了类别的原始语义，还使得模型能够直接对数值进行计算和分析。"
失败："这个编码方式不光留住了类别的原来语义，也让模型可以直接对数值做计算和分析。"
→ 问题：虽然换了"不仅→不光"、"还→也"、"为→做"，但"不仅A还B，使得C"的骨架纹丝未动，AI检测器一眼能认出来。

正确答案应该拆骨重装：
✅ "这类编码方式留住了类别原本的语义。模型的数值计算也可以直接进行，不用担心类别表示方式不对带来的偏差。"

❌ 每个连接词都找了个近义词替代：
原文："此外，不同俱乐部在智能化管理水平上存在明显分化，整体资源配置的均衡性仍有提升空间。"
失败："另外，不同俱乐部在智能化管理水平上有着明显分化，整体资源配置的均衡性还有提升空间。"
→ 问题："此外→另外"、"存在→有着"、"仍有→还有"，但结构一字未动。

正确答案应该删掉"此外"，后面换成不同节奏：
✅ "不同俱乐部在智能化管理水平上的差距也挺明显。整体来看，资源配置的均衡性往上提一提的空间还是在的。"

# 输出要求
- 直接输出改写后的文章，不要任何解释、前缀、后缀
- 不要用 markdown 代码块包裹
- 段落对应原文的段落
- 每改完一段，问自己：这段话的句式骨架跟原文还一样吗？如果一样，重新改"""


SECOND_PASS_SYSTEM = """你是AI降重质检员。你的任务是逐段对比原文和第一轮改写，判断哪些段落的句式骨架还跟原文一样，然后只对骨架没动的段落进行深度拆骨重写。

# 判断标准：什么算"骨架没动"
- 主语、谓语、宾语的排列顺序和原文完全一样
- 仅仅换了几个词（如"通过→依靠"、"能够→可以"），但句子结构纹丝未动
- 连接词还在原位，只是换了个近义词
- 因果/转折/并列的逻辑链条长度和顺序没变

# 什么算"骨架已动"（合格，不需要再改）
- 句子的主谓宾排列方式变了
- 长句被拆成了短句再重组
- 因果、转折、并列的逻辑表达方式换了（不只是换连接词）
- 整段的节奏和停顿位置变了

# 二次改写要求
对于骨架没动的段落，抛开第一轮改写直接基于原文重新改。这次唯一的目标就是彻底打碎原来的句式骨架：
- 每个长句至少拆成2个短句再组装
- 主谓宾的排列顺序必须跟原文不同
- 因果/转折/并列的逻辑链条必须换一种组织方式
- 保证学术严谨（术语、数据不动），但说话的方式必须完全不同

# 输出格式
对每个段落：
- 如果骨架已动（合格），原样输出第一轮改写
- 如果骨架没动（不合格），输出彻底拆骨重写后的版本
把所有段落按顺序拼成完整文章输出，不要任何标记、编号、解释。"""

SECOND_PASS_PROMPT = """以下是一篇文章的原文和第一轮改写结果，请逐段质检并修正：

# 改写强度要求
当前降重度为 {intensity}/5，请以此力度进行质检和二次改写。

{pairs}

请输出二次修正后的完整文章（格式同第一轮，段落对应原文）："""


def pick_examples(n=3):
    """随机选择 n 个示例（如果总数不够就全选）"""
    if len(EXAMPLES) <= n:
        return EXAMPLES
    return random.sample(EXAMPLES, n)


def build_prompt(text, intensity=3):
    """根据强度等级构建 prompt"""
    intensity_desc = {
        1: "轻度改写：只做最表面的同义词替换（如'通过'→'采用'、'为'→'是'），被动语态少量改为主动，句子结构基本不动。保持最严谨的学术语气，不加入任何人味小词",
        2: "偏轻改写：多做同义替换，被动变主动力度加大，删除明显的AI连接词（首先/其次/值得注意的是），段落首句去总结腔。仍保持学术严谨，不拆散并列结构",
        3: "中等改写：全面同义替换 + 被动变主动 + 句式适度松动 + 连接词删除 + 段落首句去总结腔。可适度拆散并列结构，每段最多加1个人味小词。学术严谨与自然表达兼顾",
        4: "偏重改写：在中等基础上，句式和表达顺序更自由，并列结构积极打散，连接词尽量删光。每段最多1-2个人味小词。学术底线依然守住，但读起来更接近真人手笔",
        5: "深度改写：彻底重写表达方式，但意思、结构、专业术语、数据全部不变。连接词全删，并列结构全部打散，被动几乎全改主动。读起来完全不像AI，但仍是一篇严谨的学术文章"
    }

    examples_text = ""
    selected = pick_examples(min(intensity, 4))  # 高强度多给示例
    for i, ex in enumerate(selected, 1):
        src_preview = ex['source'][:400] + ('...' if len(ex['source']) > 400 else '')
        res_preview = ex['result'][:400] + ('...' if len(ex['result']) > 400 else '')
        examples_text += f"\n### 示例{i}\n原文：{src_preview}\n改写：{res_preview}\n"

    return f"""请按照以下要求改写文章：

# 改写强度
{intensity_desc.get(intensity, intensity_desc[3])}

# 参考示例
以下是我之前改写过的一些例子，请参考这个风格：
{examples_text}

# 待改写的文章
{text}

请输出改写后的文章："""


# ── Persistence ────────────────────────────────────────────────

def save_examples():
    """Save current examples to file."""
    with open(EXAMPLES_FILE, 'w', encoding='utf-8') as f:
        json.dump(EXAMPLES, f, ensure_ascii=False, indent=2)


# ── Run ────────────────────────────────────────────────────────

if __name__ == '__main__':
    print(f"Loaded {len(EXAMPLES)} examples")
    app.run(host='0.0.0.0', port=5100, debug=True)
