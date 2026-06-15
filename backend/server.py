#!/usr/bin/env python3
"""
AI降重工具 - 后端服务
Flask + OpenAI 兼容 API（默认用 DeepSeek，也可切换其他）
支持三路并行改写 + 结构差异择优
"""

import json, os, re, random, time, math
from concurrent.futures import ThreadPoolExecutor, as_completed
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from openai import OpenAI

app = Flask(__name__, static_folder='../frontend', static_url_path='')
CORS(app)

# ── Configuration ──────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
EXAMPLES_FILE = os.path.join(DATA_DIR, 'examples.json')

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
    data = request.get_json(force=True)
    source = (data.get('source') or '').strip()
    result = (data.get('result') or '').strip()
    source_id = (data.get('id') or '').strip()

    if not source:
        return jsonify({'ok': False, 'error': '请填写原文内容'}), 400
    if not result:
        return jsonify({'ok': False, 'error': '请填写降重后内容'}), 400

    if not source_id:
        source_id = str(int(time.time() * 1000))

    for ex in EXAMPLES:
        if ex.get('id') == source_id:
            return jsonify({'ok': False, 'error': f'ID "{source_id}" 已存在'}), 400

    EXAMPLES.append({"id": source_id, "source": source, "result": result})
    save_examples()
    return jsonify({'ok': True, 'id': source_id, 'examples_count': len(EXAMPLES)})

@app.route('/api/examples/delete', methods=['POST'])
def delete_example():
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
    intensity = int(data.get('intensity', 3))

    if not api_key:
        return jsonify({'ok': False, 'error': '请填写 API Key'}), 400

    try:
        client = OpenAI(api_key=api_key, base_url=api_base)

        strategies = build_strategies(text, intensity)
        candidates = []
        total_tokens = 0

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {}
            for idx, (sys_prompt, user_prompt, temp) in enumerate(strategies):
                futures[executor.submit(
                    _call_rewrite, client, model, sys_prompt, user_prompt, temp
                )] = idx

            for future in as_completed(futures):
                result_text, tokens = future.result()
                if result_text:
                    candidates.append(result_text)
                    total_tokens += tokens

        if not candidates:
            return jsonify({'ok': False, 'error': '所有改写策略都失败了，请重试'}), 500

        best = select_best_candidate(text, candidates)

        return jsonify({
            'ok': True,
            'result': best,
            'usage': {
                'model': model,
                'intensity': intensity,
                'tokens': total_tokens,
                'candidates': len(candidates)
            }
        })
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500


def _call_rewrite(client, model, system_prompt, user_prompt, temp):
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temp,
            max_tokens=4096
        )
        result = r.choices[0].message.content.strip()
        result = re.sub(r'^```[\s\S]*?\n', '', result)
        result = re.sub(r'\n```$', '', result)
        tokens = r.usage.total_tokens if r.usage else 0
        return result, tokens
    except Exception as e:
        print(f"Rewrite call failed: {e}")
        return None, 0


def select_best_candidate(original, candidates):
    if len(candidates) == 1:
        return candidates[0]

    scores = []
    for c in candidates:
        score = structural_edit_distance(original, c)
        scores.append((score, c))

    scores.sort(key=lambda x: x[0], reverse=True)
    return scores[0][1]


def structural_edit_distance(orig, cand):
    """度量两段文本的结构差异。综合句长序列、词序列、句子数量三个维度。"""

    def get_sentences(text):
        parts = re.split(r'[。；;？！\n]+', text)
        return [p.strip() for p in parts if p.strip() and len(p.strip()) > 3]

    sents_orig = get_sentences(orig)
    sents_cand = get_sentences(cand)

    if not sents_orig or not sents_cand:
        return 0.0

    # 句长序列余弦距离
    len_orig = [len(s) for s in sents_orig]
    len_cand = [len(s) for s in sents_cand]
    ml = max(len(len_orig), len(len_cand))
    len_orig += [0] * (ml - len(len_orig))
    len_cand += [0] * (ml - len(len_cand))

    dot = sum(a * b for a, b in zip(len_orig, len_cand))
    na = math.sqrt(sum(a * a for a in len_orig))
    nb = math.sqrt(sum(b * b for b in len_cand))
    rhythm_diff = 1.0 - (dot / (na * nb) if na and nb else 0.0)

    # 词序列编辑距离
    def get_words(text):
        return re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z0-9]+', text)[:200]

    word_edit = levenshtein_ratio(get_words(orig), get_words(cand))

    # 句子数量差异
    sent_count_diff = abs(len(sents_orig) - len(sents_cand)) / max(len(sents_orig), 1)

    return 0.4 * rhythm_diff + 0.4 * word_edit + 0.2 * sent_count_diff


def levenshtein_ratio(a, b):
    if not a and not b:
        return 0.0
    if not a or not b:
        return 1.0

    n, m = len(a), len(b)
    if n > m:
        a, b = b, a
        n, m = m, n

    current = list(range(n + 1))
    for i in range(1, m + 1):
        previous, current = current, [i] + [0] * n
        for j in range(1, n + 1):
            add, delete = previous[j] + 1, current[j - 1] + 1
            change = previous[j - 1] + (0 if a[j - 1] == b[i - 1] else 1)
            current[j] = min(add, delete, change)

    return current[n] / max(n, m)


def build_strategies(text, intensity):
    """构建三路不同的改写策略"""
    base_temp = 0.72 + (intensity - 1) * 0.04

    return [
        # 策略A：标准
        (SYSTEM_PROMPT, build_prompt(text, intensity), base_temp + 0.02),
        # 策略B：激进拆骨
        (SYSTEM_PROMPT_B, build_prompt(text, min(intensity + 1, 5)), base_temp + 0.08),
        # 策略C：语序打乱
        (SYSTEM_PROMPT_C, build_prompt(text, intensity), base_temp + 0.05),
    ]


# ── Prompt Definitions ─────────────────────────────────────────

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
原文："通过这种二值化编码，将原本具有语义描述的类别信息转换为机器学习模型易于处理的数值形式。"
改写："这种二值化编码，是把原本用语义描述的类别信息，转成了机器学习模型容易处理的数值。"
（"通过…将…转换为"的AI框架消失了）

示例2——因果倒装：
原文："由于多分类变量包含多个不同类别，若采用简单数值编码可能引入顺序关系，因此本研究采用独热编码。"
改写："多分类变量有好几个类别，直接拿数字编码容易让模型误会类别之间有先后顺序。所以本研究用的是独热编码。"
（"由于…因此"→"容易…所以"，句式骨架完全不同）

示例3——并列重写：
原文："该编码策略不仅保留了类别语义，还使模型能够直接计算，避免了表示方式不当带来的偏差。"
改写："这个编码策略保留了类别原本的语义，模型的数值计算也能直接用。类别的表示方式如果不合理，本来很容易带来偏差，这样一来就避开了。"
（"不仅A还B，避免了C"→"A了，B也能。C本来的问题，解决了"）

## 句式骨架替换
以下AI句式骨架必须彻底替换，不要只在上面换词：
- "通过X来实现Y" → "X是Y的途径" / "用X来做Y" / "Y的达成，离不开X"
- "由于A因此B" → "A的存在，导致B" / "B跟A分不开" / "A带来了B"
- "使得/使…能够" → "让…得以" / "…就可以" / 直接删掉"使得"
- "具有…意义/价值" → "对…来说很重要" / "在…方面很关键"
- "基于…本文…" → "本文从…出发" / "以…为框架"
- "不仅是…也是…" → 拆成两句，或换成"…的同时，还…"

# 具体改写手法

1. **同义替换**：
   "通过"→"依靠/采用/经由"；"避免"→"防止/躲开/杜绝"
   "使用"→"用/运用"；"影响"→"产生…影响/作用"；"存在"→"有着/有"
   "为"（=号含义）→"是"；"从…角度"→"就…来说"；"能够"→"可以"；"需要"→"得"
   同一篇文章里同一个词不要总换同一个。

2. **被动变主动**：
   "数据被转换为…" → "数据转换成了…"
   仅在不歧义时转换。

3. **连接词直接删除**：
   "同时/此外/与此同时/不仅如此/另外" → 能删就删
   "首先/其次/再次/最后；综上所述/总而言之/值得注意的是/需要指出的是" → 必须删除

4. **拆散并列结构**：
   "A、B及C" → "A和B，另外还有C"

5. **标点符号改写**（重要）：AI文章的标点往往过于整齐规范，改写时要打乱标点节奏
   - 分号排比 → 拆成独立的短句："A；B；C" → "A。B。C" 或者 "A，而B，C也是"
   - 冒号总结腔 → 改成逗号或直接叙述："具体包括：A、B及C" → "这里面有A，有B，还有C"
   - 引号强调 → 适当删除不必要的引号："具有'启蒙—基础—进阶—竞技'四类课程" → "四类课程，从启蒙到竞技都有"
   - 破折号解释 → 改成逗号或括号："该模型——即LSTM——表现出" → "LSTM这个模型表现出"
   - 顿号列举 → 尽量拆："A、B、C、D" → "A和B，再加上C和D"
   原则：同一段落内标点类型尽量多样化，不要连续用同一标点结构

6. **拆散排比句**（AI标志性特征，优先处理）：
   AI经常写出结构完全对称的排比，这是检测器最容易抓的特征
   例如：
   "部分俱乐部拥有较完整的标准球场，能够支撑较高强度的会员训练；另一些俱乐部场地老化，限制了服务承载能力；此外不同俱乐部在场馆智能化管理水平上也存在明显分化。"
   → 必须彻底拆散排比结构：
   "有的俱乐部标准球场比较齐全，高强度的会员训练撑得住。也有的场地老化、面积不大，服务上能承接的量有限。管理智能化那块，各俱乐部之间的差距也挺明显。"
   
   改法：把排比里的每一项都换成不同的表达方式，节奏必须错开
   - 同样结构的三个分句 → 每个分句用不同的句式和长度
   - "A的特点是X，B的特点是Y，C的特点是Z" → "A在X方面表现突出，Y是B的强项，至于C，Z上更有优势"
   - 排比中相同的句首词全部替换成不同表达

7. **段落首句去总结腔**：
   不用"总体来看"、"研究发现"、"结果表明"、"通过…可以看出"、"根据…不难发现"开头。

8. **适度人味小词**（学术范围内限制）：
   可以："可以说"、"换句话说"、"大概"、"也许"、"这一来"。每段最多1-2个。
   绝对不用："说白了"、"其实这事"、"你想想"、"不得不说"

# 反面示例
❌ 失败：只换词不动骨
原文："这种编码方式不仅保留了类别的原始语义，还使得模型能够直接对数值进行计算和分析。"
失败改写："这个编码方式不光留住了类别的原来语义，也让模型可以直接对数值做计算和分析。"
→ 句式骨架"不仅A还B，使得C"没变

✅ 正确：拆骨重装
"这类编码方式留住了类别原本的语义。模型的数值计算也可以直接进行，不用担心类别表示方式不对带来的偏差。"

# 输出要求
- 直接输出改写后的文章，不要任何解释、前缀、后缀
- 不要用 markdown 代码块包裹
- 段落对应原文的段落
- 每改完一段，问自己：这段话的句式骨架跟原文还一样吗？如果一样，重新改"""

SYSTEM_PROMPT_B = """你是一个极其激进的文本改写者。你的唯一目标是让改写后的文章跟原文在句式结构上完全不同。

# 排比句和标点优先处理（最重要的第一步）
AI最爱写结构对称的排比句，这是AI检测器的头号目标。你必须：
- 所有排比句全部拆散，每个分句换成不同的句式和长度
- 分号全部删除，换成句号或逗号断句
- 冒号总结腔全部删掉，改成直接叙述
- 同一段落内不能用同一种标点结构连用两次
- 顿号列举全部拆成不同的表达方式

# 铁律
- 意思不变、术语不变、数据不变
- 保持学术严谨，不用口语词
- 每条句子都必须从骨架上改变：主谓宾排列、因果顺序、句子长度、停顿位置——全部要跟原文不一样

# 深度拆骨手法（强制执行）

1. 对每个长句做至少两次拆分再重组：
   原文一句话 → 拆成3-4个短句 → 调换顺序 → 重新组装成一个节奏完全不同的段落

2. 因果逻辑必须翻转表达：
   "因为A所以B" → "B的出现，根源在A" 或 "A带来了B的结果"
   "通过X分析Y" → "把Y放在X框架下看" 或 "从X的角度切入Y"

3. 修饰语移位：
   "清晰地展示了特征之间的相关性" → "特征之间的关系，通过热力图能比较清楚地看到"
   把副词/修饰语从动词前移到句首或句尾

4. 数字和数据的表达方式也换：
   "相关系数为-0.58" → "相关系数是-0.58"
   "MAE降至239.01元" → "MAE降到了239.01元"

5. 完全不用以下AI句型骨架：
   "通过…将…转换为" / "不仅…还…" / "之所以…是因为…"
   "使得…能够" / "具有…意义" / "基于…本文…"

在阅读上依然要通顺自然，但结构和原文必须没有任何相似之处。

输出：直接输出完整文章，不要任何标记或解释。"""

SYSTEM_PROMPT_C = """你的任务是改变段落内句子的表达顺序和逻辑组织方式，让同样的意思用完全不同的表达顺序说出来。

# 核心方法：逻辑重组

1. **结论前置**：把原文放在句末的结论挪到句首
   原文："依靠时序特征体系，结合网格搜索法开展超参数寻优，模型能够有效捕捉销售数据背后的长期依赖关系。"
   改写："模型有效捕捉到了销售数据的长期依赖关系。这靠的是：一套时序特征体系，加上网格搜索法做超参数寻优。"

2. **反转信息流**：把原文"A→B→C"的顺序变成"C→B→A"或"B→A→C"
   实验不同的信息呈现顺序，选最自然的那种。

3. **句间不加连接词**：句子之间完全靠内在逻辑衔接，不用任何"同时"、"此外"、"另外"。

4. **同义词库打乱**：每篇文章随机从多个同义词中选不同的表达，避免重复。
   "存在"→从["有着","有","表现出","呈现"]中随机选
   "通过"→从["靠","用","从","经过"]中随机选

5. **拆散一切排比、列举、递进结构**：三个及以上并列全部打散。

# 约束
- 学术术语、数据、数字不动
- 不用口语词
- 字数保持原文幅度

输出：直接输出完整文章，不要标记或解释。"""


def pick_examples(n=3):
    if len(EXAMPLES) <= n:
        return EXAMPLES
    return random.sample(EXAMPLES, n)


def build_prompt(text, intensity=3):
    intensity_desc = {
        1: "轻度改写：只做最表面的同义词替换（如'通过'→'采用'、'为'→'是'），被动语态少量改为主动，句子结构基本不动。保持最严谨的学术语气，不加入任何人味小词",
        2: "偏轻改写：多做同义替换，被动变主动力度加大，删除明显的AI连接词（首先/其次/值得注意的是），段落首句去总结腔。仍保持学术严谨，不拆散并列结构",
        3: "中等改写：全面同义替换 + 被动变主动 + 句式适度松动 + 连接词删除 + 段落首句去总结腔。可适度拆散并列结构，每段最多加1个人味小词。学术严谨与自然表达兼顾",
        4: "偏重改写：在中等基础上，句式和表达顺序更自由，并列结构积极打散，连接词尽量删光。每段最多1-2个人味小词。学术底线依然守住，但读起来更接近真人手笔",
        5: "深度改写：彻底重写表达方式，但意思、结构、专业术语、数据全部不变。连接词全删，并列结构全部打散，被动几乎全改主动。读起来完全不像AI，但仍是一篇严谨的学术文章"
    }

    examples_text = ""
    selected = pick_examples(min(intensity, 4))
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


def save_examples():
    with open(EXAMPLES_FILE, 'w', encoding='utf-8') as f:
        json.dump(EXAMPLES, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    print(f"Loaded {len(EXAMPLES)} examples")
    app.run(host='0.0.0.0', port=5100, debug=True)
