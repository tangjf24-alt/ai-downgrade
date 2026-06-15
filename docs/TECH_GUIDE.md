# AI降重工具 - 技术详解

> 一份写给项目作者的完整教程，零基础友好。

---

## 一、项目总览

这是一个**网页版 AI 文章降重工具**，架构非常简单：

```
浏览器（前端页面） ←→ Flask 后端 ←→ DeepSeek / 通义千问 API
```

- **前端**：你在浏览器里看到的页面，负责输入文章、选参数、展示结果
- **后端**：Flask 服务，负责组装 prompt、调用大模型 API、择优输出
- **大模型 API**：DeepSeek、通义千问等，真正做文章改写

---

## 二、文件结构

```
ai-downgrade/
├── backend/
│   └── server.py              # 整个后端（Flask + prompt + 择优算法）
├── frontend/
│   └── index.html             # 整个前端（纯 HTML/CSS/JS，零框架）
├── data/
│   └── examples.json          # 你的8对参考示例（不传 GitHub，保护隐私）
├── .gitignore                 # 告诉 Git 哪些文件不上传
├── LICENSE                    # MIT 开源许可证
└── README.md                  # 项目说明
```

---

## 三、前端（index.html）

### 3.1 前端做了什么

前端只有一个 HTML 文件，包含三样东西：**HTML（页面结构）+ CSS（样式）+ JavaScript（交互逻辑）**。

核心功能：

1. **布局**：用 CSS Grid 分左中右三栏——原文输入框 | 按钮 | 降重结果框
2. **参数收集**：用户在页面上填的 API Key、选模型、拉滑块的降重度，全部通过 JS 收集
3. **调后端**：点击"开始降重"→ JS 调 `fetch('/api/rewrite', ...)` 发 POST 请求 → 拿结果显示在右边
4. **示例管理**：页面底部的面板可以添加/删除参考示例，调 `/api/examples/add` 和 `/api/examples/delete`

### 3.2 关键 JS 函数

```javascript
// 核心：调后端降重接口
async function doRewrite() {
    const text = sourceEl.value;        // 用户贴的原文
    const apiKey = apiKeyEl.value;      // API Key
    const intensity = ...;              // 降重度 1-5

    // POST JSON 到后端
    const res = await fetch('/api/rewrite', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, api_key: apiKey, intensity, model, api_base })
    });

    const data = await res.json();
    resultEl.value = data.result;  // 显示结果
}
```

前端本身没啥复杂逻辑，就是个传话筒——收参数 → 调接口 → 展示结果。

---

## 四、后端（server.py）——核心

### 4.1 Flask 框架（最外层）

```python
from flask import Flask
app = Flask(__name__)

@app.route('/api/rewrite', methods=['POST'])
def rewrite():
    # 当有人访问 http://xxx:5100/api/rewrite 时执行这个函数
    ...
```

Flask 是 Python 最轻量的 Web 框架。`@app.route()` 就是"把这个 URL 绑定到这个函数"。

### 4.2 rewrite 函数——完整流程

```
接收用户请求
    ↓
构建三路不同的改写策略（不同 prompt + 不同 temperature）
    ↓
并行调用 3 次 DeepSeek API（ThreadPoolExecutor）
    ↓
收集 3 道改写结果
    ↓
算法自动选跟原文差异最大的那道
    ↓
返回给前端
```

### 4.3 为什么要三路并行？

单次调用的结果不稳定——有时候 prompt 再好，模型也会偷懒只换几个词。

三条路用不同的"人格"同时改同一篇文章，相当于三个人用不同风格各改一遍，然后机器自动挑改得最狠（跟原文结构差异最大）的版本。

### 4.4 三路策略的区别

**策略A（标准）**：
- 系统提示词：`SYSTEM_PROMPT`
- 温度（temperature）：`base + 0.02`
- 特点：你最早的 prompt，强调"换个人写"的心流，列出各种具体技法

**策略B（激进拆骨）**：
- 系统提示词：`SYSTEM_PROMPT_B`
- 温度：最高（`base + 0.08`，约 0.80）
- 特点：第一句就是"你是一个极其激进的文本改写者"，把排比和标点放最优先
- 降重度自动 +1（因为要更激进）

**策略C（语序打乱）**：
- 系统提示词：`SYSTEM_PROMPT_C`
- 温度：`base + 0.05`
- 特点：强调结论前置、反转信息流、句间不加连接词

**温度（temperature）是什么？**
> 控制模型回答的"随机性"。0=完全死板固定，1=非常放飞。我们的范围在 0.72~0.80。
> B 策略温度最高，因为它需要更大胆地推翻原结构。

### 4.5 择优算法——select_best_candidate

```python
def select_best_candidate(original, candidates):
    # 对3道结果，每题算一个"与原文差异分"
    for c in candidates:
        score = structural_edit_distance(original, c)
    # 选分最高的
    return scores[0][1]
```

**差异分怎么算？** `structural_edit_distance` 综合三个维度：

| 维度 | 权重 | 怎么算 | 白话解释 |
|------|------|--------|----------|
| 句长节奏 | 40% | 句长序列的余弦距离 | 原文第一句50字第二句30字，改写后第一句20字第二句40字——节奏变了，分就高 |
| 词序列差 | 40% | 莱文斯坦编辑距离 | 换了多少词、增删了多少词 |
| 句子数量 | 20% | 句子总数差/原句数 | 原文3句改成5句——分就高 |

**莱文斯坦距离（Levenshtein）科普：**
> 是"把A变成B最少需要多少步编辑"（增/删/改一个字算一步）。
> 比如 "你好" → "您好" = 1步（改一个字），距离很小。
> "通过分析数据" → "把数据拿来看了看" = 很多步，距离很大。

---

## 五、Prompt 设计详解

### 5.1 两层 Prompt 结构

每次调用 DeepSeek 时发送两个东西：

**系统提示词（System Prompt）**：
- 定义"你是谁、你要做什么、规则是什么"
- 这条在整个对话中起统率作用

**用户提示词（User Prompt）**：
- 每次改写现拼的，包含：改写强度要求 + 随机选3-4个你的示例 + 待改文章原文

### 5.2 你的示例怎么用？

```python
def build_prompt(text, intensity=3):
    # 从8对示例中随机抽3-4对
    selected = pick_examples(min(intensity, 4))
    # 拼进 prompt 里，作为"参考风格"
    examples_text = "### 示例1\n原文：...\n改写：...\n..."
    return f"请按照以下要求改写文章...参考示例...待改写的文章...{text}"
```

这就是"few-shot 学习"——你不训练模型，而是每次在 prompt 里塞几个"示例对"，让模型照着你的风格改。这是 prompt engineering 里最有效的技巧之一。

### 5.3 SYSTEM_PROMPT 的设计思路

系统提示词的结构从上到下：

```
1. 身份定义（"你不是改写器，你是换个人写"）
2. 核心原则（不增删信息、保持字数、学术严谨）
3. 学术底线（术语/数据/引用不动、禁止口语词）
4. 最高优先级：拆句重组（给3个正反示例）
5. 句式骨架替换表（6种常见AI句型→替换方案）
6. 具体技法（同义替换、被动变主动、连接词删除、排比拆散、标点改写）
7. 反面示例（给"失败改写"让模型避坑）
8. 自检要求（改完问自己"骨架变了吗"）
```

### 5.4 你的8对示例的价值

那8对原文/降后文示例的作用：
- 让模型知道"你要的风格"长什么样
- 覆盖科研论文的典型句式（编码方式、统计分析、俱乐部运营、理论框架）
- 示例越多元，模型越不会只会一种改法

---

## 六、示例管理系统

三个接口：

| 接口 | 方法 | 功能 |
|------|------|------|
| `/api/examples/list` | GET | 列出所有示例 |
| `/api/examples/add` | POST | 添加一对"原文+降后文" |
| `/api/examples/delete` | POST | 删除指定示例 |

数据存在 `data/examples.json`，一个 JSON 数组。删改时同时更新内存和文件。

---

## 七、关键概念速查

| 概念 | 一句话解释 |
|------|-----------|
| Flask | Python 的轻量 Web 框架，把 URL 绑定到函数 |
| API | 程序之间通信的接口，本工具调的是 OpenAI 兼容格式 |
| Prompt | 你发给大模型的指令文本 |
| System Prompt | 在对话开头定义角色的指令，权重最高 |
| Temperature | 0~1，控制生成文本的随机性，越高越大胆 |
| Few-shot | 在 prompt 里塞示例让模型模仿风格 |
| ThreadPoolExecutor | Python 的多线程工具，可以同时跑多个任务 |
| 莱文斯坦距离 | 度量两个文本需要改多少步骤才能变成一样 |
| 余弦距离 | 度量两个向量的方向差异，这里用来比句长节奏 |
| Git | 版本控制工具，记录代码每次改动 |

---

## 八、常见问题

**Q：降重度滑块 1-5 具体影响什么？**
A：影响三样——Temperature（1档最低5档最高）、改写的描述要求（1档"只换词"→5档"彻底重写"）、B策略的激进程度（5档时 B 最高温）。

**Q：为什么不用一行代码在 GitHub 上但推不上去？**
A：GitHub 2021年后不再支持密码推送。需要用 Personal Access Token（Settings → Developer settings → Tokens）。

**Q：怎么部署到公网让任何人用？**
A：租个服务器 → 装 Python → 拉代码 → `pip install flask flask-cors openai` → `python3 backend/server.py` → 开放防火墙端口 5100。

**Q：想把前端做得更专业怎么办？**
A：目前是纯 HTML，如果想用框架，可以换成 React 或 Vue。但后端不用动——接口（/api/rewrite 等）是前后端分离的，换任何前端都能用。

---

## 九、贡献指南

想扩展这个项目的话，常见方向：

1. **加更多模型支持**：在 `build_strategies` 里加策略D/E，或者支持 Claude API
2. **加流畅度检查**：降完后自动调模型自己评分，不合格的返工
3. **支持文件上传**：上传 Word/PDF，自动提取文字后降重
4. **批量处理**：一次上传多篇文章
5. **AI 率检测**：集成第三方检测 API，改前改后各测一次看降了多少

---

*文档写于 2026-06-15，随项目代码同步更新*
