# AI降重工具

一个免费开源的AI文章降重工具，基于大语言模型（支持 DeepSeek / 通义千问等），帮助学术论文、科研文章降低AI生成痕迹。

## 功能特点

- 🔧 **5档降重度控制**：从轻度同义词替换到深度句式重写
- 📚 **示例参考系统**：可上传自己的"原文-降重后"示例，越用越准
- 🔍 **预览对比**：改写后并排对比原文和结果
- 🛡️ **学术严谨底线**：专业术语、数据、引用格式原样保留
- 💰 **成本极低**：DeepSeek API 千字几分钱

## 快速开始

### 1. 安装依赖

```bash
pip install flask flask-cors openai
```

### 2. 启动服务

```bash
python3 backend/server.py
```

### 3. 打开浏览器

访问 `http://localhost:5100`

### 4. 配置 API Key

- 填入你的 API Key（推荐 [DeepSeek](https://platform.deepseek.com/)）
- 或切换到通义千问等其他兼容 OpenAI 格式的 API

## 项目结构

```
ai-downgrade/
├── backend/
│   ├── server.py          # Flask 后端服务
│   ├── build_examples.py  # 示例数据构建脚本
│   └── write_pairs.py     # 示例数据写入工具
├── frontend/
│   └── index.html         # 前端页面
└── data/
    ├── examples.json      # 参考示例库
    ├── sources/           # 原始文本（你的降重前示例）
    └── results/           # 降重后文本（你的降重后示例）
```

## 自定义示例

在页面底部的「示例管理」面板中，可以：
- 查看已有的参考示例
- 添加新的"原文-降重后"对照示例
- 删除不需要的示例

示例越多，改写效果越稳定。

## 技术方案

- **后端**：Flask + OpenAI 兼容 API
- **前端**：纯 HTML/CSS/JS，零框架
- **降重策略**：
  - 第一轮：基于 prompt engineering 的正常改写
  - 第二轮：质检 ── 逐段对比原文，识别"只换词不动骨架"的段落，强制拆骨重写

## License

MIT
