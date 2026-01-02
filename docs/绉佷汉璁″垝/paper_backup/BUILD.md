# 论文编译指南

## 环境要求

### Windows
1. **安装 MiKTeX**：https://miktex.org/download
   - 下载并运行安装程序
   - 选择 "Install missing packages on-the-fly: Yes"

2. **安装 Perl**（latexmk 需要）：
   - 推荐 Strawberry Perl：https://strawberryperl.com/

### Linux (Ubuntu/Debian)
```bash
sudo apt-get install texlive-full latexmk
```

### macOS
```bash
brew install --cask mactex
```

## 编译命令

### 使用 latexmk（推荐）
```bash
cd docs/paper
latexmk -pdf main.tex
```

### 手动编译
```bash
cd docs/paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

### 清理临时文件
```bash
latexmk -c
```

## 输出文件

- `main.pdf` - 最终论文 PDF
- `main.aux`, `main.log`, `main.bbl` - 编译临时文件

## 在线编译（无需本地安装）

### Overleaf（推荐）
1. 上传整个 `docs/paper/` 目录到 Overleaf
2. 设置 main.tex 为主文档
3. 点击 "Recompile"

## 模板切换

当前使用：`neurips_2025.sty` (preprint 模式)

- **arXiv 预印本**：`\usepackage[preprint]{neurips_2025}`
- **会议正式版**：`\usepackage[final]{neurips_2025}`

## 文件结构

```
docs/paper/
├── main.tex           # 主 LaTeX 文件
├── references.bib     # 引用数据库
├── neurips_2025.sty   # NeurIPS 样式文件
├── figures/           # 图片目录
│   └── main/          # 正文图片
└── BUILD.md           # 本文件
```

## 双轨存储

- `main.tex` - LaTeX 版本（用于提交）
- `arxiv_draft_v1.md` - Markdown 版本（便于阅读/协作）

两者内容应保持同步。

