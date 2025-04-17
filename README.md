# 课程知识图谱 LLM 应用

这是一个基于课程知识图谱的 LLM 应用项目，使用 Flask 框架构建，结合 Neo4j 图数据库存储课程知识图谱数据。

## 项目结构

```
.
├── app.py                 # Flask 应用主文件
├── create_neo4j_database.py  # Neo4j 数据库初始化脚本
├── prerequisite-dependency.csv  # 课程先修依赖关系数据
├── requirements.txt       # 项目依赖
├── static/               # 静态资源文件
└── templates/            # HTML 模板文件
```

## 环境要求

- Python 3.8+
- Neo4j 数据库 == 5.27.0
- 其他依赖见 requirements.txt

## 安装步骤

1. 克隆项目到本地：
```bash
git clone [项目地址]
cd class_knowledge_LLM
```

2. 创建并激活虚拟环境：
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
.\venv\Scripts\activate  # Windows
```

3. 安装依赖：
```bash
pip install -r requirements.txt
```

4. 配置 Neo4j 数据库：
- 确保 Neo4j 数据库已安装并运行
- 修改 `create_neo4j_database.py` 中的数据库连接信息
- 运行初始化脚本：
```bash
python create_neo4j_database.py
```

## 运行应用

```bash
python app.py
```

应用将在 http://localhost:7474 启动。

## 功能特性

- 课程知识图谱可视化
- 课程先修关系查询
- 基于 LLM 的智能问答

## 注意事项

- 确保 Neo4j 数据库已正确配置并运行
- 首次运行前需要执行数据库初始化脚本
- 建议在虚拟环境中运行项目
