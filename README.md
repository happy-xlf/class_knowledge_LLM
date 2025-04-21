# 智能教学辅助系统

这是一个基于人工智能的教学辅助系统，旨在帮助教师更好地了解学生，制定个性化教学计划，并提供学习路径推荐。

## 功能特性

### 1. 用户管理
- 用户注册和登录系统
- 安全的密码加密存储
- 用户会话管理

### 2. 学生信息管理
- 学生信息的增删改查
- 学生详细信息展示
- 学生数据统计分析

### 3. 学习风格分析
- 学习风格评估
- 学习风格指标可视化
- 个性化学习建议

### 4. 知识图谱
- 课程知识点的可视化展示
- 知识点之间的关联关系
- 基于Neo4j的知识图谱存储

### 5. 学习路径规划
- 个性化学习路径推荐
- 最优学习路径计算
- 学习进度追踪

### 6. 教学计划生成
- 智能教学计划生成
- 教学计划导出（Markdown格式）
- 课程统计分析

### 7. 学情分析评估
- 学生成绩趋势分析
- 知识点掌握程度评估
- 学习能力综合评价
- 学习问题诊断与预警
- 班级整体学情分析报告
- 个性化学习建议生成

## 技术栈

- 后端框架：Flask 3.0.2
- 数据库：
  - MySQL：存储用户和学生数据
  - Neo4j：存储知识图谱
- 前端技术：HTML, CSS, JavaScript
- AI集成：OpenAI API
- 其他工具：
  - PDF生成：pdfkit
  - Markdown支持：markdown
  - 数据处理：pandas

## 项目结构

```
.
├── app.py                      # Flask应用主文件
├── best_path.py               # 最优学习路径计算
├── hard_path.py               # 困难路径计算
├── create_neo4j_database.py   # Neo4j数据库初始化
├── create_student_table.py    # MySQL学生表创建
├── student_generate.py        # 学生数据生成
├── prerequisite-dependency.csv # 课程先修依赖关系数据
├── requirements.txt           # 项目依赖
├── data/                      # 数据文件目录
├── static/                    # 静态资源目录
│   ├── css/                  # 样式文件
│   ├── js/                   # JavaScript文件
│   ├── lib/                  # 第三方库
│   ├── images/               # 图片资源
│   ├── fonts/                # 字体文件
│   └── icon/                 # 图标文件
├── templates/                 # HTML模板目录
│   ├── login.html            # 登录页面
│   ├── register.html         # 注册页面
│   ├── index.html            # 主页
│   ├── welcome.html          # 欢迎页
│   ├── student_info.html     # 学生信息页
│   ├── student_detail.html   # 学生分析详情页
│   ├── learning_path.html    # 学习路径页
│   ├── teaching_plan.html    # 教学计划页
│   └── learning_analysis.html # 学情分析页
└── utils/                     # 工具类目录
    ├── mysqlhelper.py        # MySQL数据库操作
    ├── db_config.py          # 数据库配置
    └── db_dbutils_init.py    # 数据库连接池初始化
```

## 安装说明

1. 克隆项目到本地
```bash
git clone [项目地址]
```

2. 安装依赖
```bash
pip install -r requirements.txt
```

3. 配置数据库
- 确保MySQL服务已启动
- 确保Neo4j服务已启动（默认端口：7687）
- 配置数据库连接信息

4. 配置环境变量
- 设置OpenAI API密钥
- 配置其他必要的环境变量

5. 初始化数据库
```bash
python create_student_table.py
python create_neo4j_database.py
```

## 使用方法

1. 启动应用
```bash
python app.py
```

2. 访问系统
- 打开浏览器访问 `http://localhost:5000`
- 使用注册的账号登录系统

3. 主要功能入口
- 学生信息管理：/student_info
- 学习路径规划：/learning_path
- 教学计划生成：/teaching_plan
- 学习分析：/learning_analysis

## 注意事项

- 确保所有依赖服务（MySQL、Neo4j）正常运行
- 定期备份数据库
- 妥善保管API密钥等敏感信息

## 贡献指南

欢迎提交Issue和Pull Request来帮助改进项目。

## 许可证

[待补充]
