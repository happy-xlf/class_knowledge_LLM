import pandas as pd
from utils.mysqlhelper import MySqLHelper

# 读取Excel文件
df = pd.read_excel('data/模拟学生信息.xlsx')

# 创建数据库连接
db = MySqLHelper()

# 删除表
drop_sql = "DROP TABLE IF EXISTS student;"
db.execute(drop_sql)

# 创建表结构
create_table_sql = """
CREATE TABLE IF NOT EXISTS student (
    id INT PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(50) COMMENT '姓名',
    gender VARCHAR(10) COMMENT '性别',
    student_id VARCHAR(20) COMMENT '学号',
    learning_style VARCHAR(50) COMMENT '学习风格',
    regular_score FLOAT COMMENT '平时成绩',
    quiz_score FLOAT COMMENT '测验成绩',
    final_score FLOAT COMMENT '期末成绩',
    mastered_knowledge TEXT COMMENT '已掌握的知识点',
    course_name VARCHAR(100) COMMENT '课程名称'
);
"""

# 执行创建表的SQL语句
db.execute(create_table_sql)

# 准备插入数据的SQL语句
insert_sql = """
INSERT INTO student (name, gender, student_id, learning_style, regular_score, quiz_score, final_score, mastered_knowledge, course_name)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);
"""

# 插入数据
for _, row in df.iterrows():
    # 添加课程名称列，这里假设课程名称为"高等数学"
    row_data = list(row)
    print(row_data)
    db.insertone(insert_sql, tuple(row_data))

# 获取所有数据
select_sql = "SELECT * FROM student;"
result, count = db.selectall(select_sql)

print(f"成功导入 {count} 条数据！")
print("\n表内所有数据：")
for row in result:
    print(row) 