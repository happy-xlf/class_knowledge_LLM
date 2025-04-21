import csv
import random

# 课程名称列表
course_names = ['计算机基础', '数据结构', '算法分析']
# 已掌握的知识点
mastered_knowledge = 'K_计算机科学_计算机科学技术'
# 性别列表
genders = ['男', '女']
# 学习风格列表
learning_styles = ['同化者', '发散者', '收敛者', '适应者']

# 生成 100 条数据
data = []
for i in range(1, 101):
    name = f'学生{i}'
    gender = random.choice(genders)
    student_id = f'202500{i:03d}'
    learning_style = random.choice(learning_styles)
    homework_score = round(random.uniform(0, 100), 2)
    participation_score = round(random.uniform(0, 100), 2)
    unit_test1_score = round(random.uniform(0, 100), 2)
    unit_test2_score = round(random.uniform(0, 100), 2)
    final_exam_score = round(random.uniform(0, 100), 2)
    total_score = round(
        homework_score * 0.2 + participation_score * 0.1 + unit_test1_score * 0.2 + unit_test2_score * 0.2 +
        final_exam_score * 0.3, 2)
    course_name = random.choice(course_names)

    data.append([
        name, gender, student_id, learning_style, homework_score, participation_score, unit_test1_score,
        unit_test2_score, final_exam_score, total_score, mastered_knowledge, course_name
    ])

# 保存为 CSV 文件
with open('./data/student_data.csv', 'w', newline='', encoding='utf-8-sig') as csvfile:
    writer = csv.writer(csvfile)
    # 写入表头
    writer.writerow([
        'name', 'gender', 'student_id', 'learning_style', 'homework_score', 'participation_score',
        'unit_test1_score', 'unit_test2_score', 'final_exam_score', 'total_score', 'mastered_knowledge', 'course_name'
    ])
    # 写入数据
    writer.writerows(data)

print('数据已成功保存为 student_data.csv')
    