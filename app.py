from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify, Response, stream_with_context, send_file
from werkzeug.security import generate_password_hash, check_password_hash
from utils.mysqlhelper import MySqLHelper
from py2neo import Node, Graph, Relationship, NodeMatcher
import json
import time
import requests
import markdown
import pdfkit
import os
import tempfile

graph = Graph("neo4j://localhost:7687", auth=("neo4j","fengge666"))
app = Flask(__name__)
app.secret_key = 'your-secret-key'  # 用于session加密

# 定义课程ID映射
COURSES = {
    '1': '计算机基础',
    '2': '数据结构',
    '3': '算法分析'
}

# 初始化数据库连接
db = MySqLHelper()

# 创建用户表
def create_user_table():
    sql = """
    CREATE TABLE IF NOT EXISTS users (
        id INT AUTO_INCREMENT PRIMARY KEY,
        username VARCHAR(50) UNIQUE NOT NULL,
        password VARCHAR(255) NOT NULL,
        email VARCHAR(100) UNIQUE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """
    db.execute(sql)

# 在应用启动时创建表
# create_user_table()

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        # 查询用户
        sql = "SELECT * FROM users WHERE username = %s"
        user = db.selectone(sql, (username,))
        
        if user and check_password_hash(user[2], password):  # user[2]是password字段
            session['user_id'] = user[0]
            session['username'] = user[1]
            flash('登录成功！', 'success')
            return redirect(url_for('index'))
        else:
            flash('用户名或密码错误！', 'error')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        email = request.form.get('email')
        
        # 检查用户名是否已存在
        sql = "SELECT * FROM users WHERE username = %s"
        if db.selectone(sql, (username,)):
            flash('用户名已存在！', 'error')
            return redirect(url_for('register'))
        
        # 检查邮箱是否已存在
        if email:
            sql = "SELECT * FROM users WHERE email = %s"
            if db.selectone(sql, (email,)):
                flash('邮箱已被注册！', 'error')
                return redirect(url_for('register'))
        
        # 加密密码
        hashed_password = generate_password_hash(password)
        
        # 插入新用户
        sql = "INSERT INTO users (username, password, email) VALUES (%s, %s, %s)"
        result = db.insertone(sql, (username, hashed_password, email))
        
        if result:
            flash('注册成功！请登录', 'success')
            return redirect(url_for('login'))
        else:
            flash('注册失败，请重试！', 'error')
    
    return render_template('register.html')

# 其他页面路由
@app.route('/index')
def index():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('index.html')

@app.route('/welcome')
def welcome():
    return render_template('welcome.html')

# 学生信息页面
@app.route('/student_info')
def student_info():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('student_info.html')

# 学习路径页面
@app.route('/learning_path')
def learning_path():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('learning_path.html')

# 教学方案生成页面
@app.route('/teaching_plan')
def teaching_plan():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('teaching_plan.html')

# 学情分析评估页面
@app.route('/learning_analysis')
def learning_analysis():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('learning_analysis.html')

# 学生数据API
@app.route('/api/student_data')
def get_student_data():
    if 'username' not in session:
        return jsonify({'code': 401, 'msg': '未登录'})
    
    course_id = request.args.get('course_id', '1')
    if course_id not in COURSES:
        return jsonify({'code': 400, 'msg': '无效的课程ID'})
    
    # 获取课程名称
    course_name = COURSES[course_id]
    
    # 从数据库获取学生数据
    sql = """
        SELECT student_id, name, gender, learning_style, mastered_knowledge 
        FROM student 
        WHERE course_name = %s
    """
    
    try:
        result, count = db.selectall(sql, (course_name,))
        
        # 将查询结果转换为前端需要的格式
        student_data = []
        for row in result:
            student_data.append({
                'student_id': row[0],
                'name': row[1], 
                'gender': row[2],
                'learning_style': row[3],
                'mastered_knowledge': row[4]
            })
        
        return jsonify({
            'code': 0,
            'msg': '',
            'count': count,
            'data': student_data
        })
    except Exception as e:
        return jsonify({
            'code': 500,
            'msg': f'数据库查询错误: {str(e)}',
            'count': 0,
            'data': []
        })

# 删除学生信息API
@app.route('/api/delete_student', methods=['POST'])
def delete_student():
    if 'username' not in session:
        return jsonify({'code': 401, 'msg': '未登录'})
    
    try:
        data = request.get_json()
        student_id = data.get('student_id')
        
        if not student_id:
            return jsonify({'code': 400, 'msg': '缺少学生ID参数'})
        
        # 删除学生信息
        sql = "DELETE FROM student WHERE student_id = %s"
        result = db.execute(sql, (student_id,))
        
        if result:
            return jsonify({'code': 0, 'msg': '删除成功'})
        else:
            return jsonify({'code': 500, 'msg': '删除失败'})
            
    except Exception as e:
        return jsonify({'code': 500, 'msg': f'删除失败: {str(e)}'})

# 修改学生信息API
@app.route('/api/update_student', methods=['POST'])
def update_student():
    if 'username' not in session:
        return jsonify({'code': 401, 'msg': '未登录'})
    
    try:
        data = request.get_json()
        student_id = data.get('student_id')
        name = data.get('name')
        gender = data.get('gender')
        learning_style = data.get('learning_style')
        mastered_knowledge = data.get('mastered_knowledge')
        course_name = data.get('course_name')
        
        if not all([student_id, name, gender, learning_style, mastered_knowledge, course_name]):
            return jsonify({'code': 400, 'msg': '缺少必要参数'})
        
        # 更新学生信息
        sql = """
            UPDATE student 
            SET name = %s, 
                gender = %s, 
                learning_style = %s, 
                mastered_knowledge = %s,
                course_name = %s
            WHERE student_id = %s
        """
        result = db.execute(sql, (name, gender, learning_style, mastered_knowledge, course_name, student_id))
        
        if result:
            return jsonify({'code': 0, 'msg': '修改成功'})
        else:
            return jsonify({'code': 500, 'msg': '修改失败'})
            
    except Exception as e:
        return jsonify({'code': 500, 'msg': f'修改失败: {str(e)}'})

# 获取学生详细信息API
@app.route('/api/get_student_detail', methods=['GET'])
def get_student_detail():
    if 'username' not in session:
        return jsonify({'code': 401, 'msg': '未登录'})
    
    try:
        student_id = request.args.get('student_id')
        if not student_id:
            return jsonify({'code': 400, 'msg': '缺少学生ID参数'})
        
        # 查询学生详细信息
        sql = """
            SELECT student_id, name, gender, learning_style, mastered_knowledge, course_name
            FROM student 
            WHERE student_id = %s
        """
        result = db.selectone(sql, (student_id,))
        
        if result:
            return jsonify({
                'code': 0,
                'msg': '',
                'data': {
                    'student_id': result[0],
                    'name': result[1],
                    'gender': result[2],
                    'learning_style': result[3],
                    'mastered_knowledge': result[4],
                    'course_name': result[5]
                }
            })
        else:
            return jsonify({'code': 404, 'msg': '未找到该学生信息'})
            
    except Exception as e:
        return jsonify({'code': 500, 'msg': f'获取学生信息失败: {str(e)}'})

# 获取学习风格选项API
@app.route('/api/get_learning_styles', methods=['GET'])
def get_learning_styles():
    if 'username' not in session:
        return jsonify({'code': 401, 'msg': '未登录'})
    
    try:
        # 查询所有不重复的学习风格
        sql = "SELECT DISTINCT learning_style FROM student WHERE learning_style IS NOT NULL"
        result, count = db.selectall(sql)
        
        if result:
            learning_styles = [row[0] for row in result]
            return jsonify({
                'code': 0,
                'msg': '',
                'data': learning_styles
            })
        else:
            return jsonify({'code': 404, 'msg': '未找到学习风格数据'})
            
    except Exception as e:
        return jsonify({'code': 500, 'msg': f'获取学习风格失败: {str(e)}'})

# 获取课程选项API
@app.route('/api/get_courses', methods=['GET'])
def get_courses():
    if 'username' not in session:
        return jsonify({'code': 401, 'msg': '未登录'})
    
    try:
        # 查询所有不重复的课程
        sql = "SELECT DISTINCT course_name FROM student WHERE course_name IS NOT NULL"
        result, count = db.selectall(sql)
        
        if result:
            courses = [row[0] for row in result]
            return jsonify({
                'code': 0,
                'msg': '',
                'data': courses
            })
        else:
            return jsonify({'code': 404, 'msg': '未找到课程数据'})
            
    except Exception as e:
        return jsonify({'code': 500, 'msg': f'获取课程失败: {str(e)}'})

# 获取学习路径API
@app.route('/api/get_learning_path', methods=['GET'])
def get_learning_path():
    if 'username' not in session:
        return jsonify({'code': 401, 'msg': '未登录'})
    
    try:
        student_id = request.args.get('student_id')
        if not student_id:
            return jsonify({'code': 400, 'msg': '缺少学生ID参数'})
        
        # 获取学生已掌握的知识点
        sql = "SELECT mastered_knowledge, course_name FROM student WHERE student_id = %s"
        result = db.selectone(sql, (student_id,))
        
        if not result:
            return jsonify({'code': 404, 'msg': '未找到该学生信息'})
        
        mastered_knowledge = result[0]
        course_name = result[1]
        
        # 获取已掌握的知识点列表
        mastered_list = mastered_knowledge.split('，') if mastered_knowledge else []
        
        # 如果没有已掌握的知识点，从课程的第一个知识点开始
        if not mastered_list:
            # 查询课程的第一个知识点（没有先修知识点的知识点）
            cypher = """
                MATCH (n:计算机基础)
                WHERE NOT (n)<-[:属于]-(:计算机基础)
                RETURN n.name as name
                LIMIT 1
            """
            result = graph.run(cypher).data()
            if result:
                return jsonify({
                    'code': 0,
                    'msg': '',
                    'data': [result[0]['name']]
                })
            else:
                return jsonify({'code': 404, 'msg': '未找到课程知识点'})
        
        # 获取最后一个已掌握的知识点
        last_mastered = mastered_list[-1]
        
        # 查询下一个知识点
        cypher = """
            MATCH (n:计算机基础 {name: $name})<-[:属于]-(next:计算机基础)
            RETURN next.name as name
        """
        result = graph.run(cypher, name=last_mastered).data()
        data = []
        if result:
            for it in result:
                next_knowledge = it['name']
                data.append(next_knowledge)
            return jsonify({
                'code': 0,
                'msg': '',
                'data': data
            })
        else:
            return jsonify({'code': 404, 'msg': '已学完所有知识点'})
        
    except Exception as e:
        return jsonify({'code': 500, 'msg': f'获取学习路径失败: {str(e)}'})

# 获取知识点关系图API
@app.route('/api/get_knowledge_graph', methods=['GET'])
def get_knowledge_graph():
    if 'username' not in session:
        return jsonify({'code': 401, 'msg': '未登录'})
    
    try:
        student_id = request.args.get('student_id')
        if not student_id:
            return jsonify({'code': 400, 'msg': '缺少学生ID参数'})
        
        # 获取学生已掌握的知识点
        sql = "SELECT mastered_knowledge FROM student WHERE student_id = %s"
        result = db.selectone(sql, (student_id,))
        
        if not result:
            return jsonify({'code': 404, 'msg': '未找到该学生信息'})
        
        mastered_knowledge = result[0]
        mastered_list = mastered_knowledge.split('，') if mastered_knowledge else []
        
        # 获取最后一个已掌握的知识点
        last_mastered = mastered_list[-1] if mastered_list else None
        
        # 使用py2neo查询相关节点
        if last_mastered:
            # 查询与最后一个知识点相关的所有节点
            cypher = """
                MATCH (n:计算机基础 {name: $name})
                OPTIONAL MATCH (n)-[r1:包含]->(next:计算机基础)
                OPTIONAL MATCH (prev:计算机基础)-[r2:包含]->(n)
                RETURN n, next, prev, r1, r2
            """
            result = graph.run(cypher, name=last_mastered).data()
            
            # 构建节点和边的数据
            nodes = set()
            links = []
            
            for row in result:
                # 添加中心节点
                if row['n']:
                    nodes.add((row['n']['name'], True))  # True表示已掌握
                
                # 添加下一个节点
                if row['next']:
                    nodes.add((row['next']['name'], False))  # False表示未掌握
                    links.append({
                        'source': row['n']['name'],
                        'target': row['next']['name'],
                        'type': '包含'
                    })
                
                # 添加前一个节点
                if row['prev']:
                    nodes.add((row['prev']['name'], True))  # True表示已掌握
                    links.append({
                        'source': row['prev']['name'],
                        'target': row['n']['name'],
                        'type': '包含'
                    })
            
            # 转换为前端需要的格式
            nodes_data = [{'id': name, 'name': name, 'mastered': mastered} for name, mastered in nodes]
            
            return jsonify({
                'code': 0,
                'msg': '',
                'data': {
                    'nodes': nodes_data,
                    'links': links
                }
            })
        else:
            # 如果没有已掌握的知识点，返回第一个知识点
            cypher = """
                MATCH (n:计算机基础)
                WHERE NOT (n)<-[:属于]-(:计算机基础)
                RETURN n.name as name
                LIMIT 1
            """
            result = graph.run(cypher).data()
            
            if result:
                return jsonify({
                    'code': 0,
                    'msg': '',
                    'data': {
                        'nodes': [{
                            'id': result[0]['name'],
                            'name': result[0]['name'],
                            'mastered': False
                        }],
                        'links': []
                    }
                })
            else:
                return jsonify({'code': 404, 'msg': '未找到课程知识点'})
        
    except Exception as e:
        return jsonify({'code': 500, 'msg': f'获取知识点关系图失败: {str(e)}'})

@app.route('/api/get_statistics', methods=['GET'])
def get_statistics():
    if 'username' not in session:
        return jsonify({'code': 401, 'msg': '未登录'})
    
    try:
        # 获取总学生数量
        sql_total = "SELECT COUNT(*) FROM student"
        result_total = db.selectone(sql_total)
        total_students = result_total[0] if result_total else 0
        
        # 获取各课程学生数量
        course_stats = []
        for course_id, course_name in COURSES.items():
            sql_course = "SELECT COUNT(*) FROM student WHERE course_name = %s"
            result_course = db.selectone(sql_course, (course_name,))
            count = result_course[0] if result_course else 0
            course_stats.append({
                'course_id': course_id,
                'course_name': course_name,
                'count': count
            })
        
        return jsonify({
            'code': 0,
            'msg': '',
            'data': {
                'total_students': total_students,
                'total_courses': len(COURSES),
                'course_stats': course_stats
            }
        })
    except Exception as e:
        return jsonify({
            'code': 500,
            'msg': f'获取统计数据失败: {str(e)}',
            'data': {
                'total_students': 0,
                'total_courses': 0,
                'course_stats': []
            }
        })

# 教学方案生成API
@app.route('/api/generate_teaching_plan', methods=['POST'])
def generate_teaching_plan():
    if 'username' not in session:
        return jsonify({'code': 401, 'msg': '未登录'})
    
    try:
        data = request.get_json()
        course_id = data.get('course')
        keywords = data.get('keywords')
        content = data.get('content')
        
        if not course_id or not keywords:
            return jsonify({'code': 400, 'msg': '缺少必要参数'})
        
        # 获取课程名称
        course_name = COURSES.get(course_id, '未知课程')
        
        # 构建提示词
        prompt = f"""
        你是一位专业的教育专家，请根据以下信息生成一份详细的教学方案：
        
        课程：{course_name}
        关键词：{keywords}
        
        教案文本：
        {content}
        
        请生成一份包含以下内容的教学方案：
        1. 教学目标
        2. 教学重点和难点
        3. 教学过程（包括导入、讲解、练习、总结等环节）
        4. 教学资源
        5. 教学评价
        
        请确保教学方案详细、实用，并符合教育规律。
        """
        
        # 调用DeepSeek API进行流式生成
        # 注意：这里使用的是模拟的流式响应，实际使用时需要替换为真实的API调用
        def generate():
            # 模拟流式响应
            response_text = f"""
            # {course_name}教学方案
            
            ## 教学目标
            通过本课程的学习，学生将掌握{keywords}相关的核心概念和基本原理，能够运用所学知识解决实际问题。
            
            ## 教学重点和难点
            - 重点：{keywords.split('，')[0]}的基本原理和应用
            - 难点：{keywords.split('，')[1] if '，' in keywords else '相关知识的综合运用'}
            
            ## 教学过程
            
            ### 1. 导入环节（10分钟）
            通过提问或案例引入，激发学生的学习兴趣，引导学生思考{keywords.split('，')[0]}的重要性。
            
            ### 2. 讲解环节（30分钟）
            详细讲解{keywords}的基本概念、原理和应用，结合实例进行说明。
            
            ### 3. 练习环节（20分钟）
            提供相关练习题，让学生巩固所学知识，加深理解。
            
            ### 4. 总结环节（10分钟）
            总结本节课的重点内容，强调{keywords.split('，')[0]}的重要性，布置课后作业。
            
            ## 教学资源
            - 教材：{course_name}教材
            - 多媒体：PPT课件、相关视频
            - 实验环境：计算机实验室
            
            ## 教学评价
            - 课堂表现：参与度、回答问题的准确性
            - 作业完成情况：课后作业的完成质量和及时性
            - 测试成绩：单元测试和期末考试的成绩
            """
            
            # 模拟流式输出
            for i in range(0, len(response_text), 5):
                yield response_text[i:i+5]
                time.sleep(0.05)  # 模拟延迟
        
        # 返回流式响应
        return Response(stream_with_context(generate()), content_type='text/plain')
        
    except Exception as e:
        return jsonify({'code': 500, 'msg': f'生成教学方案失败: {str(e)}'})

# 导出PDF API
@app.route('/api/export_pdf', methods=['POST'])
def export_pdf():
    if 'username' not in session:
        return jsonify({'code': 401, 'msg': '未登录'})
    
    try:
        data = request.get_json()
        markdown_content = data.get('content')
        course_name = data.get('course_name', '教学方案')
        
        if not markdown_content:
            return jsonify({'code': 400, 'msg': '缺少内容参数'})
        
        # 将Markdown转换为HTML
        html_content = markdown.markdown(markdown_content, extensions=['tables', 'fenced_code', 'codehilite'])
        
        # 添加CSS样式
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>{course_name}</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    margin: 40px;
                }}
                h1 {{
                    color: #333;
                    border-bottom: 1px solid #ddd;
                    padding-bottom: 10px;
                }}
                h2 {{
                    color: #444;
                    margin-top: 20px;
                }}
                h3 {{
                    color: #555;
                }}
                code {{
                    background-color: #f5f5f5;
                    padding: 2px 5px;
                    border-radius: 3px;
                }}
                pre {{
                    background-color: #f5f5f5;
                    padding: 10px;
                    border-radius: 5px;
                    overflow-x: auto;
                }}
                table {{
                    border-collapse: collapse;
                    width: 100%;
                    margin: 20px 0;
                }}
                th, td {{
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                }}
                th {{
                    background-color: #f2f2f2;
                }}
                blockquote {{
                    border-left: 4px solid #ddd;
                    padding-left: 10px;
                    color: #666;
                }}
            </style>
        </head>
        <body>
            <h1>{course_name}</h1>
            {html_content}
        </body>
        </html>
        """
        
        # 创建临时文件
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
            temp_path = temp_file.name
        
        # 配置pdfkit选项
        options = {
            'page-size': 'A4',
            'margin-top': '20mm',
            'margin-right': '20mm',
            'margin-bottom': '20mm',
            'margin-left': '20mm',
            'encoding': 'UTF-8',
            'no-outline': None
        }
        
        # 生成PDF
        pdfkit.from_string(html_template, temp_path, options=options)
        
        # 返回PDF文件
        return send_file(
            temp_path,
            as_attachment=True,
            download_name=f"{course_name}_{time.strftime('%Y%m%d')}.pdf",
            mimetype='application/pdf'
        )
        
    except Exception as e:
        return jsonify({'code': 500, 'msg': f'生成PDF失败: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5999) 