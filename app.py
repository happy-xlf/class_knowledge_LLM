from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from utils.mysqlhelper import MySqLHelper
from py2neo import Node, Graph, Relationship, NodeMatcher

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

if __name__ == '__main__':
    app.run(debug=True) 