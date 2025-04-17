from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.security import generate_password_hash, check_password_hash
from utils.mysqlhelper import MySqLHelper

app = Flask(__name__)
app.secret_key = 'your-secret-key'  # 用于session加密

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
create_user_table()

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

# 会员管理相关路由
@app.route('/member-list')
def member_list():
    return render_template('member-list.html')

@app.route('/member-list1')
def member_list1():
    return render_template('member-list1.html')

@app.route('/member-del')
def member_del():
    return render_template('member-del.html')

# 订单管理路由
@app.route('/order-list')
def order_list():
    return render_template('order-list.html')

# 分类管理路由
@app.route('/cate')
def cate():
    return render_template('cate.html')

# 城市联动路由
@app.route('/city')
def city():
    return render_template('city.html')

# 管理员管理相关路由
@app.route('/admin-list')
def admin_list():
    return render_template('admin-list.html')

@app.route('/admin-role')
def admin_role():
    return render_template('admin-role.html')

@app.route('/admin-cate')
def admin_cate():
    return render_template('admin-cate.html')

@app.route('/admin-rule')
def admin_rule():
    return render_template('admin-rule.html')

# 系统统计相关路由
@app.route('/echarts1')
def echarts1():
    return render_template('echarts1.html')

@app.route('/echarts2')
def echarts2():
    return render_template('echarts2.html')

@app.route('/echarts3')
def echarts3():
    return render_template('echarts3.html')

@app.route('/echarts4')
def echarts4():
    return render_template('echarts4.html')

@app.route('/echarts5')
def echarts5():
    return render_template('echarts5.html')

@app.route('/echarts6')
def echarts6():
    return render_template('echarts6.html')

@app.route('/echarts7')
def echarts7():
    return render_template('echarts7.html')

@app.route('/echarts8')
def echarts8():
    return render_template('echarts8.html')

# 图标字体路由
@app.route('/unicode')
def unicode():
    return render_template('unicode.html')

# 其他页面路由
@app.route('/index')
def index():
    if 'user_id' not in session:
        flash('请先登录！', 'error')
        return redirect(url_for('login'))
    return render_template('index.html')

@app.route('/error')
def error():
    return render_template('error.html')

@app.route('/welcome')
def welcome():
    return render_template('welcome.html')

if __name__ == '__main__':
    app.run(debug=True) 