from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'your-secret-key'  # 用于session加密

@app.route('/', methods=['GET', 'POST'])
def login():
    return render_template('index.html')

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
@app.route('/error')
def error():
    return render_template('error.html')

@app.route('/welcome')
def welcome():
    return render_template('welcome.html')

if __name__ == '__main__':
    app.run(debug=True) 