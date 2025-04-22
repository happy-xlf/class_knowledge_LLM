1. 申请豆包pro的api：doubao-pro-32k-241215
2. 创建数据库：class_knowledge
3. 下载代码：git clone https://github.com/zhengyuxiang/class_knowledge.git
4. 安装依赖：pip install -r requirements.txt
5. 修改mysql密码和neo4j密码：
6. mysql密码可见：utils/db_config.py, DB_TEST_PASSWORD = "12345678" # 修改为你的密码
7. neo4j密码可见：app.py, graph = Graph("neo4j://localhost:7687", auth=("neo4j","fengge666")) # 修改为你的密码
8. 初始化数据库
```bash
python create_student_table.py
python create_neo4j_database.py
```
9. 启动服务
创建一个run.sh文件，内容如下：
```bash
#设定api_key
export API_KEY="豆包的api_key"

#启动flask
python app.py
```
10. 访问：http://127.0.0.1:5999