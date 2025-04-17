from py2neo import Node, Graph, Relationship, NodeMatcher
graph = Graph("neo4j://localhost:7687", auth=("neo4j","fengge666"))

def clean_neo4j(): 
    graph.run("MATCH (n) DETACH DELETE n")
# graph.delete_all()


def create_node(label,property,value):
    node = Node(label,**{property:value})
    graph.create(node)
    return node

def create_relation(node1,node2,relation_type):
    relation = Relationship(node1,relation_type,node2)
    graph.create(relation)

def find(label,property,value):
    matcher = NodeMatcher(graph)
    node = matcher.match(label).where(**{property:value}).first()
    return node

def find_all_person():
    all_person = graph.nodes.match('Person').all()
    print(len(all_person))
    for it in all_person:
        print(f"name:{it['name']}")
    print("================================")

def get_csv_data():
    file_path = "./prerequisite-dependency.csv"
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()[1:]  # 跳过表头
        label = "计算机基础"
        for line in lines:
            course1, course2 = line.strip().split(',')
            # 创建课程节点
            node1 = find(label, 'name', course1)
            if not node1:
                node1 = create_node(label, 'name', course1)
            node2 = find(label, 'name', course2)
            if not node2:
                node2 = create_node(label, 'name', course2)
            # 创建先修关系
            create_relation(node1, node2, '包含')
            # 创建依赖关系
            create_relation(node2, node1, '属于')

clean_neo4j()
get_csv_data()

