import numpy as np
import random
import matplotlib.pyplot as plt
import networkx as nx
from typing import List, Dict, Tuple

class KnowledgeGraphACO:
    def __init__(self, 
                 knowledge_graph: Dict[str, Dict[str, float]],
                 importance: Dict[str, float],
                 difficulty: Dict[str, float],
                 n_ants: int = 20,
                 alpha: float = 1.0,  # 信息素重要性系数
                 beta: float = 2.0,   # 启发式信息重要性系数
                 gamma: float = 1.0,  # 知识点重要性系数
                 rho: float = 0.1,    # 信息素挥发系数
                 q0: float = 0.5,     # 知识探索与利用平衡参数
                 n_iterations: int = 100):
        
        # 初始化图结构
        self.graph = knowledge_graph
        self.nodes = list(knowledge_graph.keys())
        self.n_nodes = len(self.nodes)
        self.importance = importance
        self.difficulty = difficulty
        
        # 初始化蚁群参数
        self.n_ants = n_ants
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.rho = rho
        self.q0 = q0
        self.n_iterations = n_iterations
        
        # 初始化邻接矩阵和信息素矩阵
        self.adjacency_matrix = self._create_adjacency_matrix()
        self.pheromone_matrix = np.ones((self.n_nodes, self.n_nodes)) * 0.1
        self.heuristic_matrix = self._create_heuristic_matrix()
        
        # 计算图的拓扑排序，确保学习顺序尊重先修关系
        self.topological_order = self._topological_sort()
        
        # 最优路径
        self.best_path = None
        self.best_path_score = float('-inf')
    
    def _create_adjacency_matrix(self) -> np.ndarray:
        """创建邻接矩阵"""
        matrix = np.zeros((self.n_nodes, self.n_nodes))
        for i, node_i in enumerate(self.nodes):
            for j, node_j in enumerate(self.nodes):
                if node_j in self.graph.get(node_i, {}):
                    matrix[i][j] = self.graph[node_i][node_j]
        return matrix
    
    def _create_heuristic_matrix(self) -> np.ndarray:
        """创建启发式矩阵，考虑知识点重要性和学习难度"""
        matrix = np.zeros((self.n_nodes, self.n_nodes))
        for i, node_i in enumerate(self.nodes):
            for j, node_j in enumerate(self.nodes):
                if self.adjacency_matrix[i][j] > 0:
                    # 启发式值 = 重要性 / 难度
                    matrix[i][j] = self.importance.get(node_j, 1.0) / self.difficulty.get(node_j, 1.0)
        return matrix
    
    def _topological_sort(self) -> List[int]:
        """对图进行拓扑排序，确保学习路径尊重前置知识关系"""
        visited = [False] * self.n_nodes
        temp = [False] * self.n_nodes
        order = []
        
        def dfs(node):
            if temp[node]:
                # 检测到环，知识图谱中不应该有环
                raise ValueError("Knowledge graph contains cycles, which is not allowed.")
            if not visited[node]:
                temp[node] = True
                
                # 探索所有依赖这个知识点的节点
                for i in range(self.n_nodes):
                    if self.adjacency_matrix[node][i] > 0:
                        dfs(i)
                
                temp[node] = False
                visited[node] = True
                order.append(node)
        
        for i in range(self.n_nodes):
            if not visited[i]:
                dfs(i)
        
        return order[::-1]  # 反转获得正确的拓扑顺序
    
    def _calculate_path_score(self, path: List[int]) -> float:
        """计算学习路径的得分
        
        得分考虑以下因素：
        1. 路径遵循拓扑排序的程度
        2. 知识点的重要性总和
        3. 学习难度平滑过渡
        """
        score = 0
        
        # 检查是否遵循拓扑排序
        topo_dict = {node: idx for idx, node in enumerate(self.topological_order)}
        prev_topo_idx = -1
        violations = 0
        
        for i in range(len(path)):
            node = path[i]
            topo_idx = topo_dict[node]
            
            # 增加基于知识点重要性的分数
            score += self.importance.get(self.nodes[node], 1.0)
            
            # 减少基于难度的分数
            difficulty_penalty = self.difficulty.get(self.nodes[node], 1.0)
            score -= 0.5 * difficulty_penalty
            
            # 检查拓扑排序违反情况
            if topo_idx < prev_topo_idx:
                violations += 1
            prev_topo_idx = topo_idx
            
            # 如果有前一个节点，检查难度跳跃
            if i > 0:
                prev_node = path[i-1]
                prev_difficulty = self.difficulty.get(self.nodes[prev_node], 1.0)
                curr_difficulty = self.difficulty.get(self.nodes[node], 1.0)
                
                # 惩罚难度跳跃过大的情况
                if curr_difficulty - prev_difficulty > 1.5:
                    score -= 0.5 * (curr_difficulty - prev_difficulty)
        
        # 大幅惩罚拓扑违反
        score -= violations * 10
        
        return score
    
    def _select_next_node(self, ant_path: List[int], current_node: int,
                         available_nodes: List[int]) -> int:
        """根据蚁群算法选择下一个知识点"""
        if not available_nodes:
            return -1
        
        # 探索与利用平衡策略
        if random.random() < self.q0:
            # 贪婪选择 - 利用
            scores = []
            for next_node in available_nodes:
                # 计算信息素与启发式的组合得分
                pheromone = self.pheromone_matrix[current_node][next_node]
                heuristic = self.heuristic_matrix[current_node][next_node]
                if heuristic == 0:  # 如果没有直接连接，使用默认值
                    heuristic = 0.1
                
                score = (pheromone ** self.alpha) * (heuristic ** self.beta)
                scores.append((next_node, score))
            
            # 选择分数最高的节点
            next_node = max(scores, key=lambda x: x[1])[0]
            return next_node
        else:
            # 轮盘赌选择 - 探索
            total_score = 0
            probabilities = []
            
            for next_node in available_nodes:
                pheromone = self.pheromone_matrix[current_node][next_node]
                heuristic = self.heuristic_matrix[current_node][next_node]
                if heuristic == 0:
                    heuristic = 0.1
                
                score = (pheromone ** self.alpha) * (heuristic ** self.beta)
                total_score += score
                probabilities.append((next_node, score))
            
            if total_score == 0:
                return random.choice(available_nodes)
            
            # 轮盘赌选择
            r = random.random() * total_score
            cumsum = 0
            for node, score in probabilities:
                cumsum += score
                if cumsum >= r:
                    return node
            
            return probabilities[-1][0]  # 默认返回最后一个
    
    def _update_pheromones(self, ant_paths: List[List[int]], ant_scores: List[float]):
        """更新信息素矩阵"""
        # 信息素挥发
        self.pheromone_matrix = (1 - self.rho) * self.pheromone_matrix
        
        # 增加新的信息素
        for path, score in zip(ant_paths, ant_scores):
            if score <= 0:  # 避免负分数或零分数
                continue
                
            pheromone_deposit = score / 10.0  # 信息素增加量与路径得分成正比
            
            for i in range(len(path) - 1):
                self.pheromone_matrix[path[i]][path[i+1]] += pheromone_deposit
    
    def optimize(self):
        """运行蚁群优化算法寻找最优学习路径"""
        for iteration in range(self.n_iterations):
            ant_paths = []
            ant_scores = []
            
            for ant in range(self.n_ants):
                # 为每只蚂蚁随机选择起点，倾向于选择入度为0的节点
                start_candidates = [i for i in range(self.n_nodes)]
                current_node = random.choice(start_candidates)
                
                path = [current_node]
                available_nodes = list(range(self.n_nodes))
                available_nodes.remove(current_node)
                
                # 构建完整路径
                while available_nodes:
                    next_node = self._select_next_node(path, current_node, available_nodes)
                    if next_node == -1:
                        break
                    
                    path.append(next_node)
                    available_nodes.remove(next_node)
                    current_node = next_node
                
                # 计算路径分数
                path_score = self._calculate_path_score(path)
                
                ant_paths.append(path)
                ant_scores.append(path_score)
                
                # 更新全局最优解
                if path_score > self.best_path_score:
                    self.best_path_score = path_score
                    self.best_path = path.copy()
            
            # 更新信息素
            self._update_pheromones(ant_paths, ant_scores)
            
            # 打印每次迭代的最佳得分
            if (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}/{self.n_iterations}, Best score: {self.best_path_score:.2f}")
        
        return self.best_path, self.best_path_score
    
    def get_optimal_learning_path(self) -> List[str]:
        """返回最优学习路径中的知识点名称"""
        if not self.best_path:
            self.optimize()
        
        return [self.nodes[node_idx] for node_idx in self.best_path]
    
    def visualize_learning_path(self):
        """可视化最优学习路径"""
        if not self.best_path:
            self.optimize()
        
        G = nx.DiGraph()
        
        # 添加所有节点
        for i, node in enumerate(self.nodes):
            G.add_node(node)
        
        # 添加所有边
        for i, node_i in enumerate(self.nodes):
            for j, node_j in enumerate(self.nodes):
                if self.adjacency_matrix[i][j] > 0:
                    G.add_edge(node_i, node_j, weight=self.adjacency_matrix[i][j])
        
        # 设置节点位置
        pos = nx.spring_layout(G)
        
        # 绘制图
        plt.figure(figsize=(12, 10))
        
        # 绘制所有节点
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=500)
        
        # 绘制所有边
        nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
        
        # 高亮显示最优路径
        path_edges = []
        for i in range(len(self.best_path) - 1):
            node1 = self.nodes[self.best_path[i]]
            node2 = self.nodes[self.best_path[i+1]]
            path_edges.append((node1, node2))
        
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, 
                               width=3.0, alpha=1.0, edge_color='red')
        
        # 添加节点标签
        nx.draw_networkx_labels(G, pos)
        
        # 添加图例和标题
        plt.title("Optimal Knowledge Learning Path")
        
        # 在节点旁边标注顺序
        for i, node_idx in enumerate(self.best_path):
            node_name = self.nodes[node_idx]
            x, y = pos[node_name]
            plt.text(x + 0.1, y + 0.1, f"Step {i+1}", fontsize=9, 
                    bbox=dict(facecolor='white', alpha=0.7))
        
        plt.axis('off')
        plt.tight_layout()
        plt.show()


# 定义一个简单的计算机科学知识图谱
knowledge_graph = {
    "计算机基础": {"编程基础": 1.0, "数据结构基础": 0.8},
    "编程基础": {"Python编程": 1.0, "JavaScript基础": 0.9},
    "数据结构基础": {"数组和链表": 1.0, "栈和队列": 0.9},
    "Python编程": {"Web开发基础": 0.7, "数据分析基础": 0.9},
    "JavaScript基础": {"Web开发基础": 0.9, "前端框架": 0.8},
    "数组和链表": {"排序算法": 0.8, "搜索算法": 0.7},
    "栈和队列": {"递归": 0.8, "图论基础": 0.6},
    "排序算法": {"算法复杂度": 0.7, "高级排序": 0.9},
    "搜索算法": {"算法复杂度": 0.7, "图论算法": 0.6},
    "递归": {"动态规划": 0.9},
    "Web开发基础": {"前端框架": 0.7, "后端开发": 0.7},
    "数据分析基础": {"机器学习基础": 0.8, "数据可视化": 0.7},
    "算法复杂度": {"高级算法": 0.8},
    "图论基础": {"图论算法": 0.9},
    "前端框架": {"Web应用开发": 0.9},
    "后端开发": {"Web应用开发": 0.9, "数据库系统": 0.8},
    "机器学习基础": {"深度学习": 0.9, "自然语言处理": 0.7},
    "高级排序": {"高级算法": 0.7},
    "图论算法": {"高级算法": 0.8},
    "动态规划": {"高级算法": 0.9},
    "高级算法": {},
    "Web应用开发": {},
    "数据库系统": {},
    "深度学习": {},
    "自然语言处理": {},
    "数据可视化": {}
}

# 定义知识点重要性
importance = {
    "计算机基础": 5.0,
    "编程基础": 5.0,
    "数据结构基础": 4.5,
    "Python编程": 4.0,
    "JavaScript基础": 3.8,
    "数组和链表": 4.2,
    "栈和队列": 4.0,
    "排序算法": 3.8,
    "搜索算法": 3.8,
    "递归": 4.0,
    "Web开发基础": 3.9,
    "数据分析基础": 4.0,
    "算法复杂度": 3.5,
    "图论基础": 3.7,
    "前端框架": 3.8,
    "后端开发": 4.1,
    "机器学习基础": 4.3,
    "高级排序": 3.4,
    "图论算法": 3.6,
    "动态规划": 4.0,
    "高级算法": 4.2,
    "Web应用开发": 3.9,
    "数据库系统": 4.0,
    "深度学习": 4.5,
    "自然语言处理": 4.2,
    "数据可视化": 3.8
}

# 定义知识点学习难度
difficulty = {
    "计算机基础": 1.0,
    "编程基础": 1.5,
    "数据结构基础": 2.0,
    "Python编程": 2.0,
    "JavaScript基础": 2.2,
    "数组和链表": 2.5,
    "栈和队列": 2.7,
    "排序算法": 3.0,
    "搜索算法": 3.0,
    "递归": 3.5,
    "Web开发基础": 2.8,
    "数据分析基础": 3.0,
    "算法复杂度": 3.2,
    "图论基础": 3.5,
    "前端框架": 3.3,
    "后端开发": 3.4,
    "机器学习基础": 3.8,
    "高级排序": 3.7,
    "图论算法": 4.0,
    "动态规划": 4.2,
    "高级算法": 4.5,
    "Web应用开发": 3.8,
    "数据库系统": 3.3,
    "深度学习": 4.3,
    "自然语言处理": 4.4,
    "数据可视化": 3.0
}

# 初始化算法并运行
aco = KnowledgeGraphACO(
    knowledge_graph=knowledge_graph,
    importance=importance,
    difficulty=difficulty,
    n_ants=30,
    alpha=1.0,
    beta=2.5,
    gamma=1.0,
    rho=0.1,
    q0=0.7,
    n_iterations=100
)

# 运行优化
best_path, best_score = aco.optimize()

# 获取最优学习路径
optimal_path = aco.get_optimal_learning_path()
print("\nOptimal Learning Path:")
for i, topic in enumerate(optimal_path):
    print(f"{i+1}. {topic}")

# 可视化学习路径
aco.visualize_learning_path()




