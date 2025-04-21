import numpy as np
import random
import matplotlib.pyplot as plt
import networkx as nx
from typing import List, Dict, Tuple, Set
import pandas as pd
from sklearn.cluster import KMeans
import seaborn as sns

import numpy as np
import random
import matplotlib.pyplot as plt
import networkx as nx
from typing import List, Dict, Tuple, Set
import pandas as pd
from sklearn.cluster import KMeans
import seaborn as sns

class AdaptiveKnowledgeGraphACO:
    def __init__(self, 
                 knowledge_graph: Dict[str, Dict[str, float]],
                 importance: Dict[str, float],
                 difficulty: Dict[str, float],
                 learning_time: Dict[str, float] = None,
                 dependencies: Dict[str, List[str]] = None,
                 learner_profile: Dict = None,
                 n_ants: int = 20,
                 alpha: float = 1.0,  # 信息素重要性系数
                 beta: float = 2.0,   # 启发式信息重要性系数
                 gamma: float = 1.0,  # 知识点重要性系数
                 delta: float = 0.5,  # 知识点与学习者兴趣匹配系数
                 rho: float = 0.1,    # 信息素挥发系数
                 q0: float = 0.5,     # 知识探索与利用平衡参数
                 n_iterations: int = 100,
                 time_constraint: float = float('inf')):
        
        # 初始化图结构
        self.graph = knowledge_graph
        self.nodes = list(knowledge_graph.keys())
        self.n_nodes = len(self.nodes)
        self.importance = importance
        self.difficulty = difficulty
        
        # 学习时间约束
        self.learning_time = learning_time or {node: 1.0 for node in self.nodes}
        self.time_constraint = time_constraint
        
        # 明确的依赖关系（如果提供）
        self.dependencies = dependencies or {}
        
        # 学习者个人信息
        self.learner_profile = learner_profile or {
            "mastered_topics": set(),  # 已掌握的知识点
            "learning_goal": set(),    # 学习目标
            "interests": {},           # 兴趣领域及权重
            "learning_style": "visual", # 学习风格 (visual, auditory, reading, kinesthetic)
            "available_time": float('inf'),  # 可用学习时间
            "difficulty_preference": "balanced"  # 难度偏好 (easy, balanced, challenging)
        }
        
        # 初始化蚁群参数
        self.n_ants = n_ants
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.rho = rho
        self.q0 = q0
        self.n_iterations = n_iterations
        
        # 初始化邻接矩阵和信息素矩阵
        self.adjacency_matrix = self._create_adjacency_matrix()
        self.pheromone_matrix = np.ones((self.n_nodes, self.n_nodes)) * 0.1
        self.heuristic_matrix = self._create_heuristic_matrix()
        
        # 计算图的拓扑排序，确保学习顺序尊重先修关系
        self.topological_order = self._topological_sort()
        
        # 根据已掌握知识和依赖关系确定可学习节点
        self.available_nodes = self._get_available_nodes()
        
        # 知识点聚类
        self.topic_clusters = self._cluster_knowledge_topics()
        
        # 最优路径
        self.best_path = None
        self.best_path_score = float('-inf')
        
        # 学习计划细节
        self.learning_plan_details = {}
    
    def _create_adjacency_matrix(self) -> np.ndarray:
        """创建邻接矩阵"""
        matrix = np.zeros((self.n_nodes, self.n_nodes))
        
        # 根据知识图谱建立基本邻接关系
        for i, node_i in enumerate(self.nodes):
            for j, node_j in enumerate(self.nodes):
                if node_j in self.graph.get(node_i, {}):
                    matrix[i][j] = self.graph[node_i][node_j]
        
        # 如果有明确的依赖关系，加入到邻接矩阵
        if self.dependencies:
            for node, deps in self.dependencies.items():
                if node in self.nodes:
                    i = self.nodes.index(node)
                    for dep in deps:
                        if dep in self.nodes:
                            j = self.nodes.index(dep)
                            # 依赖节点指向当前节点
                            matrix[j][i] = max(matrix[j][i], 1.0)
        
        return matrix
    
    def _create_heuristic_matrix(self) -> np.ndarray:
        """创建启发式矩阵，考虑知识点重要性、学习难度和学习者兴趣"""
        matrix = np.zeros((self.n_nodes, self.n_nodes))
        
        # 学习者兴趣偏好
        interests = self.learner_profile.get("interests", {})
        
        # 难度偏好调整系数
        difficulty_adjustment = {
            "easy": 1.5,      # 更偏好简单的内容
            "balanced": 1.0,  # 平衡难度
            "challenging": 0.7 # 更偏好有挑战性的内容
        }.get(self.learner_profile.get("difficulty_preference", "balanced"), 1.0)
        
        for i, node_i in enumerate(self.nodes):
            for j, node_j in enumerate(self.nodes):
                if self.adjacency_matrix[i][j] > 0:
                    # 计算基础启发式值 = 重要性 / 难度
                    importance_value = self.importance.get(node_j, 1.0)
                    difficulty_value = self.difficulty.get(node_j, 1.0) ** difficulty_adjustment
                    
                    # 基础启发式值
                    heuristic = importance_value / max(0.1, difficulty_value)
                    
                    # 考虑学习者兴趣
                    # 确定知识点所属的领域
                    for domain, weight in interests.items():
                        # 检查知识点名称中是否包含领域关键词
                        if domain.lower() in node_j.lower():
                            heuristic *= (1.0 + weight * self.delta)
                    
                    # 保存到矩阵
                    matrix[i][j] = heuristic
        
        return matrix
    
    def _topological_sort(self) -> List[int]:
        """对图进行拓扑排序，确保学习路径尊重前置知识关系"""
        visited = [False] * self.n_nodes
        temp = [False] * self.n_nodes
        order = []
        
        def dfs(node):
            if temp[node]:
                # 检测到环，尝试打破环路
                print(f"Warning: Cycle detected including node {self.nodes[node]}. Breaking the cycle.")
                return
            if not visited[node]:
                temp[node] = True
                
                # 探索所有依赖这个知识点的节点
                for i in range(self.n_nodes):
                    if self.adjacency_matrix[node][i] > 0:
                        dfs(i)
                
                temp[node] = False
                visited[node] = True
                order.append(node)
        
        # 首先添加入度为0的节点（没有依赖的基础知识点）
        in_degree = [0] * self.n_nodes
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                if self.adjacency_matrix[j][i] > 0:
                    in_degree[i] += 1
        
        # 对每个节点进行DFS
        for i in range(self.n_nodes):
            if not visited[i]:
                dfs(i)
        
        return order[::-1]  # 反转获得正确的拓扑顺序
    
    def _get_available_nodes(self) -> List[int]:
        """根据学习者已掌握的知识，确定当前可学习的知识点"""
        mastered_topics = self.learner_profile.get("mastered_topics", set())
        mastered_indices = set()
        
        # 将已掌握的知识点转换为索引
        for topic in mastered_topics:
            if topic in self.nodes:
                mastered_indices.add(self.nodes.index(topic))
        
        # 找出可学习的新知识点（所有依赖已满足）
        available = []
        for i in range(self.n_nodes):
            # 跳过已掌握的知识点
            if i in mastered_indices:
                continue
            
            # 检查所有前置依赖是否满足
            all_prerequisites_met = True
            for j in range(self.n_nodes):
                # 如果j是i的前置要求但还没掌握
                if self.adjacency_matrix[j][i] > 0 and j not in mastered_indices:
                    all_prerequisites_met = False
                    break
            
            # 如果所有前置条件都满足，则该知识点可学习
            if all_prerequisites_met:
                available.append(i)
        
        return available
    
    def _cluster_knowledge_topics(self, n_clusters=5) -> Dict[str, int]:
        """使用知识点特征对知识点进行聚类"""
        # 提取知识点特征：重要性、难度、学习时间
        features = []
        for node in self.nodes:
            importance = self.importance.get(node, 1.0)
            difficulty = self.difficulty.get(node, 1.0)
            time = self.learning_time.get(node, 1.0)
            
            # 可以添加更多特征
            features.append([importance, difficulty, time])
        
        # 特征标准化
        features = np.array(features)
        features_normalized = (features - features.mean(axis=0)) / features.std(axis=0)
        
        # 进行聚类
        n_clusters = min(n_clusters, len(self.nodes))  # 确保聚类数不超过节点数
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(features_normalized)
        
        # 返回节点到聚类的映射
        return {node: int(cluster) for node, cluster in zip(self.nodes, clusters)}
    
    def _calculate_path_similarity(self, path: List[int]) -> float:
        """计算路径中知识点的聚类相似性分数"""
        if len(path) <= 1:
            return 0
        
        # 计算路径中连续知识点之间的聚类相似性
        similarity_score = 0
        for i in range(len(path) - 1):
            node1 = self.nodes[path[i]]
            node2 = self.nodes[path[i+1]]
            
            # 如果连续的知识点属于同一聚类，增加相似性分数
            if self.topic_clusters[node1] == self.topic_clusters[node2]:
                similarity_score += 1
        
        # 归一化分数
        return similarity_score / (len(path) - 1)
    
    def _calculate_goal_alignment(self, path: List[int]) -> float:
        """计算路径与学习目标的一致性"""
        learning_goals = self.learner_profile.get("learning_goal", set())
        if not learning_goals:
            return 1.0  # 如果没有明确目标，则任何路径都视为一致
        
        # 计算路径中包含学习目标的知识点比例
        goal_nodes = set()
        for goal in learning_goals:
            if goal in self.nodes:
                goal_nodes.add(self.nodes.index(goal))
        
        # 路径中的目标知识点数量
        goals_in_path = len(set(path).intersection(goal_nodes))
        
        # 如果目标知识点都在路径中，则得分最高
        if goals_in_path == len(goal_nodes):
            return 1.0
        
        # 否则，根据覆盖比例计分
        return goals_in_path / len(goal_nodes)
    
    def _calculate_time_feasibility(self, path: List[int]) -> float:
        """计算路径时间可行性分数"""
        available_time = self.learner_profile.get("available_time", float('inf'))
        
        if available_time == float('inf'):
            return 1.0  # 无时间限制
        
        # 计算路径所需总时间
        total_time = sum(self.learning_time.get(self.nodes[node], 1.0) for node in path)
        
        # 如果总时间在可用时间内，则返回满分
        if total_time <= available_time:
            return 1.0
        
        # 否则，返回一个基于超出时间的惩罚分数
        return max(0.1, available_time / total_time)
    
    def _calculate_path_score(self, path: List[int]) -> float:
        """计算学习路径的综合得分
        
        得分考虑以下因素：
        1. 路径遵循拓扑排序的程度
        2. 知识点的重要性总和
        3. 学习难度平滑过渡
        4. 知识点聚类相似性（相关知识点连续学习）
        5. 与学习目标的一致性
        6. 时间可行性
        """
        if not path:
            return float('-inf')
        
        base_score = 0
        
        # 检查是否遵循拓扑排序
        topo_dict = {node: idx for idx, node in enumerate(self.topological_order)}
        prev_topo_idx = -1
        violations = 0
        
        # 根据难度偏好调整系数
        difficulty_preference = {
            "easy": 1.5,      # 更倾向于简单路径
            "balanced": 1.0,  # 平衡难度
            "challenging": 0.7 # 更倾向于有挑战的路径
        }.get(self.learner_profile.get("difficulty_preference", "balanced"), 1.0)
        
        for i in range(len(path)):
            node = path[i]
            node_name = self.nodes[node]
            topo_idx = topo_dict[node]
            
            # 增加基于知识点重要性的分数
            base_score += self.importance.get(node_name, 1.0)
            
            # 根据难度偏好调整难度惩罚
            difficulty_penalty = self.difficulty.get(node_name, 1.0) ** difficulty_preference
            base_score -= 0.5 * difficulty_penalty
            
            # 检查拓扑排序违反情况
            if topo_idx < prev_topo_idx:
                violations += 1
            prev_topo_idx = topo_idx
            
            # 如果有前一个节点，检查难度跳跃
            if i > 0:
                prev_node = path[i-1]
                prev_name = self.nodes[prev_node]
                prev_difficulty = self.difficulty.get(prev_name, 1.0)
                curr_difficulty = self.difficulty.get(node_name, 1.0)
                
                # 惩罚难度跳跃过大的情况
                if curr_difficulty - prev_difficulty > 1.5:
                    base_score -= 0.5 * (curr_difficulty - prev_difficulty)
        
        # 大幅惩罚拓扑违反
        base_score -= violations * 10
        
        # 计算其他分数组件
        similarity_score = self._calculate_path_similarity(path) * 5  # 相似性得分的权重
        goal_alignment = self._calculate_goal_alignment(path) * 8    # 目标一致性的权重
        time_feasibility = self._calculate_time_feasibility(path) * 7  # 时间可行性的权重
        
        # 组合所有分数
        total_score = base_score + similarity_score + goal_alignment + time_feasibility
        
        return total_score
    
    def _select_next_node(self, ant_path: List[int], current_node: int,
                        candidates: List[int]) -> int:
        """根据蚁群算法选择下一个知识点"""
        if not candidates:
            return -1
        
        # 探索与利用平衡策略
        if random.random() < self.q0:
            # 贪婪选择 - 利用
            scores = []
            for next_node in candidates:
                # 计算信息素与启发式的组合得分
                pheromone = self.pheromone_matrix[current_node][next_node]
                heuristic = self.heuristic_matrix[current_node][next_node]
                if heuristic == 0:  # 如果没有直接连接，使用默认值
                    heuristic = 0.1
                
                score = (pheromone ** self.alpha) * (heuristic ** self.beta)
                
                # 考虑知识点聚类相似性
                current_cluster = self.topic_clusters[self.nodes[current_node]]
                next_cluster = self.topic_clusters[self.nodes[next_node]]
                
                # 如果属于同一个聚类，稍微提高得分
                if current_cluster == next_cluster:
                    score *= 1.2
                
                scores.append((next_node, score))
            
            # 选择分数最高的节点
            if not scores:
                return random.choice(candidates) if candidates else -1
                
            next_node = max(scores, key=lambda x: x[1])[0]
            return next_node
        else:
            # 轮盘赌选择 - 探索
            total_score = 0
            probabilities = []
            
            for next_node in candidates:
                pheromone = self.pheromone_matrix[current_node][next_node]
                heuristic = self.heuristic_matrix[current_node][next_node]
                if heuristic == 0:
                    heuristic = 0.1
                
                # 基础分数
                score = (pheromone ** self.alpha) * (heuristic ** self.beta)
                
                # 考虑知识点聚类
                current_cluster = self.topic_clusters[self.nodes[current_node]]
                next_cluster = self.topic_clusters[self.nodes[next_node]]
                if current_cluster == next_cluster:
                    score *= 1.2
                
                total_score += score
                probabilities.append((next_node, score))
            
            if total_score == 0:
                return random.choice(candidates) if candidates else -1
            
            # 轮盘赌选择
            r = random.random() * total_score
            cumsum = 0
            for node, score in probabilities:
                cumsum += score
                if cumsum >= r:
                    return node
            
            return probabilities[-1][0] if probabilities else -1
    
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
    
    def _get_candidate_nodes(self, current_path: List[int]) -> List[int]:
        """获取适合作为下一步的候选知识点"""
        if not current_path:
            # 如果是起点，可以从任何可用节点开始
            return self.available_nodes
        
        current_node = current_path[-1]
        candidates = []
        
        # 已访问节点集合
        visited = set(current_path)
        
        # 首先考虑直接相连的节点
        for i in range(self.n_nodes):
            # 如果有连接且未访问过
            if self.adjacency_matrix[current_node][i] > 0 and i not in visited:
                # 检查是否满足所有前置条件
                all_prerequisites_met = True
                for j in range(self.n_nodes):
                    if self.adjacency_matrix[j][i] > 0 and j not in visited and j not in self.learner_profile.get("mastered_topics_idx", set()):
                        all_prerequisites_met = False
                        break
                
                if all_prerequisites_met:
                    candidates.append(i)
        
        # 如果没有直接相连的候选节点，考虑所有可用节点
        if not candidates:
            for i in self.available_nodes:
                if i not in visited:
                    candidates.append(i)
        
        return candidates
    
    def optimize(self):
        """运行蚁群优化算法寻找最优学习路径"""
        # 将已掌握的知识点转换为索引
        mastered_topics_idx = set()
        for topic in self.learner_profile.get("mastered_topics", set()):
            if topic in self.nodes:
                mastered_topics_idx.add(self.nodes.index(topic))
        self.learner_profile["mastered_topics_idx"] = mastered_topics_idx
        
        # 更新可用节点
        self.available_nodes = self._get_available_nodes()
        
        # 如果没有可用节点，返回空路径
        if not self.available_nodes:
            print("No available knowledge points to learn. All prerequisites not met.")
            return [], 0
        
        for iteration in range(self.n_iterations):
            ant_paths = []
            ant_scores = []
            
            for ant in range(self.n_ants):
                # 随机选择起点
                if self.available_nodes:
                    current_node = random.choice(self.available_nodes)
                    path = [current_node]
                    visited = {current_node}
                    
                    # 计算总学习时间
                    total_time = self.learning_time.get(self.nodes[current_node], 1.0)
                    available_time = self.learner_profile.get("available_time", float('inf'))
                    
                    # 构建路径
                    while True:
                        # 获取候选下一节点
                        candidates = self._get_candidate_nodes(path)
                        candidates = [c for c in candidates if c not in visited]
                        
                        # 如果没有候选节点或已达到时间限制，结束路径构建
                        if not candidates or total_time >= available_time:
                            break
                        
                        # 选择下一个节点
                        next_node = self._select_next_node(path, current_node, candidates)
                        if next_node == -1:
                            break
                        
                        # 更新路径和已访问集合
                        path.append(next_node)
                        visited.add(next_node)
                        current_node = next_node
                        
                        # 更新总时间
                        total_time += self.learning_time.get(self.nodes[next_node], 1.0)
                    
                    # 计算路径得分
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
        
        # 如果找到了路径，生成学习计划详情
        if self.best_path:
            self._generate_learning_plan_details()
        
        return self.best_path, self.best_path_score
    
    def _generate_learning_plan_details(self):
        """生成详细的学习计划"""
        if not self.best_path:
            return
        
        total_time = 0
        details = []
        
        for i, node_idx in enumerate(self.best_path):
            node_name = self.nodes[node_idx]
            time = self.learning_time.get(node_name, 1.0)
            importance = self.importance.get(node_name, 1.0)
            difficulty = self.difficulty.get(node_name, 1.0)
            
            # 找出前置知识点
            prerequisites = []
            for j in range(self.n_nodes):
                if self.adjacency_matrix[j][node_idx] > 0:
                    prereq_name = self.nodes[j]
                    if prereq_name not in self.learner_profile.get("mastered_topics", set()):
                        prerequisites.append(prereq_name)
            
            # 找出后续可学习的知识点
            next_topics = []
            for j in range(self.n_nodes):
                if self.adjacency_matrix[node_idx][j] > 0:
                    next_topics.append(self.nodes[j])
            
            # 累计学习时间
            total_time += time
            
            details.append({
                "step": i + 1,
                "topic": node_name,
                "estimated_time": time,
                "cumulative_time": total_time,
                "importance": importance,
                "difficulty": difficulty,
                "prerequisites": prerequisites,
                "next_topics": next_topics,
                "cluster": self.topic_clusters[node_name]
            })
        
        self.learning_plan_details = details
    
    def get_optimal_learning_path(self) -> List[str]:
        """返回最优学习路径中的知识点名称"""
        if not self.best_path:
            self.optimize()
        
        return [self.nodes[node_idx] for node_idx in self.best_path]
    
    def get_learning_plan_details(self) -> List[Dict]:
        """返回详细的学习计划"""
        if not self.learning_plan_details and self.best_path:
            self._generate_learning_plan_details()
        
        return self.learning_plan_details
    
    def predict_next_topics(self, current_topics: Set[str], n: int = 3) -> List[str]:
        """基于当前已学习的知识点，预测下一步应该学习的知识点"""
        # 更新已掌握的知识点
        mastered = self.learner_profile.get("mastered_topics", set()).union(current_topics)
        
        # 创建临时学习者画像用于预测
        temp_profile = self.learner_profile.copy()
        temp_profile["mastered_topics"] = mastered
        
        # 使用临时配置运行一次优化
        temp_aco = AdaptiveKnowledgeGraphACO(
            knowledge_graph=self.graph,
            importance=self.importance,
            difficulty=self.difficulty,
            learning_time=self.learning_time,
            dependencies=self.dependencies,
            learner_profile=temp_profile,
            n_ants=10,  # 减少蚂蚁数量以加速计算
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
            delta=self.delta,
            rho=self.rho,
            q0=self.q0,
            n_iterations=20,  # 减少迭代次数以加速计算
            time_constraint=self.time_constraint
        )
        
        # 运行优化算法
        _, _ = temp_aco.optimize()
        
        # 获取预测的下一步学习路径
        prediction = temp_aco.get_optimal_learning_path()
        
        # 返回前n个预测结果
        return prediction[:n]
    
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

# 1. 创建知识图谱
ml_knowledge_graph = {
    "数学基础": {"线性代数": 1.0, "微积分": 0.9, "概率与统计": 1.0},
    "线性代数": {"矩阵运算": 1.0, "特征值与特征向量": 0.8},
    "微积分": {"导数与梯度": 1.0, "多元微积分": 0.9},
    "概率与统计": {"概率分布": 1.0, "统计推断": 0.9, "贝叶斯理论": 0.8},
    "编程基础": {"Python基础": 1.0, "数据结构": 0.8},
    "Python基础": {"NumPy": 0.9, "Pandas": 0.9, "Matplotlib": 0.7},
    "数据结构": {"算法": 0.8},
    "矩阵运算": {"NumPy": 0.7},
    "导数与梯度": {"梯度下降": 0.9},
    "多元微积分": {"梯度下降": 0.8},
    "概率分布": {"统计推断": 0.7},
    "NumPy": {"数据预处理": 0.8},
    "Pandas": {"数据预处理": 1.0, "数据可视化": 0.8},
    "Matplotlib": {"数据可视化": 1.0},
    "数据预处理": {"特征工程": 0.9, "数据可视化": 0.6},
    "特征工程": {"监督学习": 0.8, "无监督学习": 0.7},
    "梯度下降": {"监督学习": 0.9},
    "贝叶斯理论": {"朴素贝叶斯": 0.9},
    "监督学习": {"线性回归": 0.8, "逻辑回归": 0.8, "决策树": 0.7, "随机森林": 0.6},
    "无监督学习": {"聚类算法": 0.9, "降维算法": 0.8},
    "线性回归": {"模型评估": 0.7, "正则化": 0.6},
    "逻辑回归": {"模型评估": 0.7, "正则化": 0.6},
    "决策树": {"随机森林": 0.9, "模型评估": 0.7},
    "随机森林": {"集成学习": 0.9, "模型评估": 0.7},
    "聚类算法": {"K均值聚类": 0.9, "层次聚类": 0.7},
    "降维算法": {"主成分分析": 1.0, "t-SNE": 0.8},
    "朴素贝叶斯": {"模型评估": 0.7, "文本分类": 0.8},
    "模型评估": {"交叉验证": 0.8, "过拟合与欠拟合": 0.9},
    "正则化": {"过拟合与欠拟合": 0.9},
    "集成学习": {"梯度提升": 0.9, "堆叠集成": 0.7},
    "K均值聚类": {"应用实践": 0.7},
    "主成分分析": {"应用实践": 0.7},
    "交叉验证": {"超参数调优": 0.9},
    "过拟合与欠拟合": {"超参数调优": 0.8},
    "梯度提升": {"XGBoost": 0.9, "LightGBM": 0.8},
    "文本分类": {"自然语言处理": 0.9},
    "超参数调优": {"网格搜索": 0.8, "随机搜索": 0.7, "贝叶斯优化": 0.6},
    "自然语言处理": {"词嵌入": 0.9, "RNN": 0.8},
    "XGBoost": {"应用实践": 0.8},
    "词嵌入": {"Word2Vec": 0.9, "GloVe": 0.8},
    "RNN": {"LSTM": 0.9, "GRU": 0.8},
    "LSTM": {"深度学习": 0.7},
    "应用实践": {"机器学习项目": 0.9},
    "深度学习": {"神经网络": 0.9, "CNN": 0.8, "迁移学习": 0.7},
    "神经网络": {"前馈神经网络": 0.9, "反向传播": 0.9},
    "CNN": {"计算机视觉": 0.9},
    "计算机视觉": {"图像分类": 0.8, "目标检测": 0.7},
    "图像分类": {"机器学习项目": 0.7},
    "目标检测": {"机器学习项目": 0.7},
    "机器学习项目": {"部署与生产": 0.8},
    "部署与生产": {"模型监控": 0.9, "A/B测试": 0.7},
    # 以下是没有后续节点的终端知识点
    "矩阵运算": {},
    "特征值与特征向量": {},
    "统计推断": {},
    "算法": {},
    "层次聚类": {},
    "t-SNE": {},
    "堆叠集成": {},
    "LightGBM": {},
    "网格搜索": {},
    "随机搜索": {},
    "贝叶斯优化": {},
    "Word2Vec": {},
    "GloVe": {},
    "GRU": {},
    "前馈神经网络": {},
    "反向传播": {},
    "迁移学习": {},
    "模型监控": {},
    "A/B测试": {}
}

# 2. 定义知识点的重要性
importance = {
    "数学基础": 5.0,
    "线性代数": 4.5,
    "微积分": 4.5,
    "概率与统计": 4.7,
    "编程基础": 5.0,
    "Python基础": 4.8,
    "数据结构": 4.3,
    "矩阵运算": 4.0,
    "特征值与特征向量": 3.8,
    "导数与梯度": 4.2,
    "多元微积分": 4.0,
    "概率分布": 4.4,
    "统计推断": 4.2,
    "贝叶斯理论": 4.0,
    "NumPy": 4.6,
    "Pandas": 4.7,
    "Matplotlib": 4.0,
    "算法": 4.2,
    "数据预处理": 4.8,
    "特征工程": 4.5,
    "数据可视化": 4.3,
    "梯度下降": 4.4,
    "朴素贝叶斯": 3.8,
    "监督学习": 4.6,
    "无监督学习": 4.2,
    "线性回归": 4.3,
    "逻辑回归": 4.3,
    "决策树": 4.0,
    "随机森林": 4.1,
    "聚类算法": 3.9,
    "降维算法": 3.7,
    "K均值聚类": 3.8,
    "层次聚类": 3.3,
    "主成分分析": 3.9,
    "t-SNE": 3.5,
    "模型评估": 4.5,
    "正则化": 4.2,
    "集成学习": 4.0,
    "梯度提升": 3.9,
    "堆叠集成": 3.5,
    "文本分类": 3.8,
    "交叉验证": 4.3,
    "过拟合与欠拟合": 4.4,
    "超参数调优": 4.1,
    "XGBoost": 3.8,
    "LightGBM": 3.7,
    "网格搜索": 3.6,
    "随机搜索": 3.5,
    "贝叶斯优化": 3.4,
    "自然语言处理": 4.2,
    "词嵌入": 4.0,
    "RNN": 3.9,
    "Word2Vec": 3.7,
    "GloVe": 3.6,
    "LSTM": 3.9,
    "GRU": 3.7,
    "应用实践": 4.6,
    "深度学习": 4.5,
    "神经网络": 4.3,
    "CNN": 4.2,
    "前馈神经网络": 4.0,
    "反向传播": 4.1,
    "计算机视觉": 4.2,
    "迁移学习": 3.8,
    "图像分类": 4.0,
    "目标检测": 3.9,
    "机器学习项目": 4.7,
    "部署与生产": 4.4,
    "模型监控": 4.2,
    "A/B测试": 4.0
}

# 3. 定义知识点的学习难度
difficulty = {
    "数学基础": 2.0,
    "线性代数": 3.0,
    "微积分": 3.5,
    "概率与统计": 3.2,
    "编程基础": 1.5,
    "Python基础": 2.0,
    "数据结构": 3.0,
    "矩阵运算": 3.2,
    "特征值与特征向量": 3.8,
    "导数与梯度": 3.5,
    "多元微积分": 4.0,
    "概率分布": 3.3,
    "统计推断": 3.5,
    "贝叶斯理论": 3.8,
    "NumPy": 2.5,
    "Pandas": 2.7,
    "Matplotlib": 2.5,
    "算法": 3.5,
    "数据预处理": 3.0,
    "特征工程": 3.5,
    "数据可视化": 2.8,
    "梯度下降": 3.5,
    "朴素贝叶斯": 3.2,
    "监督学习": 3.3,
    "无监督学习": 3.5,
    "线性回归": 2.8,
    "逻辑回归": 3.0,
    "决策树": 3.2,
    "随机森林": 3.3,
    "聚类算法": 3.5,
    "降维算法": 3.8,
    "K均值聚类": 3.0,
    "层次聚类": 3.4,
    "主成分分析": 3.7,
    "t-SNE": 4.0,
    "模型评估": 3.2,
    "正则化": 3.5,
    "集成学习": 3.7,
    "梯度提升": 3.9,
    "堆叠集成": 4.0,
    "文本分类": 3.5,
    "交叉验证": 3.2,
    "过拟合与欠拟合": 3.4,
    "超参数调优": 3.6,
    "XGBoost": 3.8,
    "LightGBM": 3.8,
    "网格搜索": 3.2,
    "随机搜索": 3.0,
    "贝叶斯优化": 4.0,
    "自然语言处理": 4.0,
    "词嵌入": 3.8,
    "RNN": 4.2,
    "Word2Vec": 3.9,
    "GloVe": 4.0,
    "LSTM": 4.3,
    "GRU": 4.3,
    "应用实践": 3.5,
    "深度学习": 4.2,
    "神经网络": 4.0,
    "CNN": 4.1,
    "前馈神经网络": 4.0,
    "反向传播": 4.2,
    "计算机视觉": 4.0,
    "迁移学习": 4.0,
    "图像分类": 3.8,
    "目标检测": 4.2,
    "机器学习项目": 3.7,
    "部署与生产": 3.5,
    "模型监控": 3.3,
    "A/B测试": 3.4
}

# 4. 定义知识点的学习时间（小时）
learning_time = {
    "数学基础": 5.0,
    "线性代数": 10.0,
    "微积分": 12.0,
    "概率与统计": 10.0,
    "编程基础": 6.0,
    "Python基础": 8.0,
    "数据结构": 10.0,
    "矩阵运算": 6.0,
    "特征值与特征向量": 7.0,
    "导数与梯度": 5.0,
    "多元微积分": 8.0,
    "概率分布": 6.0,
    "统计推断": 8.0,
    "贝叶斯理论": 7.0,
    "NumPy": 4.0,
    "Pandas": 5.0,
    "Matplotlib": 3.0,
    "算法": 12.0,
    "数据预处理": 6.0,
    "特征工程": 8.0,
    "数据可视化": 5.0,
    "梯度下降": 6.0,
    "朴素贝叶斯": 4.0,
    "监督学习": 7.0,
    "无监督学习": 7.0,
    "线性回归": 4.0,
    "逻辑回归": 4.0,
    "决策树": 5.0,
    "随机森林": 5.0,
    "聚类算法": 6.0,
    "降维算法": 6.0,
    "K均值聚类": 3.0,
    "层次聚类": 4.0,
    "主成分分析": 5.0,
    "t-SNE": 4.0,
    "模型评估": 6.0,
    "正则化": 4.0,
    "集成学习": 6.0,
    "梯度提升": 5.0,
    "堆叠集成": 4.0,
    "文本分类": 6.0,
    "交叉验证": 3.0,
    "过拟合与欠拟合": 4.0,
    "超参数调优": 5.0,
    "XGBoost": 4.0,
    "LightGBM": 4.0,
    "网格搜索": 2.0,
    "随机搜索": 2.0,
    "贝叶斯优化": 3.0,
    "自然语言处理": 10.0,
    "词嵌入": 6.0,
    "RNN": 8.0,
    "Word2Vec": 4.0,
    "GloVe": 4.0,
    "LSTM": 7.0,
    "GRU": 6.0,
    "应用实践": 10.0,
    "深度学习": 12.0,
    "神经网络": 8.0,
    "CNN": 9.0,
    "前馈神经网络": 6.0,
    "反向传播": 7.0,
    "计算机视觉": 10.0,
    "迁移学习": 6.0,
    "图像分类": 7.0,
    "目标检测": 8.0,
    "机器学习项目": 15.0,
    "部署与生产": 10.0,
    "模型监控": 5.0,
    "A/B测试": 4.0
}

# 5. 定义明确的依赖关系（可选，如果知识图谱已经包含依赖关系，可以不定义）
dependencies = {
    "线性回归": ["线性代数", "梯度下降"],
    "逻辑回归": ["线性代数", "梯度下降"],
    "决策树": ["概率与统计"],
    "LSTM": ["RNN", "梯度下降"],
    "CNN": ["深度学习", "梯度下降"],
    "深度学习": ["神经网络", "梯度下降", "Python基础", "NumPy"]
}

# 6. 创建不同的学习者画像进行测试

# 测试场景1：初学者，刚开始学习机器学习
beginner_profile = {
    "mastered_topics": {"数学基础", "编程基础", "Python基础"},
    "learning_goal": {"机器学习项目", "监督学习", "数据预处理"},
    "interests": {
        "数据科学": 0.9,
        "机器学习": 0.8,
        "Python": 0.7
    },
    "learning_style": "visual",
    "available_time": 120,  # 总共可用120小时
    "difficulty_preference": "balanced"  # 平衡的难度偏好
}

# 测试场景2：中级学习者，想要学习深度学习
intermediate_profile = {
    "mastered_topics": {
        "数学基础", "编程基础", "Python基础", "线性代数", "微积分", "概率与统计",
        "NumPy", "Pandas", "数据预处理", "特征工程", "监督学习", "线性回归", "逻辑回归"
    },
    "learning_goal": {"深度学习", "CNN", "自然语言处理"},
    "interests": {
        "深度学习": 0.9,
        "计算机视觉": 0.7,
        "自然语言处理": 0.8
    },
    "learning_style": "reading",
    "available_time": 80,  # 总共可用80小时
    "difficulty_preference": "challenging"  # 喜欢有挑战性的内容
}

# 测试场景3：高级学习者，想要完善知识体系并应用于实践
advanced_profile = {
    "mastered_topics": {
        "数学基础", "编程基础", "Python基础", "线性代数", "微积分", "概率与统计",
        "NumPy", "Pandas", "Matplotlib", "数据预处理", "特征工程", "数据可视化",
        "监督学习", "无监督学习", "线性回归", "逻辑回归", "决策树", "随机森林",
        "模型评估", "交叉验证", "正则化", "过拟合与欠拟合", "梯度下降"
    },
    "learning_goal": {"机器学习项目", "部署与生产", "深度学习"},
    "interests": {
        "应用实践": 0.9,
        "部署": 0.8,
        "深度学习": 0.7
    },
    "learning_style": "kinesthetic",
    "available_time": 60,  # 总共可用60小时
    "difficulty_preference": "easy"  # 希望相对简单的内容
}

# 7. 使用AdaptiveKnowledgeGraphACO算法运行测试

def test_algorithm(profile_name, learner_profile):
    print(f"\n===== 测试场景: {profile_name} =====")
    print(f"已掌握的知识点: {len(learner_profile['mastered_topics'])}")
    print(f"学习目标: {learner_profile['learning_goal']}")
    print(f"可用学习时间: {learner_profile['available_time']}小时")
    
    # 初始化算法
    aco = AdaptiveKnowledgeGraphACO(
        knowledge_graph=ml_knowledge_graph,
        importance=importance,
        difficulty=difficulty,
        learning_time=learning_time,
        dependencies=dependencies,
        learner_profile=learner_profile,
        n_ants=30,
        alpha=1.0,
        beta=2.0,
        gamma=1.0,
        delta=0.7,
        rho=0.1,
        q0=0.6,
        n_iterations=50,
        time_constraint=learner_profile["available_time"]
    )
    
    # 运行算法
    _, best_score = aco.optimize()
    
    # 获取最优学习路径
    optimal_path = aco.get_optimal_learning_path()
    print("\n最优学习路径:")
    for i, topic in enumerate(optimal_path):
        print(f"{i+1}. {topic}")
    
    # 获取学习计划详情
    learning_plan = aco.get_learning_plan_details()
    
    # 打印学习计划
    print("\n详细学习计划:")
    total_time = 0
    for step in learning_plan:
        total_time += step["estimated_time"]
        print(f"步骤 {step['step']}: {step['topic']} - 难度: {step['difficulty']:.1f}, 重要性: {step['importance']:.1f}, 学习时间: {step['estimated_time']}小时")
    
    print(f"\n总学习时间: {total_time}小时")
    
    # 假设学习者已经学习了部分推荐的知识点，预测下一步学习内容
    if len(optimal_path) > 3:
        learned_topics = set(optimal_path[:3])  # 假设学习了前3个推荐的知识点
        print(f"\n假设学习者已经学习了: {learned_topics}")
        
        # 预测下一步学习内容
        next_topics = aco.predict_next_topics(learned_topics, n=3)
        
        print("预测的下一步学习内容:")
        for i, topic in enumerate(next_topics):
            print(f"{i+1}. {topic}")
    
    # 可视化学习路径
    print("\n生成学习路径可视化...")
    aco.visualize_learning_path()
    
    return aco

# 8. 运行测试
beginner_aco = test_algorithm("初学者", beginner_profile)
intermediate_aco = test_algorithm("中级学习者", intermediate_profile)
advanced_aco = test_algorithm("高级学习者", advanced_profile)

# 9. 额外测试：更新学习者状态后的路径预测
print("\n===== 额外测试: 学习进度更新后的路径调整 =====")

# 假设初学者完成了部分学习路径
beginner_learned_path = beginner_aco.get_optimal_learning_path()[:5]  # 学习了前5个知识点
print(f"初学者已学习的知识点: {beginner_learned_path}")

# 更新学习者画像
updated_beginner_profile = beginner_profile.copy()
updated_beginner_profile["mastered_topics"] = updated_beginner_profile["mastered_topics"].union(set(beginner_learned_path))
updated_beginner_profile["available_time"] -= sum(learning_time.get(topic, 0) for topic in beginner_learned_path)

print(f"更新后的可用时间: {updated_beginner_profile['available_time']}小时")

# 运行更新后的推荐
updated_beginner_aco = test_algorithm("更新后的初学者", updated_beginner_profile)

# 10. 知识点聚类分析
print("\n===== 知识点聚类分析 =====")

# 使用前面的任何一个ACO实例来获取聚类信息
clusters = beginner_aco.topic_clusters
cluster_groups = {}

for topic, cluster_id in clusters.items():
    if cluster_id not in cluster_groups:
        cluster_groups[cluster_id] = []
    cluster_groups[cluster_id].append(topic)

# 打印每个聚类的知识点
for cluster_id, topics in cluster_groups.items():
    print(f"\n聚类 {cluster_id}:")
    for topic in topics:
        print(f"- {topic} (重要性: {importance.get(topic, 0):.1f}, 难度: {difficulty.get(topic, 0):.1f})")
