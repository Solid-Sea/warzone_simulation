# 路径规划模块（A*算法实现）
import numpy as np
from heapq import heappush, heappop

class Pathfinder:
    def __init__(self, terrain_map):
        """
        初始化路径规划器
        :param terrain_map: 地形矩阵（0=平原, 1=山地, 2=水域, 3=弹坑）
        """
        self.map = terrain_map
        # 定义地形通行成本
        self.terrain_costs = {
            0: 1.0,  # 平原
            1: 3.0,  # 山地
            2: float('inf'),  # 水域（不可通行）
            3: 2.5   # 弹坑
        }
    
    def heuristic(self, a, b):
        """曼哈顿距离启发式函数"""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def find_path(self, start, end):
        """使用A*算法查找路径"""
        neighbors = [(0,1), (0,-1), (1,0), (-1,0)]  # 上下左右四个方向
        
        # 初始化数据结构
        open_set = []
        heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, end)}
        
        while open_set:
            _, current = heappop(open_set)
            
            if current == end:
                # 重构路径
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                return path[::-1]  # 反转路径
            
            for dx, dy in neighbors:
                neighbor = (current[0] + dx, current[1] + dy)
                
                # 检查边界
                if (neighbor[0] < 0 or neighbor[0] >= self.map.shape[0] or 
                    neighbor[1] < 0 or neighbor[1] >= self.map.shape[1]):
                    continue
                
                # 获取地形成本
                terrain_type = self.map[neighbor]
                move_cost = self.terrain_costs.get(terrain_type, float('inf'))
                if move_cost == float('inf'):
                    continue  # 不可通行地形
                
                tentative_g = g_score[current] + move_cost
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor, end)
                    heappush(open_set, (f_score[neighbor], neighbor))
        
        return None  # 无路径
