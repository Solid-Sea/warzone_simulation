# 地形动态更新引擎
import numpy as np

class TerrainEngine:
    def __init__(self, base_terrain):
        """
        初始化地形引擎
        :param base_terrain: 基础地形矩阵 (0=平原, 1=山地, 2=水域)
        """
        self.terrain = base_terrain.copy()
        self.dynamic_features = np.zeros_like(base_terrain)  # 存储动态变化
        
    def add_crater(self, x, y, radius=3):
        """在指定位置添加弹坑效果"""
        # 创建圆形掩模
        y_indices, x_indices = np.ogrid[-y:len(self.terrain)-y, -x:len(self.terrain[0])-x]
        mask = x_indices*x_indices + y_indices*y_indices <= radius*radius
        
        # 更新动态特征（弹坑标记为3）
        self.dynamic_features[mask] = 3
        
    def update_building_damage(self, building_id, damage_level):
        """更新建筑物损伤状态"""
        # TODO: 实现建筑损伤逻辑
        pass
    
    def get_current_terrain(self):
        """获取当前地形状态（基础+动态）"""
        return np.where(self.dynamic_features > 0, self.dynamic_features, self.terrain)
