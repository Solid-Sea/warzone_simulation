# 地图生成模块
import numpy as np
from scipy.ndimage import gaussian_filter

class MapGenerator:
    def __init__(self, width=100, height=100):
        self.width = width
        self.height = height
    
    def generate_terrain(self, terrain_type="desert"):
        """
        生成基础地形网格 (0=平原, 1=山地, 2=水域)
        使用噪声生成算法创建自然地形特征
        """
        # 创建基本噪声图
        noise = np.random.rand(self.height, self.width)
        
        # 应用高斯模糊使地形更自然
        terrain = gaussian_filter(noise, sigma=3)
        
        # 根据地形类型设置阈值
        if terrain_type == "desert":
            thresholds = (0.4, 0.7)  # 平原阈值, 山地阈值
        elif terrain_type == "forest":
            thresholds = (0.3, 0.6)
        elif terrain_type == "arctic":
            thresholds = (0.35, 0.65)
        else:  # 默认
            thresholds = (0.4, 0.7)
        
        # 将噪声值转换为地形类型
        terrain_classes = np.zeros_like(terrain, dtype=int)
        terrain_classes[terrain < thresholds[0]] = 2  # 水域
        terrain_classes[(terrain >= thresholds[0]) & (terrain < thresholds[1])] = 0  # 平原
        terrain_classes[terrain >= thresholds[1]] = 1  # 山地
        
        # 添加一些随机水域特征
        water_mask = terrain_classes == 2
        expanded_water = gaussian_filter(water_mask.astype(float), sigma=2) > 0.3
        terrain_classes[expanded_water] = 2
        
        return terrain_classes
