# 建筑/障碍物实体类
class Structure:
    def __init__(self, structure_id, structure_type, position, dimensions=(1, 1)):
        """
        初始化建筑
        :param structure_id: 建筑唯一标识
        :param structure_type: 建筑类型（房屋、碉堡、桥梁等）
        :param position: 左上角位置 (x, y)
        :param dimensions: 建筑尺寸 (宽度, 高度)
        """
        self.id = structure_id
        self.type = structure_type
        self.position = position
        self.dimensions = dimensions
        self.health = self._get_max_health()
        self.status = "intact"  # intact, damaged, destroyed
        
    def _get_max_health(self):
        """根据建筑类型获取最大耐久度"""
        if self.type == "house":
            return 100
        elif self.type == "bunker":
            return 300
        elif self.type == "bridge":
            return 200
        return 150
    
    def get_occupancy(self):
        """获取建筑占用的网格坐标"""
        x, y = self.position
        w, h = self.dimensions
        return [(x + dx, y + dy) for dx in range(w) for dy in range(h)]
    
    def take_damage(self, damage):
        """承受伤害"""
        self.health -= damage
        if self.health <= 0:
            self.status = "destroyed"
        elif self.health < self._get_max_health() * 0.5:
            self.status = "damaged"
    
    def is_destroyed(self):
        """检查建筑是否被摧毁"""
        return self.status == "destroyed"
    
    def provides_cover(self):
        """是否提供掩体保护"""
        return self.status != "destroyed" and self.type in ["bunker", "house"]
