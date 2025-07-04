# 军事单位实体类
class Unit:
    def __init__(self, unit_id, unit_type, position=(0, 0), camp="red"):
        """
        初始化军事单位
        :param unit_id: 单位唯一标识
        :param unit_type: 单位类型（步兵、坦克、火炮、工兵）
        :param position: 初始位置 (x, y)
        :param camp: 所属阵营（红方/蓝方）
        """
        self.id = unit_id
        self.type = unit_type
        self.position = position
        self.camp = camp
        self.health = 100
        self.status = "active"
        self.respawn_count = 5  # 重生次数
        self.movement_speed = self._get_base_speed()
        
        # 根据单位类型设置属性
        self.attack_range = 0
        self.damage = 0
        self.vision_range = 0  # 视野范围
        self.explosive_range = 0  # 爆炸攻击范围
        self._set_unit_attributes()
        
        # 阵营颜色标识
        self.camp_color = (255, 0, 0) if camp == "red" else (0, 0, 255)
        
        # 攻击目标（用于信息板显示）
        self.attack_target = None
    
    def _get_base_speed(self):
        """根据单位类型获取基础移动速度"""
        if self.type == "infantry":
            return 3
        elif self.type == "tank":
            return 5
        elif self.type == "artillery":
            return 2
        elif self.type == "engineer":  # 工兵
            return 4
        return 4
    
    def _set_unit_attributes(self):
        """根据单位类型设置战斗属性"""
        if self.type == "infantry":
            self.attack_range = 2
            self.damage = 10
            self.vision_range = 8
        elif self.type == "tank":
            self.attack_range = 6
            self.damage = 30
            self.explosive_range = 2  # 短程爆炸范围
            self.vision_range = 10
        elif self.type == "artillery":
            self.attack_range = 20
            self.damage = 50
            self.explosive_range = 4  # 炮兵爆炸范围
            self.vision_range = 15
        elif self.type == "engineer":  # 工兵
            self.attack_range = 1
            self.damage = 5
            self.vision_range = 6
    
    def move_to(self, new_position):
        """移动单位到新位置"""
        self.position = new_position
    
    def attack(self, target):
        """攻击目标单位"""
        if self.status != "active":
            return False
            
        # 计算实际伤害（考虑距离衰减等因素）
        actual_damage = self.damage
        target.take_damage(actual_damage)
        return True
    
    def take_damage(self, damage):
        """承受伤害"""
        self.health -= damage
        if self.health <= 0:
            self.status = "destroyed"
            # 重生逻辑
            if self.respawn_count > 0:
                self.respawn_count -= 1
                self.health = 100
                self.status = "active"
                # 重生在安全区域（临时实现）
                self.position = (10, 10) if self.camp == "red" else (70, 50)
        elif self.health < 50:
            self.status = "damaged"
            
    def build_obstacle(self, obstacle_type, position):
        """建造障碍物（沟壕/沙包）"""
        if self.type != "engineer":
            return None
            
        obstacle_types = {
            "trench": {"block_tank": True, "block_infantry": False},
            "sandbag": {"block_tank": False, "block_infantry": True}
        }
        
        if obstacle_type not in obstacle_types:
            return None
            
        return {
            "type": obstacle_type,
            "position": position,
            "health": 100,
            **obstacle_types[obstacle_type]
        }
            
    def explosive_attack(self, target_position):
        """执行爆炸攻击（炮兵和坦克）"""
        if self.type not in ["tank", "artillery"]:
            return False
        if self.explosive_range <= 0:
            return False
            
        # 返回爆炸范围和伤害系数
        return {
            "position": target_position,
            "radius": self.explosive_range,
            "damage_factor": 0.8 if self.type == "tank" else 1.2
        }
    
    def is_active(self):
        """检查单位是否仍可行动"""
        return self.status == "active"
