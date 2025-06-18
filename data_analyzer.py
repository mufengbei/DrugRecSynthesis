import pandas as pd
import matplotlib.pyplot as plt
import os
from args import arg
from pylab import mpl
# 配置matplotlib支持中文显示
import matplotlib.font_manager as fm
import matplotlib

# 字体缓存重建（与font_with_chinese.py保持一致）
try:
    # 适用于不同版本的matplotlib
    if hasattr(matplotlib.font_manager, '_rebuild'):
        matplotlib.font_manager._rebuild()
    else:
        # 新版本可能使用这个方法
        matplotlib.font_manager.fontManager.__init__()
    print("字体缓存已重建")
except Exception as e:
    print(f"字体缓存重建失败，继续执行: {e}")

# 查找中文字体（使用与font_with_chinese.py相同的方法）
print("=== 查找系统中的中文字体 ===")
chinese_fonts = []
for font in fm.fontManager.ttflist:
    if 'CJK' in font.name or 'Noto' in font.name or 'WenQuanYi' in font.name or 'SimHei' in font.name:
        chinese_fonts.append(font.name)

# 设置字体
if chinese_fonts:
    chosen_font = chinese_fonts[0]
    print(f"找到并使用中文字体: {chosen_font}")
    plt.rcParams['font.sans-serif'] = [chosen_font, 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
else:
    print("未找到中文字体，使用默认字体")
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

class DataAnalyzer:
    """数据分析类，用于对生成的人群数据进行统计分析"""
    
    def __init__(self, output_dir=None):
        """
        初始化数据分析器
        
        Args:
            output_dir: 输出目录路径，如果为None则使用默认路径
        """
        self.output_dir = output_dir or f"output/{arg.out_doc}"
        
        # 确保输出目录存在
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def group_analysis(self, people_data):
        """
        根据人群分组进行统计分析，包括：
        基础分组：儿童、青少年、成人、老年人
        特殊分组：孕妇、肝功能不全、肾功能不全等
        
        Args:
            people_data: 人群数据列表
            
        Returns:
            dict: 分组统计结果
        """
        # 收集所有分组信息
        all_groups = []
        total_people = len(people_data)
        
        # 定义所有可能的分组类别（按照优先级排序）
        group_categories = [
            '儿童', '青少年', '成人', '老年人', 
            '孕妇', '哺乳期', 
            '肝功能不全', '肾功能不全'
        ]
        
        # 统计每个分组的出现次数
        group_counts = {}
        
        for person in people_data:
            person_groups = person['group'] if isinstance(person['group'], list) else [person['group']]
            
            # 将每个人的分组添加到总列表中
            for group in person_groups:
                if group in group_counts:
                    group_counts[group] += 1
                else:
                    group_counts[group] = 1
                all_groups.append(group)
        
        # 确保所有预定义的分组都在统计中（即使数量为0）
        for category in group_categories:
            if category not in group_counts:
                group_counts[category] = 0
        
        # 按照预定义顺序排序
        ordered_groups = {}
        for category in group_categories:
            if category in group_counts:
                ordered_groups[category] = group_counts[category]
        
        # 添加其他未预定义的分组
        for group, count in group_counts.items():
            if group not in ordered_groups:
                ordered_groups[group] = count
        
        # 计算百分比并输出统计结果
        self._print_group_statistics(ordered_groups, total_people)
        
        # 绘制条形图
        self._plot_group_distribution(ordered_groups)
        
        return ordered_groups
    
    def age_analysis(self, people_data):
        """
        年龄分布统计分析
        按年龄区间统计人群分布情况
        
        Args:
            people_data: 人群数据列表
            
        Returns:
            dict: 年龄分析结果
        """
        # 收集年龄数据
        ages = []
        total_people = len(people_data)
        
        for person in people_data:
            # 支持字典和对象两种格式
            age = person['age'] if isinstance(person, dict) else person.age
            ages.append(age)
        
        # 定义年龄区间（左闭右开）
        age_ranges = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        age_labels = ["0-9岁", "10-19岁", "20-29岁", "30-39岁", "40-49岁", 
                      "50-59岁", "60-69岁", "70-79岁", "80-89岁", "90-99岁"]
        
        # 使用pandas进行年龄区间划分
        age_series = pd.Series(ages)
        age_bins = pd.cut(age_series, bins=age_ranges, right=False, include_lowest=True, labels=age_labels)
        
        # 计算每个年龄区间的人数
        age_distribution = age_bins.value_counts().sort_index()
        
        # 输出统计结果
        self._print_age_statistics(ages, age_labels, age_distribution, total_people)

        # 绘制条形图
        self._plot_age_distribution(age_labels, age_distribution)
        
        return {
            'age_distribution': dict(zip(age_labels, age_distribution)),
            'stats': {
                'total': total_people,
                'min_age': min(ages),
                'max_age': max(ages),
                'avg_age': sum(ages)/len(ages)
            }
        }

    def gender_analysis(self, people_data):
        """
        性别分布统计分析
        统计男性和女性的分布情况
        
        Args:
            people_data: 人群数据列表
            
        Returns:
            dict: 性别分析结果
        """
        # 收集性别数据
        gender_list = []
        total_people = len(people_data)
        
        for person in people_data:
            gender_list.append(person['gender'])
        
        # 计算男女数量
        gender_series = pd.Series(gender_list)
        gender_counts = gender_series.value_counts()
        
        # 输出统计结果
        self._print_gender_statistics(gender_counts, total_people)
        
        # 绘制条形图
        self._plot_gender_distribution(gender_counts)
        
        return {
            'gender_distribution': dict(gender_counts),
            'stats': {
                'total': total_people,
                'male_count': gender_counts.get('男', 0),
                'female_count': gender_counts.get('女', 0)
            }
        }
    
    def _print_group_statistics(self, ordered_groups, total_people):
        """打印人群分组统计信息"""
        print("=" * 50)
        print("生成人群分组统计：")
        print("=" * 50)
        print(f"总人数: {total_people}")
        print("-" * 30)
        
        # 基础年龄分组统计
        print("基础年龄分组：")
        age_groups = ['儿童', '青少年', '成人', '老年人']
        for group in age_groups:
            count = ordered_groups.get(group, 0)
            percentage = (count / total_people) * 100 if total_people > 0 else 0
            print(f"  {group}: {count}人 ({percentage:.2f}%)")
        
        print("-" * 30)
        
        # 特殊人群统计
        print("特殊人群分组：")
        special_groups = ['孕妇', '哺乳期', '肝功能不全', '肾功能不全']
        for group in special_groups:
            count = ordered_groups.get(group, 0)
            percentage = (count / total_people) * 100 if total_people > 0 else 0
            if count > 0:  # 只显示有人数的特殊分组
                print(f"  {group}: {count}人 ({percentage:.2f}%)")
        
        # 其他分组
        other_groups = {k: v for k, v in ordered_groups.items() 
                       if k not in age_groups and k not in special_groups and v > 0}
        if other_groups:
            print("-" * 30)
            print("其他分组：")
            for group, count in other_groups.items():
                percentage = (count / total_people) * 100 if total_people > 0 else 0
                print(f"  {group}: {count}人 ({percentage:.2f}%)")
        
        print("=" * 50)
    
    def _print_age_statistics(self, ages, age_labels, age_distribution, total_people):
        """打印年龄统计信息"""
        # 计算百分比
        age_distribution_percent = (age_distribution / total_people) * 100
        
        print("=" * 50)
        print("年龄分布统计：")
        print("=" * 50)
        print(f"总人数: {total_people}")
        print(f"年龄范围: {min(ages)}岁 - {max(ages)}岁")
        print(f"平均年龄: {sum(ages)/len(ages):.1f}岁")
        print("-" * 30)
        
        print("按年龄区间分组：")
        for label, count, percent in zip(age_labels, age_distribution, age_distribution_percent):
            print(f"  {label}: {count}人 ({percent:.2f}%)")
        print("=" * 50)
    
    def _print_gender_statistics(self, gender_counts, total_people):
        """打印性别统计信息"""
        print("=" * 50)
        print("性别分布统计：")
        print("=" * 50)
        print(f"总人数: {total_people}")
        print("-" * 30)
        
        # 计算每个性别的百分比并输出
        print("按性别分组：")
        for gender in ['男', '女']:
            count = gender_counts.get(gender, 0)
            percentage = (count / total_people) * 100 if total_people > 0 else 0
            print(f"  {gender}性: {count}人 ({percentage:.2f}%)")
        
        print("=" * 50)
    
    def _plot_group_distribution(self, ordered_groups):
        """绘制人群分组分布图"""
        plt.figure(figsize=(14, 8))
        
        # 准备绘图数据（只显示有人数的分组）
        plot_groups = {k: v for k, v in ordered_groups.items() if v > 0}
        
        if plot_groups:
            groups_names = list(plot_groups.keys())
            groups_counts = list(plot_groups.values())
            
            # 创建条形图
            bars = plt.bar(groups_names, groups_counts, 
                          color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', 
                                '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F'][:len(groups_names)])
            
            # 在每个柱子上显示数量
            for bar, count in zip(bars, groups_counts):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + max(groups_counts)*0.01,
                        f'{count}', ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            plt.xlabel('人群分组', fontsize=12, fontweight='bold')
            plt.ylabel('人数', fontsize=12, fontweight='bold')
            plt.title('人群分组分布统计', fontsize=14, fontweight='bold')
            plt.xticks(rotation=45, ha='right')  # 旋转x轴标签以便更好地阅读
            
            # 调整布局避免标签被切断
            plt.tight_layout()
            
            # 添加网格线
            plt.grid(axis='y', alpha=0.3)
            
            # 保存图片
            plt.savefig(f"{self.output_dir}/group_analysis.png", dpi=300, bbox_inches='tight')
            plt.show()
        else:
            print("警告：没有有效的分组数据用于绘图")
    
    def _plot_age_distribution(self, age_labels, age_distribution):
        """绘制年龄分布图"""
        plt.figure(figsize=(14, 8))
        
        # 准备绘图数据（移除0值区间）
        plot_labels = []
        plot_counts = []
        for label, count in zip(age_labels, age_distribution):
            if count > 0:
                plot_labels.append(label)
                plot_counts.append(count)
        
        if plot_counts:
            # 创建条形图 - 使用与group_analysis相同的颜色方案
            bars = plt.bar(plot_labels, plot_counts, 
                          color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', 
                                '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F', 
                                '#FF9F43', '#54A0FF'][:len(plot_labels)])
            
            # 在每个柱子上显示数量
            for bar, count in zip(bars, plot_counts):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + max(plot_counts)*0.01,
                        f'{count}', ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            plt.xlabel('年龄区间', fontsize=12, fontweight='bold')
            plt.ylabel('人数', fontsize=12, fontweight='bold')
            plt.title('年龄分布统计', fontsize=14, fontweight='bold')
            plt.xticks(rotation=45, ha='right')  # 旋转x轴标签以便更好地阅读
            
            # 调整布局避免标签被切断
            plt.tight_layout()
            
            # 添加网格线
            plt.grid(axis='y', alpha=0.3)
            
            # 保存图片
            plt.savefig(f"{self.output_dir}/age_analysis.png", dpi=300, bbox_inches='tight')
            plt.show()
        else:
            print("警告：没有有效的年龄数据用于绘图")
    
    def _plot_gender_distribution(self, gender_counts):
        """绘制性别分布图"""
        plt.figure(figsize=(14, 8))
        
        # 准备绘图数据
        if len(gender_counts) > 0:
            genders = list(gender_counts.index)
            counts = list(gender_counts.values)
            
            # 创建条形图 - 使用相同的颜色方案
            bars = plt.bar(genders, counts, 
                          color=['#FF6B6B', '#4ECDC4'][:len(genders)])
            
            # 在每个柱子上显示数量
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + max(counts)*0.01,
                        f'{count}', ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            plt.xlabel('性别', fontsize=12, fontweight='bold')
            plt.ylabel('人数', fontsize=12, fontweight='bold')
            plt.title('性别分布统计', fontsize=14, fontweight='bold')
            
            # 调整布局
            plt.tight_layout()
            
            # 添加网格线
            plt.grid(axis='y', alpha=0.3)
            
            # 保存图片
            plt.savefig(f"{self.output_dir}/gender_analysis.png", dpi=300, bbox_inches='tight')
            plt.show()
        else:
            print("警告：没有有效的性别数据用于绘图") 