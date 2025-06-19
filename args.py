import argparse
parser = argparse.ArgumentParser()
# 肝功能不全概率
parser.add_argument("--liver_prob", type=float, default=0.1)
# 肾功能不全概率
parser.add_argument("--kidney_prob", type=float, default=0.1)
# 是否在历史数据的基础上生成 (0: 重新生成, 1: 从历史数据继续生成)
parser.add_argument("--history_data", type=int, default=1)
# 历史数据文件夹
parser.add_argument("--history_doc", type=str, default="DrugRec_20250618_5_4")
# 是否有病史概率
parser.add_argument("--medhistory_prob", type=float, default=0.2)
# 是否有过敏原概率
parser.add_argument("--allergen_prob", type=float, default=0.4)
# 生成人数量
parser.add_argument("--people_num", type=int, default=5)
# 生成数据保存文件夹
parser.add_argument("--out_doc", type=str, default="DrugRec_20250618_5_5")
# 与大模型交互缓存数据
parser.add_argument("--out_LLMcache", type=str, default="output/llm_cache.pkl")
# 读入人群pickle数据路径
parser.add_argument("--read_person", type=str, default='output/patientdata6/people_cache.pkl')
# 过敏原列表提取源csv文件
parser.add_argument("--allergen_filename", type=str, default='data/allergen.csv')
# 症状列表提取源csv文件
parser.add_argument("--diagnosis_filename", type=str, default='data/diagnosis_medicine_dict.json')
# 人群年龄概率文件
parser.add_argument("--geography_file_path", type=str, default='data/agedemo2.csv')
# 病史备选列表文件
parser.add_argument("--medhistory_file_path", type=str, default='data/medical_history.csv')

# 是否生成人的同时，传入大模型生成症状
parser.add_argument("--to_LLM", type=int, default=0)
# 是否考虑疾病的覆盖率
parser.add_argument("--consider_coverage", type=int, default=1)
# 允许同名疾病出现多少次,只在考虑疾病覆盖率时起效
parser.add_argument("--upper_limit", type=int, default=2)

arg = parser.parse_args()

if __name__ == "__main__":
    pass
