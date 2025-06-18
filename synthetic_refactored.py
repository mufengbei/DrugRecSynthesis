from py2neo import Graph
import json
import pickle
from LLMAPI import LLMAPI
from args import arg
import pandas as pd
import os
from tqdm import tqdm
import random
from DrugReview import DrugReviewSystem
from data_analyzer import DataAnalyzer


class Synthetic:
    """人工合成数据生成类"""
    
    def __init__(self, neo4j_uri="http://localhost:7474", 
                 api_key="sk-d1255a437700465a8709fd302d31834b", 
                 model="qwen-max"):
        """
        初始化合成数据生成器
        
        Args:
            neo4j_uri: Neo4j数据库连接URI
            api_key: LLM API密钥
            model: 使用的模型名称
        """
        self.graph = Graph(neo4j_uri)
        self.reviewer = DrugReviewSystem(neo4j_uri)
        self.llm_api = LLMAPI(api_key=api_key, model=model)
        self.llm_cache = {}
        self.data_analyzer = DataAnalyzer()
        
        # 数据文件加载状态
        self._medicine_symptoms_dict = None
        self._age_probabilities = None
        self._allergen_list = None
        self._interaction_dict = None
        self._drugmsg_dict = None
    
    def _load_data_files(self):
        """延迟加载数据文件"""
        if self._medicine_symptoms_dict is None:
            with open(arg.diagnosis_filename, 'r', encoding='utf-8') as f:
                self._medicine_symptoms_dict = json.load(f)
        
        if self._age_probabilities is None:
            self._age_probabilities = pd.read_csv(arg.geography_file_path)
        
        if self._allergen_list is None:
            self._allergen_list = pd.read_csv(arg.allergen_filename)['allergen'].values.tolist()
        
        if self._interaction_dict is None:
            try:
                with open("data/drug_interaction_analysis_dict.pkl", "rb") as f:
                    self._interaction_dict = pickle.load(f)
            except FileNotFoundError:
                print("警告：未找到药物相互作用数据文件")
                self._interaction_dict = {}
        
        if self._drugmsg_dict is None:
            try:
                with open("data/drugMsg_linux_dict.pkl", "rb") as f:
                    self._drugmsg_dict = pickle.load(f)
            except FileNotFoundError:
                print("警告：未找到药物信息数据文件")
                self._drugmsg_dict = {}

    def check_diagnosis_reasonable(self, diagnosis, person):
        """
        根据诊断名称的字面含义，判断是否符合该病人的性别和年龄特征
        
        Args:
            diagnosis: 诊断名称
            person: 病人信息，包含age和gender字段
        
        Returns:
            bool: True表示符合，False表示不符合
        """
        # 定义年龄相关关键词
        age_keywords = {
            'children': ['小儿', '儿童', '婴儿', '新生儿', '幼儿', '婴幼儿', '小孩', '先天性'],
            'adult': ['成人'],
            'elderly': ['老年', '老人', '退行性', '老年性']
        }
        
        # 定义性别相关关键词
        gender_keywords = {
            'female_only': ['妇', '妇女', '女性', '孕妇', '妊娠', '哺乳期', '产妇', '经期', '月经', '更年期', 
                           '子宫', '卵巢', '宫颈', '阴道', '乳腺', '盆腔', '妇科', '产科', '绝经',
                           '宫内膜', '附件', '外阴', '白带', '痛经', '闭经'],
            'male_only': ['前列腺', '男性', '阳痿', '早泄', '遗精', '男科', '睾丸', '附睾', 
                         '精囊', '阴囊', '包皮', '龟头', '尿道', '精索'],
            'pregnant': ['孕妇', '妊娠', '孕期', '胎儿', '安胎', '流产', '早产', '产前', '产后'],
            'lactating': ['哺乳期', '授乳', '催乳', '回奶']
        }
        
        # 年龄相关检查
        if any(keyword in diagnosis for keyword in age_keywords['children']):
            if person['age'] >= 12:
                return False
        
        elif any(keyword in diagnosis for keyword in age_keywords['adult']):
            if person['age'] < 18:
                return False
        
        elif any(keyword in diagnosis for keyword in age_keywords['elderly']):
            if person['age'] < 50:
                return False
        
        # 性别相关检查
        if any(keyword in diagnosis for keyword in gender_keywords['female_only']):
            if person['gender'] != '女' and person['gender'] != 1:
                return False
        
        elif any(keyword in diagnosis for keyword in gender_keywords['male_only']):
            if person['gender'] != '男' and person['gender'] != 0:
                return False
        
        # 特殊人群疾病检查
        if any(keyword in diagnosis for keyword in gender_keywords['pregnant']):
            if (person['gender'] != '女' and person['gender'] != 1) or person['age'] < 15 or person['age'] > 45:
                return False
        
        elif any(keyword in diagnosis for keyword in gender_keywords['lactating']):
            if (person['gender'] != '女' and person['gender'] != 1) or person['age'] < 18 or person['age'] > 40:
                return False
        
        return True

    def get_diagnosis_symptom(self, person, diagnosis_dict):
        """
        获取诊断和症状
        
        Args:
            person: 病人信息
            diagnosis_dict: 诊断字典
            
        Returns:
            tuple: (诊断名称, 症状)
        """
        keys_list = list(diagnosis_dict.keys())
        max_attempts = 100
        attempts = 0
        
        while attempts < max_attempts:
            random_diagnosis = random.choice(keys_list)
            
            if not self.check_diagnosis_reasonable(random_diagnosis, person):
                attempts += 1
                continue
            
            person['diagnosis'] = [random_diagnosis]
            symptom_result, symptom = self.llm_api.get_patient_symptom(person)
            self.llm_cache[symptom_result['input']] = symptom_result['output']
            
            if symptom:
                person['symptom'] = symptom.split('、')
                person['antecedents'] = []
                check_result, error_code = self.llm_api.check_data_error(person)
                self.llm_cache[check_result['input']] = check_result['output']
                print(error_code)
                if error_code == '0':
                    break
            else:
                person['symptom'] = None
            
            attempts += 1
        
        if attempts >= max_attempts:
            print(f"警告：经过{max_attempts}次尝试，未找到完全符合条件的诊断，使用最后一次结果")
        
        return random_diagnosis, symptom

    def get_medicine_and_symptom(self, medicine_symptoms_dict, person):
        """
        为病人获取疾病、症状和对应的药品
        
        Args:
            medicine_symptoms_dict: 疾病-药物字典
            person: 病人信息字典
            
        Returns:
            tuple: (诊断名称, 药物列表)
        """
        max_attempts = 50
        attempts = 0
        
        while attempts < max_attempts:
            try:
                diagnosis, symptom = self.get_diagnosis_symptom(person, medicine_symptoms_dict)
                print(diagnosis, symptom)
                
                medicine_data = medicine_symptoms_dict[diagnosis]
                
                drug_list = []
                for drug in medicine_data:
                    if 'drugid' in drug:
                        drug_list.append(drug['drugid'])
                
                person['medicine'] = drug_list
                
                self.check_medicine_with_KG(person)
                
                if len(person['medicine']) > 0:
                    return diagnosis, person['medicine']
            except Exception as e:
                print(f"获取疾病、症状和对应的药品时出现错误: {e}")
                continue
                
            attempts += 1
        
        print(f"警告：经过{max_attempts}次尝试，未能获取有效的药物和症状")
        person['medicine'] = []
        return "未知疾病", []

    def get_age(self, age_probabilities):
        """
        根据给定年龄区间及概率生成人的年龄
        
        Args:
            age_probabilities: 年龄概率分布数据
            
        Returns:
            int: 生成的年龄
        """
        age_ranges = list(zip(age_probabilities['age_start'], age_probabilities['age_end']))
        probabilities = age_probabilities['probability'].values
        index = random.choices(range(10), weights=probabilities)[0]
        age_range = age_ranges[index]
        age = random.randint(age_range[0], age_range[1])
        return age

    def check_medicine_reasonable(self, medicine_list, person):
        """
        根据药品字面含义，过滤掉不符合该病人性别和年龄的药品
        
        Args:
            medicine_list: 药品列表
            person: 病人信息
        
        Returns:
            list: 过滤后的药品列表
        """
        filtered_medicines = []
        
        # 定义关键词规则
        age_keywords = {
            'children': ['小儿', '儿童', '婴儿', '新生儿', '幼儿', '婴幼儿', '小孩'],
            'adult': ['成人'],
            'elderly': ['老年', '老人']
        }
        
        gender_keywords = {
            'female_only': ['妇', '妇女', '女性', '孕妇', '妊娠', '哺乳期', '授乳', '产妇', '经期', '月经', '更年期'],
            'male_only': ['壮阳', '前列腺', '男性', '阳痿', '早泄', '遗精', '补肾壮阳', '男科'],
            'pregnant': ['孕妇', '妊娠', '孕期', '胎儿', '安胎'],
            'lactating': ['哺乳期', '授乳', '催乳', '回奶']
        }
        
        for medicine in medicine_list:
            medicine_name = medicine.get('drug', '') if isinstance(medicine, dict) else str(medicine)
            should_include = True
            
            # 年龄相关检查
            if any(keyword in medicine_name for keyword in age_keywords['children']):
                if person['age'] >= 12:
                    should_include = False
            
            elif any(keyword in medicine_name for keyword in age_keywords['adult']):
                if person['age'] < 18:
                    should_include = False
            
            elif any(keyword in medicine_name for keyword in age_keywords['elderly']):
                if person['age'] < 65:
                    should_include = False
            
            # 性别相关检查
            if should_include:
                if any(keyword in medicine_name for keyword in gender_keywords['female_only']):
                    if person['gender'] != '女' and person['gender'] != 1:
                        should_include = False
                
                elif any(keyword in medicine_name for keyword in gender_keywords['male_only']):
                    if person['gender'] != '男' and person['gender'] != 0:
                        should_include = False
            
            # 特殊人群检查
            if should_include:
                if any(keyword in medicine_name for keyword in gender_keywords['pregnant']):
                    if (person['gender'] != '女' and person['gender'] != 1) or person['age'] < 15 or person['age'] > 45:
                        should_include = False
                
                elif any(keyword in medicine_name for keyword in gender_keywords['lactating']):
                    if (person['gender'] != '女' and person['gender'] != 1) or person['age'] < 18 or person['age'] > 40:
                        should_include = False
            
            # 额外的药品名称检查
            if should_include:
                if any(keyword in medicine_name for keyword in ['避孕', '紧急避孕']):
                    if (person['gender'] != '女' and person['gender'] != 1) or person['age'] < 15 or person['age'] > 50:
                        should_include = False
                
                elif any(keyword in medicine_name for keyword in ['雌激素', '雌二醇', '黄体酮']):
                    if person['gender'] != '女' and person['gender'] != 1:
                        should_include = False
                
                elif any(keyword in medicine_name for keyword in ['睾酮', '雄激素']):
                    if person['gender'] != '男' and person['gender'] != 0:
                        should_include = False
            
            if should_include:
                filtered_medicines.append(medicine)
        
        return filtered_medicines

    def add_antecedents_and_on_medicine(self, person):
        """
        生成病人的病史antecedents和正在用药on_medicine
        
        Args:
            person: 病人信息字典，必须包含'medicine'字段
        """
        try:
            reviewer = DrugReviewSystem(self.graph.service.uri)
            
            if not self._interaction_dict or not self._drugmsg_dict:
                self._load_data_files()
                
        except Exception as e:
            print(f"错误：初始化数据时出现问题 - {e}")
            person['antecedents'] = []
            person['on_medicine'] = []
            return

        if 'medicine' not in person or not isinstance(person['medicine'], list):
            print("警告：病人药物信息不存在或格式错误")
            person['antecedents'] = []
            person['on_medicine'] = []
            return
            
        gold_answer_list = person['medicine'].copy()
        num_medicines = len(gold_answer_list)
        
        if num_medicines <= 1:
            num_to_select = 0
        elif 2 <= num_medicines <= 4:
            num_to_select = 1
        else:
            num_to_select = random.randint(1, min(3, num_medicines))
        
        antecedents = []
        on_medicine = []
        
        if num_to_select > 0:
            try:
                selected_medicine_ids = random.sample(gold_answer_list, num_to_select)
                
                for selected_medicine_id in selected_medicine_ids:
                    interact_drugs = self._interaction_dict.get(str(selected_medicine_id), {}).get("interaction_drug", [])
                    if not interact_drugs:
                        continue
                    
                    selected_interact_drug = random.choice(interact_drugs)
                    interact_drug_id = selected_interact_drug.get("id")
                    if not interact_drug_id:
                        continue
                    
                    drugmsg = self._drugmsg_dict.get(str(interact_drug_id))
                    if not drugmsg:
                        continue
                    
                    treat_list = drugmsg.get('治疗', [])
                    if treat_list:
                        selected_symptom = random.choice(treat_list)
                        antecedents.append(selected_symptom)
                    
                    on_medicine.append(interact_drug_id)
                    
            except (ValueError, KeyError, TypeError) as e:
                print(f"生成病史和伴随用药时出现错误: {e}")
                antecedents = []
                on_medicine = []

        person['antecedents'] = antecedents
        person['on_medicine'] = on_medicine
        
        # 使用LLM验证病史是否合理
        try:
            result, error_code = self.llm_api.check_data_error(person)
            if error_code != 0:
                person['antecedents'] = []
                print("LLM判断病史与病人特征不符，已清空病史")
        except Exception as e:
            print(f"LLM验证过程中出现错误: {e}")
        
        # 使用interaction_check检查并删除与on_medicine有相互作用的药物
        if on_medicine and gold_answer_list:
            try:
                drugs_to_remove = []
                for gold_drug_id in gold_answer_list:
                    for on_drug_id in on_medicine:
                        if reviewer.interaction_check(int(gold_drug_id), int(on_drug_id)):
                            drugs_to_remove.append(gold_drug_id)
                            break
                
                for drug_id in drugs_to_remove:
                    if drug_id in gold_answer_list:
                        gold_answer_list.remove(drug_id)
                        
            except Exception as e:
                print(f"检查药物相互作用时出现错误: {e}")
        
        person['medicine'] = gold_answer_list

    def read_all_msg(self):
        """读取所有缓存的消息"""
        diagnosis_dict = {}
        
        with open(f"output/{arg.out_doc}/people_cache.pkl", "rb") as fp:
            people_list = pickle.load(fp)
        with open("output/zhipu_llm_cache.pkl", "rb") as f:
            llm_cache = pickle.load(f)
        
        for p in people_list:
            if p.diagnosis in diagnosis_dict:
                diagnosis_dict[p.diagnosis] = diagnosis_dict[p.diagnosis] + 1
            else:
                diagnosis_dict[p.diagnosis] = 1
        
        return people_list, llm_cache, diagnosis_dict
    
    def check_medicine_with_KG(self, person):
        """
        通过知识图谱审查药物，删除不合格的药品
        
        Args:
            person: 病人信息，包含medicine, age, group, allergen等字段
        """
        if not person.get('medicine'):
            return
            
        medicines_to_remove = set()
        
        # 年龄审查
        age_pass, age_failed_medicines = self.reviewer.age_review(person['medicine'], person['age'])
        if not age_pass:
            medicines_to_remove.update(age_failed_medicines)
        
        # 特殊人群审查
        if 'group' in person:
            population_pass, population_failed_medicines = self.reviewer.special_population_review(
                person['medicine'], person['group']
            )
            if not population_pass:
                medicines_to_remove.update(population_failed_medicines)
        
        # 过敏原审查
        allergen = person.get('allergen', [])
        if allergen and allergen != '无' and allergen != ['无']:
            allergen_pass, allergen_failed_medicines = self.reviewer.allergy_review(
                person['medicine'], allergen
            )
            if not allergen_pass:
                medicines_to_remove.update(allergen_failed_medicines)
        # 检查能否正确删除
        print(person['medicine'])
        print(medicines_to_remove)
        # 从药物列表中删除不合格的药物
        person['medicine'] = [med for med in person['medicine'] 
                             if med not in medicines_to_remove]
        print(person['medicine'])

    def decide_group(self, person):
        """
        根据年龄确定人群分组
        儿童/Children（<12）; 青少年/Adolescents（>=12, <18）; 成人/Adults (>=18, <65); 老年人/Elderlies (>=65)
        根据概率添加肝功能不全者和肾功能不全者
        
        Args:
            person: 病人信息字典
        """
        age = person['age']
        group_list = []
        
        # 基础年龄分组
        if age < 12:
            group_list.append('儿童')
        elif 12 <= age < 18:
            group_list.append('青少年')
        elif 18 <= age < 65:
            group_list.append('成人')
        else:
            group_list.append('老年人')
        
        # 根据概率添加特殊人群
        if random.random() < arg.liver_prob:
            group_list.append('肝功能不全')
        
        if random.random() < arg.kidney_prob:
            group_list.append('肾功能不全')
        
        person['group'] = group_list

    def decide_gender(self, person):
        """
        根据性别编号确定性别信息
        0为男，1为女
        
        Args:
            person: 病人信息字典
        """
        gender_id = person['gender']
        
        if gender_id == 0:
            person['gender'] = '男'
        elif gender_id == 1:
            person['gender'] = '女'

    def get_medicine_msg(self, medicine_list):
        """
        获取药物信息列表
        
        Args:
            medicine_list: 药物ID列表
            
        Returns:
            list: 药物信息列表
        """
        drug_msg_list = []
        for medicine in medicine_list:
            drug_msg = self.get_drugmsg_from_mkg(medicine)
            drug_msg_list.append(drug_msg)
        return drug_msg_list

    def get_drugmsg_from_mkg(self, drugid):
        """
        从知识图谱获取药物信息
        
        Args:
            drugid: 药物ID
            
        Returns:
            dict: 药物信息记录
        """
        search = self.graph.run(
            """
            MATCH (drug:`药品`)
            WHERE id(drug) = $drugid
            
            WITH drug, drug.name AS name, drug.number AS CMAN
            OPTIONAL MATCH p1=(drug)-[:用药*0..2]->(fact:`知识组`)-[:用药]->(crowd:`人群`),
                            p2=(fact)-[:用药结果]->(useResult:`用药结果级别`)
            WITH drug, name, CMAN, 
                    collect(DISTINCT {crowdid: id(crowd), crowd: crowd.name, useresultid: id(useResult), useresult: useResult.name}) AS crowdInfo
            
            OPTIONAL MATCH p3=(drug)-[:治疗*0..3]->(treatment:`病症`)
            WITH drug, name, CMAN, crowdInfo,
                    collect(DISTINCT {treatid: id(treatment), treat: treatment.name}) AS treatmentInfo
            
            OPTIONAL MATCH p4=(drug)-[:成分*0..3]->(ingre:`药物`)
            WITH drug, name, CMAN, crowdInfo, treatmentInfo,
                    collect(DISTINCT {ingredientId: id(ingre), ingredient: ingre.name}) AS ingredients
            
            OPTIONAL MATCH p5=(drug)-[:相互作用*0..3]->(inter:`药物`)
            WITH drug, name, CMAN, crowdInfo, treatmentInfo, ingredients,
                    collect(DISTINCT {interactionId: id(inter), interaction: inter.name}) AS interactions
            
            RETURN name, CMAN, crowdInfo, treatmentInfo, ingredients, interactions
            """, drugid=drugid)
        result = search.data()[0]
        
        # 处理知识图谱返回的结果
        caution = []
        for item in result["crowdInfo"]:
            temp = {
                "crowd_id": item["crowdid"],
                "crowd": item["crowd"],
                "caution_levelid": item["useresultid"],
                "caution_level": item["useresult"]
            }
            caution.append(temp)
        
        treat = []
        for item in result["treatmentInfo"]:
            temp = {
                "treat_id": item["treatid"],
                "treat": item["treat"]
            }
            treat.append(temp)
        
        ingredients_list = []
        for item in result["ingredients"]:
            temp = {
                "ingredient_id": item['ingredientId'],
                "ingredient": item["ingredient"],
            }
            ingredients_list.append(temp)
        
        interaction_list = []
        for item in result["interactions"]:
            temp = {
                "interaction_id": item['interactionId'],
                "interaction": item["interaction"],
            }
            interaction_list.append(temp)
            
        if result["interactions"][0]["interaction"] is None:
            interaction_list = []
        if result["ingredients"][0]["ingredient"] is None:
            ingredients_list = []
        if result["crowdInfo"][0]["crowd"] is None:
            caution = []
            
        record = {
            "drugid": drugid,
            "name": result["name"],
            "CMAN": result["CMAN"],
            "treat": treat,
            "caution": caution,
            "ingredients": ingredients_list,
            "interaction": interaction_list
        }
        
        return record

    def generate_people_data(self, num):
        """
        生成人群数据
        
        Args:
            num: 要生成的病人数量
            
        Returns:
            list: 生成的病人数据列表
        """
        # 初始化数据结构
        diagnosis_dict = {}
        people_list = []
        
        # 如果需要在历史数据的基础上生成
        if arg.history_data == 1:
            people_list, self.llm_cache, diagnosis_dict = self.read_all_msg()

        # 创建输出目录
        if not os.path.exists(f"output/{arg.out_doc}"):
            os.makedirs(f"output/{arg.out_doc}")
        
        # 加载配置数据
        self._load_data_files()

        # 生成病人数据
        print(f"开始生成 {num} 份病人信息\n")
        now_num = 0
        pbar = tqdm(total=num)
        
        while now_num < num:
            person = {}
            person['id'] = now_num
            try:
                # 1. 生成基础信息：年龄和性别
                person['age'] = self.get_age(self._age_probabilities)
                person['gender'] = random.randint(0, 1)
                # 2. 转换性别编号为文字（0->男，1->女）
                self.decide_gender(person)
                
                # 3. 确定人群分组（基于年龄和特殊情况）- 必须在获取药物之前
                self.decide_group(person)
                
                # 4. 初始化过敏原（在药物检查之前需要）
                if random.random() < arg.allergen_prob:
                    person['allergen'] = [random.choice(self._allergen_list)]
                else:
                    person['allergen'] = []
                
                # 4. 获得疾病、症状、药品（现在可以安全调用check_medicine_with_KG）
                diagnosis, medicine = self.get_medicine_and_symptom(self._medicine_symptoms_dict, person)
                
                # 5. 初始化病史和伴随用药为空列表
                person['antecedents'] = []
                person['on_medicine'] = []
                
                # 6. 根据概率判断是否添加特殊病史和伴随用药
                if random.random() < arg.medhistory_prob:
                    self.add_antecedents_and_on_medicine(person)
                
                
                
                # 8. 将药品ID转换为详细药品信息
                person['medicine'] = self.get_medicine_msg(person['medicine'])
                person['on_medicine'] = self.get_medicine_msg(person['on_medicine'])
                
                # 9. 检查疾病覆盖率限制
                if diagnosis in diagnosis_dict:
                    if arg.consider_coverage == 1 and diagnosis_dict[diagnosis] >= arg.upper_limit:
                        continue
                    diagnosis_dict[diagnosis] += 1
                else:
                    diagnosis_dict[diagnosis] = 1
                
                # 10. 添加到结果列表
                people_list.append(person)
                now_num += 1
                pbar.update(1)
                
            except Exception as e:
                print(f"生成第 {now_num + 1} 个病人时出现错误: {e}")
                continue
        
        pbar.close()
        
        # 生成分析报告
        print("\n正在生成统计分析报告...")
        self.data_analyzer.age_analysis(people_list)
        self.data_analyzer.gender_analysis(people_list)
        self.data_analyzer.group_analysis(people_list)
        
        # 保存数据
        print("\n正在保存数据文件...")
        
        # 保存诊断使用统计
        with open(f"output/{arg.out_doc}/used_diagnosis_dict.json", 'w', encoding='utf-8') as f:
            json.dump(diagnosis_dict, f, ensure_ascii=False, indent=4)
        
        # 保存LLM缓存
        with open(f"output/{arg.out_doc}/LLM_cache.pkl", "wb") as fp:
            pickle.dump(self.llm_cache, fp)
        
        # 保存完整的病人数据列表 - PKL格式
        with open(f"output/{arg.out_doc}/people_data.pkl", "wb") as fp:
            pickle.dump(people_list, fp)
        print(f"✓ 已保存病人数据到: output/{arg.out_doc}/people_data.pkl")
        
        # 保存完整的病人数据列表 - JSON格式  
        try:
            with open(f"output/{arg.out_doc}/people_data.json", 'w', encoding='utf-8') as f:
                json.dump(people_list, f, ensure_ascii=False, indent=2)
            print(f"✓ 已保存病人数据到: output/{arg.out_doc}/people_data.json")
        except (TypeError, ValueError) as e:
            print(f"⚠ JSON保存失败，可能包含不可序列化的对象: {e}")
            print("  建议使用PKL格式读取完整数据")
        
        print(f"\n✅ 成功生成并保存 {len(people_list)} 份病人数据！")
        print(f"数据文件位置:")
        print(f"  - PKL格式: output/{arg.out_doc}/people_data.pkl")
        print(f"  - JSON格式: output/{arg.out_doc}/people_data.json")
        print(f"  - 诊断统计: output/{arg.out_doc}/used_diagnosis_dict.json")
        print(f"  - LLM缓存: output/{arg.out_doc}/LLM_cache.pkl")
        
        return people_list

    def load_people_data(self, file_format="pkl"):
        """
        从文件中加载病人数据
        
        Args:
            file_format: 文件格式，"pkl" 或 "json"
            
        Returns:
            list: 病人数据列表
        """
        if file_format.lower() == "pkl":
            file_path = f"output/{arg.out_doc}/people_data.pkl"
            try:
                with open(file_path, "rb") as fp:
                    people_data = pickle.load(fp)
                print(f"✓ 成功从PKL文件加载 {len(people_data)} 份病人数据")
                return people_data
            except FileNotFoundError:
                print(f"❌ 未找到文件: {file_path}")
                return []
            except Exception as e:
                print(f"❌ 加载PKL文件时出错: {e}")
                return []
                
        elif file_format.lower() == "json":
            file_path = f"output/{arg.out_doc}/people_data.json"
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    people_data = json.load(f)
                print(f"✓ 成功从JSON文件加载 {len(people_data)} 份病人数据")
                return people_data
            except FileNotFoundError:
                print(f"❌ 未找到文件: {file_path}")
                return []
            except Exception as e:
                print(f"❌ 加载JSON文件时出错: {e}")
                return []
        else:
            print(f"❌ 不支持的文件格式: {file_format}，请使用 'pkl' 或 'json'")
            return []

    @staticmethod
    def get_data_summary(people_data):
        """
        获取病人数据的基本统计摘要
        
        Args:
            people_data: 病人数据列表
            
        Returns:
            dict: 统计摘要
        """
        if not people_data:
            return {"error": "数据为空"}
        
        total_count = len(people_data)
        ages = [person['age'] for person in people_data]
        genders = [person['gender'] for person in people_data]
        
        summary = {
            "总人数": total_count,
            "年龄统计": {
                "最小年龄": min(ages),
                "最大年龄": max(ages),
                "平均年龄": round(sum(ages) / len(ages), 1)
            },
            "性别分布": {
                "男性": genders.count('男'),
                "女性": genders.count('女')
            },
            "样本字段": list(people_data[0].keys()) if people_data else []
        }
        
        return summary
    
def load_people_data(file_format="pkl"):
    """
    从文件中加载病人数据
    
    Args:
        file_format: 文件格式，"pkl" 或 "json"
    """
    synthetic = Synthetic()
    people_data = synthetic.load_people_data(file_format)
    print(Synthetic.get_data_summary(people_data))
    return people_data


# 为了保持兼容性，提供一个简单的包装函数
def generate_people_data(num):
    """
    兼容原始接口的包装函数
    
    Args:
        num: 要生成的病人数量
        
    Returns:
        list: 生成的病人数据列表
    """
    synthetic = Synthetic()
    return synthetic.generate_people_data(num)

if __name__ == '__main__':
    people_data = generate_people_data(5)