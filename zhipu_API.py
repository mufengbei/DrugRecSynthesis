import zhipuai
import Person
import json
from tqdm import tqdm
from args import arg
from datetime import datetime
from Person import *
import pickle
cache={}
people_finish=[]
def call_zhipu(prompt):
    try:
        client = zhipuai.ZhipuAI(api_key="0b5834605143147b9aadf709c1123315.H79WPTdOaodrD4Ox")
        response = client.chat.completions.create(
            model="glm-4-plus",  # 填写需要调用的模型名称
            messages=[
                {"role": "user", "content": prompt}
            ],
        )
    except zhipuai.core._errors.APIRequestFailedError as error1:
        return "请求错误"
    except zhipuai.core._errors.APITimeoutError as error2:
        return "请求错误"
    # print(response.choices[0].message)
    # print(response.usage)
    # print(type(response.choices[0].message.content))
    return str(response.choices[0].message.content)

def TO_LLM(people_data,LLM_cache):
    num=0

    for i in tqdm(range(arg.people_num)):
        #对于每一个Person对象
        person=people_data[i]
        # 获得prompt
        prompt = get_prompt(person)
        if prompt in LLM_cache:
            LLM_output = LLM_cache[prompt]
        # 将prompt传入LLM
        else:
            LLM_output = call_zhipu(prompt)
            LLM_cache[prompt] = LLM_output
        # 处理LLM输出获得symptom
        symptom = symptom_extract(LLM_output)
        person.symptom=symptom
        people_finish.append(person)
        num+=1
        if num%500==0:
            outto_cache(num, people_data=people_finish, llm_cache=LLM_cache)
    outto_cache(num, people_data=people_finish, llm_cache=LLM_cache)

def TO_LLM2(delete_list,part = "dev"):
    """2025.5.20更新数据集中gold_answers为0的数据使用"""
    with open(f"gnn-data/improvement_519/{part}.pkl","rb") as f1:
        people_data = pickle.load(f1)
    num=0

    for i in tqdm(delete_list):
        #对于每一个Person对象
        p=people_data[i]['people']
        # 获得prompt
        symptom = ','.join(p["symptom"])
        person = Person(p['age'], p['gender'], p['group'], symptom, p['diagnosis'], p['medicine'], p['antecedents'], p['allergen'])
        prompt = get_prompt(person)
        LLM_output = call_zhipu(prompt)
        # 处理LLM输出获得symptom
        symptom = symptom_extract(LLM_output)
        p['symptom']=symptom.split('、')
        print(f"{i}号病人,{p['group']},{p['diagnosis']},{symptom}")
    with open(f"gnn-data/improvement_520_addsymptom/{part}.pkl","wb") as f2:
        pickle.dump(people_data,f2)

def error_check(p):
    ERRORCHECK_PROMPT = """你是一个专业的医生，你的任务是判断病历中是否存在错误。如果存在错误，请根据错误类型编号返回数字；如果没有错误，请返回 0。

        病历的格式为：
        年龄 || 人群 || 性别 || 症状 || 疾病 || 既往病史

        错误类型包括：
        1. 疾病与性别不符；
        2. 疾病与年龄不符；
        3. 疾病与症状不符；
        4. 疾病描述不规范；
        5. 既往病史与性别不符；
        6. 既往病史与年龄不符；
        7. 既往病史描述不规范；
        0. 表示没有错误。

        请严格按照以下格式输出：
        输出：[错误编号](多个错误时用逗号分隔，如:1,5）

        给出以下例子：
        输入: 35 || 孕妇,肾功能不全 || 女 || 动脉粥状硬化 || 胸痛,呼吸困难,水肿 || 无既往病史
        输出: 2

        输入: 80 || 老年人 || 女 || 月经不调,经血颜色改变 || 经色紫暗 || 食少便溏
        输出: 2

        输入: 37 || 孕妇 || 女 || 乳房胀痛,乳头溢液呈豆渣状 || 豆渣状 || 顽痘
        输出: 4

        输入: 47 || 成人 || 男 || 疲劳感,精力不足,性欲减退 || 需求增加 || 黄褐斑
        输出: 4

        输入: 20 || 成人，哺乳期 || 女 || 恶心,呕吐,头痛,乏力 || 抗辐射 || 缺乏症
        输出: 3

        输入: 1 || 儿童,肾功能不全 || 女 || 发育迟缓,肌张力异常 || 机能障碍 || 黄体功能不足
        输出: 6

        输入: 27 || 成人 || 女 || 呼吸急促,胸闷,喉部喘鸣声 || 咳嗽,喘鸣 || 小面积
        输出: 7

        输入: 41 || 成人 || 男 || 避孕咨询,焦虑 || 女性避孕 || 重度持续性哮喘
        输出: 1

        输入: 29 || 成人|| 男 || 头皮瘙痒,鳞屑增多 || 头皮鳞屑 || 盆腔炎
        输出: 5

        输入: 53 || 成人 || 男 || 右上腹痛,乏力,食欲不振 || 肝功能不正常 || 疾病
        输出: 7

        输入: 52 || 成人 || 男 || 胸痛,烧心感 || 吞酸 || 更年期综合征
        输出: 5

        输入: 48 || 成人 || 男 || 乳房发育不良,无乳汁分泌 || 无乳 || 跌打损伤
        输出: 1

        输入: 41 || 成人 || 男 || 胸痛,心悸,气促 || 不稳定型冠状动脉疾病 || 晚期卵巢癌
        输出: 5

        输入: 38 || 成人,孕妇 || 女 || 皮肤瘙痒,光敏感性皮疹,乏力 || 皮肤卟啉病 || 黄褐斑
        输出: 0

        输入: 59 || 成人 || 男 || 阴茎勃起功能障碍,潮热,情绪波动 || 绝经 || 闭经
        输出: 1,5

        输入: """
        


#传入Person对象，形成prompt
def get_prompt(p):
    info=p.get_info()
    prompt=f"""请根据病人信息和诊断给出合理的主诉症状,按照该格式输出:年龄 || 性别 || 人群类别 || XX || 诊断。其中年龄、性别、人群类别、诊断由我给出，你只需在XX处填入主诉症状信息。
        请注意,你只需在XX处生成主诉症状信息,不要输出任何别的信息或者提示,不要修改我已经给出的其他信息.下面给出8个例子:\n
        1.input:35岁 || 男 || 成人 || XX || 呼吸道感染
        output:35岁 || 男 || 成人 || 咳嗽、咳痰、发热 || 呼吸道感染
        2.input:29岁 || 女 || 成人 || XX || 尿路感染
        output:29岁 || 女 || 成人 || 尿频、尿急、尿痛 || 尿路感染
        3.intput:33岁 || 女 || 哺乳期 || XX || 乳腺炎
        output:33岁 || 女 || 哺乳期 || 乳房疼痛、红肿
        4.intput:26岁 || 女 || 孕妇 || XX || 发烧
        output:26岁 || 女 || 孕妇 || 发热、头痛、乏力 || 发烧
        5.intput:42岁 || 男 || 成人 || XX || 前列腺炎
        output:42岁 || 男 || 成人 || 尿频、尿急、尿痛 || 前列腺炎
        6.intput:7岁 || 女 || 儿童 || XX || 消化不良
        output:7岁 || 女 || 儿童 || 食欲不振、腹胀、腹泻 || 消化不良
        7.input:67岁 || 男 || 老年人 || XX || 高血压
        output:67岁 || 男 || 老年人 || 头晕、心悸、胸闷 || 高血压
        8.input:70岁 || 男 || 老年人 || XX || 骨质疏松症
        output:70岁 || 男 || 老年人 || 腰背疼痛、易骨折 || 骨质疏松症
        下面给出病人信息和诊断，请按照格式输出。\n
        input:{info}\n
        output:
        """
    return prompt

def read_cache():
    with open("output/zhipu_llm_cache.pkl","rb") as f:
        llm_cache=pickle.load(f)
    with open(f"output/patient_22000_2/people_cache_final.pkl", "rb") as fp:
        people_list=pickle.load(fp)
    chunk = people_list[11098:]
    return chunk,llm_cache

def outto_cache(num,people_data,llm_cache):
    with open(f"output/{arg.out_doc}/{num}_people_cache.pkl", "wb") as fp:
        pickle.dump(people_data, fp)
    with open(arg.out_LLMcache, "wb") as fp:
        pickle.dump(llm_cache, fp)

def out_to_jsonfile(people_data):
    num=0
    json_dict = {}
    for i, p in enumerate(people_data):
        temp_dict = p.transfer_to_dict()
        num+=temp_dict['medicine_num']
        json_dict[i]=temp_dict
    json_str = json.dumps(json_dict, indent=4, ensure_ascii=False)
    avg = num // arg.people_num
    # 获取当前时间
    current_time = datetime.now().strftime('%Y-%m-%d-%H-%M')

    print(current_time)
    with open(f"output/{arg.out_doc}/{current_time}_generatedata_{avg}.json", "w",encoding="utf-8") as f:
        f.write(json_str)
    with open(f"output/{arg.out_doc}/people_cache_generate.pkl", "wb") as fp:
        pickle.dump(people_data, fp)
    print(f"平均药物数量{avg}\n")
    print(f"数据已导出到output/{arg.out_doc}/{current_time}_{arg.out_doc}_{avg}.json文件中。\n")
    return num


#从LLM的输出里提取出生成的主诉症状信息
def symptom_extract(text):
    # 使用split()方法，以“ || ”作为分隔符，将字符串分割为一个列表
    data_list = text.split(" || ")
    if len(data_list)==5:
        symptom=data_list[3]
        #print(symptom)
    else:
        symptom="输出数据有误"
    # 返回列表
    return symptom

def function1():
    people_list,llm_cache = read_cache()
    TO_LLM(people_list,llm_cache)
    num = out_to_jsonfile(people_list)
    
def out_tosee_update(patientid):
    with open("gnn-data/top50-interaction-109/train.pkl","rb") as f1:
        before = pickle.load(f1)
    with open("gnn-data/top50-liverrenal-1213/train.pkl", "rb") as f2:
        after = pickle.load(f2)
    
    record = {
      "before": before[patientid]['people'],
      "after":after[patientid]['people']
    }
    with open("gnn-data/top50-liverrenal-1213/5153_out_for_check.pkl","wb") as f3:
        pickle.dump(record,f3)

def dict_to_list(part = "dev"):
    with open(f"gnn-data/improvement_0603_medicineaddmsg/{part}.pkl","rb") as f2:
        data = pickle.load(f2)
    patient_list = list()
    for idx,item in data.items():
        patient = item['people']
        patient["id"] = idx
        #if isinstance(patient['medicine'], dict):
        patient['medicine'] = list(patient['medicine'].values())
        patient['on_medicine'] = list(patient['on_medicine'].values())
        patient['conflict'] = list(patient['conflict'].values())
        for drug in patient['medicine']:
            # 本来是int改为string
            drug['drugid'] = str(drug['drugid'])
        patient_list.append(patient)
    with open(f"gnn-data/improvement_63_listtodict/{part}.pkl","wb") as f3:
        pickle.dump(patient_list,f3)

def antecedents_list(part = "dev"):
    with open(f"gnn-data/improvement_63_listtodict/{part}.pkl","rb") as f2:
        data = pickle.load(f2)
    for patient in data:
        if not isinstance(patient['antecedents'], list):
            patient['antecedents'] = [patient['antecedents']]

    with open(f"gnn-data/improvement_66_listtodict/{part}.pkl","wb") as f3:
        pickle.dump(data,f3)
        

def transfer_to_excel():
    # 1. 读取JSON文件
    with open('gnn-data/improvement_66_listtodict/check_out_result_test3.json', 'r', encoding='utf-8') as file:
        data = json.load(file)  # 这会解析整个JSON文件
    # 打开文件（如果不存在会自动创建）
    with open("gnn-data/improvement_66_listtodict/checkout_output_test.txt", "w", encoding="utf-8") as file:
        for item in data:
            # 提取错误代码
            error_code = item["output"].split(":")[1].strip().split("\n")[0]

            # 组合成新格式
            new_format = f"{item['id']} || {item['input']} || {error_code}\n"

            # 写入文件
            file.write(new_format)
    print("数据已成功写入checkout_output_test.txt")

if __name__ == '__main__':
    transfer_to_excel()
    #transfer_to_excel()
    #antecedents_list("train")
    # with open(f"onmedicine_check_topkdrugs_delete0/train.pkl","rb") as f2:
    #     data = pickle.load(f2)
    # print(data['2390']['people'].medhistory)
    # print(data['2390']['people'].on_medicine)
    # patientid_list = ['198', '578', '870', '1069', '1228', '1670', '2463', '2838', '3500', '3568', '4014', '4598', '7192', '7199', '7429', '9268', '9473', '9809', '9924', '10044', '10391', '11808', '12733', '13171']
    # TO_LLM2(patientid_list,"train")
    #out_tosee_update(patientid_list)
    #function1()
    # p = Person(41, "男", "成人", "", "胆道功能性疾患", "", "无", "无")
    # p2 = Person(92, "女", "老年人", "", "瘀热互结症", "", "无", "无")
    # prompt = get_prompt(p)
    # LLM_output=call_zhipu(prompt)
    # symptom = symptom_extract(LLM_output)
    # print(symptom)