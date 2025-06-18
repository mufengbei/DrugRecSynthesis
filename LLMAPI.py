import pickle
import os
import json
import csv
import random
import dashscope

class LLMAPI:
    """
    大语言模型API类，提供病人症状获取和数据错误检查功能
    """
    
    def __init__(self, api_key, model="qwen-max"):
        """
        初始化LLMAPI类
        
        Args:
            api_key (str): API密钥
            model (str): 模型名称，默认为qwen-max
        """
        self.api_key = api_key
        self.model = model
        
    def _get_patient_prompt(self, item, spliter=" || "):
        """
        获取病人症状的prompt模版
        
        Args:
            item (dict): 病人数据项
            spliter (str): 分隔符
            
        Returns:
            tuple: (prompt字符串, 输入信息)
        """
        def format_input(item, spliter=" || "):
            age = item["age"]
            gender = item["gender"]
            group = ",".join(item["group"])
            diagnosis = ",".join(item["diagnosis"])
            return f"{age}{spliter}{gender}{spliter}{group}{spliter}XX{spliter}{diagnosis}"

        input_msg = format_input(item, spliter)

        SYMPTOM_PROMPT = """请根据病人信息和诊断给出合理的主诉症状,按照该格式输出:年龄 || 性别 || 人群类别 || XX || 诊断。其中年龄、性别、人群类别、诊断由我给出，你只需在XX处填入主诉症状信息。
            请注意,你只需在XX处生成主诉症状信息,不要输出任何别的信息或者提示,不要修改我已经给出的其他信息.下面给出8个例子:

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
            input:
        """

        prompt = SYMPTOM_PROMPT + input_msg

        return prompt, input_msg

    def _get_error_check_prompt(self, item, spliter=" || "):
        """
        获取错误检查的prompt模板
        
        Args:
            item (dict): 数据项
            spliter (str): 分隔符
            
        Returns:
            tuple: (prompt字符串, 输入信息)
        """
        def format_input_data(item, spliter=" || "):
            """
            格式化输入数据
            
            Args:
                item (dict): 数据项
                spliter (str): 分隔符
            
            Returns:
                str: 格式化后的输入字符串
            """
            age = item["age"]
            group = ",".join(item["group"])
            gender = item["gender"]
            symptom = ",".join(item["symptom"])
            diagnosis = ",".join(item["diagnosis"])
            antecedents = ",".join(item["antecedents"])
            
            if not item["antecedents"]:
                return f"{age}{spliter}{group}{spliter}{gender}{spliter}{symptom}{spliter}{diagnosis}{spliter}无既往病史"
            else:
                return f"{age}{spliter}{group}{spliter}{gender}{spliter}{symptom}{spliter}{diagnosis}{spliter}{antecedents}"
        
        input_msg = format_input_data(item)

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
        
        prompt = ERRORCHECK_PROMPT + input_msg
        return prompt, input_msg

    def _call_llm_api(self, prompt):
        """
        调用大语言模型API
        
        Args:
            prompt (str): 输入的prompt
            
        Returns:
            str: 模型返回的结果
        """
        messages = [{"role": "user", "content": prompt}]
        
        try:
            response = dashscope.Generation.call(
                api_key=self.api_key,
                model=self.model,
                messages=messages,
                temperature=0,
                top_p=0.01,
                result_format='message',
            )
            
            output = response.output.choices[0].message.content
            return output
            
        except Exception as e:
            print(f"错误信息：{e}")
            print("请参考文档：https://help.aliyun.com/zh/model-studio/developer-reference/error-code")
            return f"ERROR: {str(e)}"

    def get_patient_symptom(self, item):
        """
        获取病人症状
        
        Args:
            item (dict): 病人数据项，包含age、gender、group、diagnosis等字段
            
        Returns:
            str: 提取出的症状信息
        """
        prompt, input_msg = self._get_patient_prompt(item)
        output = self._call_llm_api(prompt)
        
        print(f"id: {item.get('id', 'N/A')}, output: {output}")
        
        # 提取症状部分
        symptom = self.extract_symptom_from_output(output)
        result = {
            'id': item.get('id', 'N/A'),
            'input': input_msg,
            'output': output,
            'symptom': symptom
        }
        return result, symptom

    def check_data_error(self, item):
        """
        检查数据错误
        
        Args:
            item (dict): 数据项，包含age、group、gender、symptom、diagnosis、antecedents等字段
            
        Returns:
            dict: 包含检查结果的字典，格式为{'id': item_id, 'input': input_msg, 'output': output}
        """
        prompt, input_msg = self._get_error_check_prompt(item)
        output = self._call_llm_api(prompt)
        error_code = output.split(":")[1].strip().split("\n")[0]
        print(f"id: {item.get('id', 'N/A')}, output: {output}")
        
        result = {
            'id': item.get('id', 'N/A'),
            'input': input_msg,
            'output': output
        }
        
        return result, error_code

    def batch_check_errors(self, datas):
        """
        批量检查数据错误
        
        Args:
            datas (list): 数据列表
        
        Returns:
            list: 检查结果列表
        """
        checked = []
        
        for item in datas:
            result = self.check_data_error(item)
            checked.append(result)
        
        return checked

    @staticmethod
    def extract_symptom_from_output(data_string, spliter=" || "):
        """
        从格式化的数据字符串中提取症状部分
        
        Args:
            data_string (str): 格式化的数据字符串，如 "29岁 || 女 || 成人 || 尿频、尿急、尿痛 || 尿路感染"
            spliter (str): 分隔符，默认为 " || "
        
        Returns:
            str: 症状部分的内容，如果格式不正确则返回None
        """
        try:
            parts = data_string.split(spliter)
            if len(parts) >= 4:
                return parts[3]  # 第4个元素（索引为3）就是症状部分
            else:
                print(f"数据格式不正确，期望至少4个部分，实际得到{len(parts)}个部分")
                return None
        except Exception as e:
            print(f"提取症状部分时出错: {e}")
            return None
    



def load_data(data_path, file_names):
    """
    加载数据文件
    
    Args:
        data_path (str): 数据文件所在路径
        file_names (list): 数据文件名列表
    
    Returns:
        list: 合并后的数据列表
    """
    datas = []
    for file in file_names:
        with open(os.path.join(data_path, file), "rb") as f:
            data = pickle.load(f)
            datas.extend(data)
    return datas


def save_results(results, output_path):
    """
    保存检查结果到JSON文件
    
    Args:
        results (list): 检查结果列表
        output_path (str): 输出文件路径
    """
    with open(output_path, "w", encoding='utf-8') as fp:
        fp.write(json.dumps(results, ensure_ascii=False, indent=4))


def main(data_path="gnn-data/improvement_66_listtodict", 
         files=["dev.pkl"], 
         output_file="check_out_result_dev.json",
         api_key="sk-d1255a437700465a8709fd302d31834b",
         model="qwen-max"):
    """
    主函数
    
    Args:
        data_path (str): 数据文件路径
        files (list): 数据文件名列表
        output_file (str): 输出文件名
        api_key (str): API密钥
        model (str): 模型名称
    """
    # 初始化LLMAPI
    llm_api = LLMAPI(api_key, model)
    
    # 加载数据
    datas = load_data(data_path, files)
    
    # 批量检查错误
    checked_results = llm_api.batch_check_errors(datas)
    
    # 保存结果
    output_path = os.path.join(data_path, output_file)
    save_results(checked_results, output_path)
    
    print(f"检查完成，结果已保存到: {output_path}")


# 保留原有的函数作为向后兼容
def get_patient_prompt(item, spliter=" || "):
    """向后兼容的函数"""
    llm_api = LLMAPI("")  # 创建临时实例
    return llm_api._get_patient_prompt(item, spliter)

def get_error_check_prompt(item, spliter=" || "):
    """向后兼容的函数"""
    llm_api = LLMAPI("")  # 创建临时实例
    return llm_api._get_error_check_prompt(item, spliter)

def check_error_single_item(item, api_key, model="qwen-max"):
    """向后兼容的函数"""
    llm_api = LLMAPI(api_key, model)
    return llm_api.check_data_error(item)

def batch_check_errors(datas, api_key, model="qwen-max"):
    """向后兼容的函数"""
    llm_api = LLMAPI(api_key, model)
    return llm_api.batch_check_errors(datas)

def get_symptom_single_item(item, api_key, model="qwen-max"):
    """向后兼容的函数"""
    llm_api = LLMAPI(api_key, model)
    return llm_api.get_patient_symptom(item)

def extract_xx_part(data_string, spliter=" || "):
    """向后兼容的函数"""
    return LLMAPI.extract_symptom_from_output(data_string, spliter)


if __name__ == "__main__":
    # 配置参数
    path = "gnn-data/improvement_66_listtodict"
    files = ["dev.pkl"]
    output_file_json = "check_out_result_dev.json"
    api_key = "sk-d1255a437700465a8709fd302d31834b"
    
    # 运行主函数
    main(
        data_path=path,
        files=files,
        output_file=output_file_json,
        api_key=api_key
    )
