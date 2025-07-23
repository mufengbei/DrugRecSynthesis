# DrugRecSynthesis - 数据分割工具
# 用于将pkl文件按指定比例分割为训练集、验证集和测试集

from data_splitter import split_pkl_data


def main():
    """
    主函数：执行数据分割任务
    """
    # 设置文件路径
    input_file = "output/DrugRec_0704/people_data_bm25_top50_retrieval.pkl"
    output_dir = "output/DrugRec_0704/split_data"
    
    print("DrugRecSynthesis - 数据分割工具")
    print("将pkl文件按照6:2:2比例分割为train.pkl, dev.pkl, test.pkl")
    print(f"输入文件: {input_file}")
    print(f"输出目录: {output_dir}")
    
    # 执行数据分割
    try:
        split_pkl_data(
            input_file=input_file,
            output_dir=output_dir,
            ratios=(0.6, 0.2, 0.2),  # 6:2:2 比例
            random_seed=42
        )
    except Exception as e:
        print(f"数据分割过程中出现错误: {e}")
        return False
    
    return True


if __name__ == '__main__':
    success = main()
    if success:
        print("\n程序执行完成！")
    else:
        print("\n程序执行失败！")
