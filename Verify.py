import numpy as np
import torch
from DATABASE import EEGAuthSystem
from DATABASE_DCNN import EEGAuthSystem_DCNN, load_model_DCNN, preprocess_data

""" 
    该py文件为DATABASE_DCNN.py文件中主函数的推广,本质上是验证的重复性实验,可以不管
"""


"""
    当前模型能够准确预测已知类别，例如将S005目录下的所有npz数据段正确归类为类别4（类别编号从0开始）。然而，现阶段的挑战在于如何将模型投入实际应用。
    核心问题在于：当输入一个全新的、来自其他受试者的脑电波数据时，模型仍会强制将其归类到现有的5个类别中。这种处理方式不符合预期需求。
    理想情况下，模型应具备类似人脸识别的能力——通过比对数据库中的已有数据，首先判断输入样本是否属于已注册的脑电波数据，再决定是否进行归类。
"""

def predict(model, device, eeg_data):
    input_tensor = preprocess_data(eeg_data).to(device)
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
    return pred_class, probs.cpu().numpy()[0]


# 常量定义（使用大写命名）
NUM_SAMPLES = 20            # 样本个数
SUBJECT_RANGE = (1, 110)    # S编号范围
RECORD_RANGE = (3, 15)      # R编号范围
ARR_INDEX_RANGE = (0, 240)  # arr_索引范围
DATA_DIR = "16_channels_seg"

if __name__ == "__main__":
    # 初始化
    auth_system = EEGAuthSystem_DCNN("models/DCNN_16x80_sec.pth")
    auth_system.db.connect()


    # # 注册用户
    # auth_system.register_user("user_1", "users_data/S001R01.npz")
    # auth_system.register_user("user_2", "users_data/S002R03.npz")
    # auth_system.register_user("user_3", "users_data/S003R09.npz")


    """
        验证ve_cnn_16*80基础模型是否有效
    """
    # # 1. 加载模型
    # model, device = load_model("ve_cnn_16x80.pth")
    #
    #
    # # 2. 准备EEG数据
    # npz_file = "./16_channels_seg/S001R04.npz"
    # data = np.load(npz_file)
    # random_int = np.random.randint(0, 201)
    # eeg_data = data[f"arr_{random_int}"]
    #
    #
    # # 3. 预处理
    # input_tensor = preprocess_data(eeg_data)
    #
    # # 4. 预测
    # pred_class, probs = predict(model, device, input_tensor)
    #
    # # 5. 输出结果
    # print(f"预测类别: {pred_class}")
    # print("各类别概率:")
    # for i, p in enumerate(probs):
    #     print(f"类别 {i}: {p:.4f}")


    """
         验证ve_cnn_16*80_sec基础模型是否有效
         正确率在92%浮动
    """

    # model, device = load_model_DCNN("models/ve_cnn_16*80_sec.pth",109)
    #
    #
    # right = 0
    # all_sum = 0
    #
    # # 随机验证
    # for i in range(NUM_SAMPLES*20):
    #     try:
    #         # 生成随机数
    #         subject_id = np.random.randint(*SUBJECT_RANGE)
    #         record_id = np.random.randint(*RECORD_RANGE)
    #         arr_index = np.random.randint(*ARR_INDEX_RANGE)
    #
    #         # 格式化文件名（使用f-string，保持一致性）
    #         npz_file = f"{DATA_DIR}/S{subject_id:03d}R{record_id:02d}.npz"
    #         print(npz_file)
    #         data = np.load(npz_file)
    #         random_int = np.random.randint(0, 101) # 在 0 到 100 之间随机取一个整数
    #         eeg_data = data[f"arr_{random_int}"]
    #
    #
    #         # 3. 预处理
    #         input_tensor = preprocess_data(eeg_data)
    #
    #         # 4. 预测
    #         pred_class, probs = predict(model, device, eeg_data)
    #
    #         # 5. 输出结果
    #         print(f"预测类别: {pred_class+1}")
    #
    #         all_sum += 1
    #         if pred_class == subject_id - 1:
    #             right += 1
    #
    #     except Exception as e:
    #         print(f"处理第{i}个样本时出错: {str(e)}")
    #         continue
    #
    # ratio = right/all_sum
    # print("正确率为: {:.2f}%".format(ratio * 100))


    # # 随机验证
    # for i in range(NUM_SAMPLES*5):
    #     # 生成随机数
    #     subject_id = np.random.randint(*SUBJECT_RANGE)
    #     record_id = np.random.randint(*RECORD_RANGE)
    #     arr_index = np.random.randint(*ARR_INDEX_RANGE)
    #
    #     # 格式化文件名（使用f-string，保持一致性）
    #     filename = f"{DATA_DIR}/S{subject_id:03d}R{record_id:02d}.npz"
    #
    #     try:
    #         # 加载数据（添加异常处理）
    #         unknown_sample = np.load(filename)[f"arr_{arr_index}"]
    #         result = db.verify(unknown_sample, threshold=0.60)
    #         print(f"样本 {i + 1:02d}: S{subject_id:03d}R{record_id:02d}[{arr_index:03d}] - 验证结果: {result}")
    #     except FileNotFoundError:
    #         print(f"样本 {i + 1:02d}: 文件 {filename} 不存在")
    #     except KeyError:
    #         print(f"样本 {i + 1:02d}: 数组索引 arr_{arr_index} 不存在")
    #     except Exception as e:
    #         print(f"样本 {i + 1:02d}: 发生错误 - {str(e)}")





    """
        验证DCNN_16*80.pth模型的正确率
    """

    model, device = load_model_DCNN("models/DCNN_16x80.pth",109)

    right = 0
    all_sum = 0

    # 随机验证
    for i in range(NUM_SAMPLES*20):
        try:
            # 生成随机数
            subject_id = np.random.randint(*SUBJECT_RANGE)
            record_id = np.random.randint(*RECORD_RANGE)
            arr_index = np.random.randint(*ARR_INDEX_RANGE)

            # 格式化文件名（使用f-string，保持一致性）
            npz_file = f"{DATA_DIR}/S{subject_id:03d}R{record_id:02d}.npz"
            data = np.load(npz_file)

            random_int = np.random.randint(0, 101) # 在 0 到 100 之间随机取一个整数
            eeg_data = data[f"arr_{random_int}"]

            # 3.预测
            pred_class, probs = predict(model, device, eeg_data)
            print(f"预测类别: {pred_class + 1} (真实类别: {subject_id})")

            all_sum += 1
            if pred_class == subject_id - 1:
                right += 1

        except Exception as e:
            print(f"处理第{i}个样本时出错: {str(e)}")
            continue

    ratio = right/all_sum
    print("正确率为: {:.2f}%".format(ratio * 100))