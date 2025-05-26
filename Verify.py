import numpy as np
import torch
from DATABASE import load_model
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score
from DATABASE_DCNN import EEGAuthSystem_DCNN, preprocess_data,load_model_DCNN
import torch.nn.functional as F

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
NUM_SAMPLES = 20  # 样本个数
SUBJECT_RANGE = (1, 110)  # S编号范围
RECORD_RANGE = (3, 15)  # R编号范围
ARR_INDEX_RANGE = (0, 240)  # arr_索引范围
DATA_DIR = "16_channels_seg"

if __name__ == "__main__":

    # 初始化系统
    auth_system = EEGAuthSystem_DCNN(
        model_path="models/DCNN_16x80.pth",
    )


    model, device = load_model_DCNN("models/DCNN_16x80.pth",109)

    right = 0
    all_sum = 0
    y_true = []
    y_pred = []
    # 随机验证
    for i in range(NUM_SAMPLES*20):

        try:
            # 生成随机数
            subject_id = np.random.randint(*SUBJECT_RANGE)
            record_id = np.random.randint(*RECORD_RANGE)
            arr_index = np.random.randint(*ARR_INDEX_RANGE)

            if subject_id in {88, 92, 100} and record_id in range(3,15):
                continue

            # 格式化文件名（使用f-string，保持一致性）
            npz_file = f"{DATA_DIR}/S{subject_id:03d}R{record_id:02d}.npz"
            data = np.load(npz_file)

            random_int = np.random.randint(0, 101) # 在 0 到 100 之间随机取一个整数
            eeg_data = data[f"arr_{random_int}"]
            # 3.预测
            pred_class, probs = predict(model, device, eeg_data)

            # print(f"预测类别: {pred_class + 1} (真实类别: {subject_id}),{np.equal(pred_class+1, subject_id)}")


            # 收集真实标签和预测标签
            y_true.append(subject_id)
            y_pred.append(pred_class + 1)


            all_sum += 1
            if pred_class == subject_id - 1:
                right += 1


        except Exception as e:
            print(f"处理第{i}个样本时出错: {str(e)}")
            continue
    # 计算各项指标
    if all_sum > 0:
        # 基础指标
        accuracy = right / all_sum
        print("\n=== 基础评估指标 ===")
        print(f"正确率(Accuracy): {accuracy:.2%}")

        # 混淆矩阵
        print("\n=== 混淆矩阵 ===")
        cm = confusion_matrix(y_true, y_pred)
        print(cm)



        # 计算每个类别的精确率、召回率和F1分数
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')

        print("\n=== 加权平均指标 ===")
        print(f"精确率(Precision): {precision:.2%}")
        print(f"召回率(Recall): {recall:.2%}")
        print(f"F1分数(F1-Score): {f1:.2%}")
    else:
        print("没有有效样本可用于评估")



