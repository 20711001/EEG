import random
from typing import Tuple
import numpy as np

"""
    此文件实现了负数据库,阈值设定为0.32,输入的特征二进制编码与数据库中的对比数字小于这个数字便认为属于同一个人
"""




"""
注意!!!
关于 PREDICTION 与 R 参数的配置关乎系统识别的准确率
"""
# 系统参数配置
PREDICTION = [0.9, 0.05, 0.05]  # 负数据生成概率分布
R = 45                        # 负数据库规模参数
EEG_SIGNAL_LENGTH = 2048      # 脑电特征数据二进制编码长度
SIMILARITY_THRESHOLD = 0.32   # 相似度阈值
CN = 2048 * R + 0.5


class Ent:
    """负数据库条目结构体"""

    def __init__(self):
        self.p = [0, 0, 0]  # 三个随机位置索引
        self.c = ['0', '0', '0']  # 三个位置的值


class EEGNegativeDatabase:
    def __init__(self):
        self.registered_eeg = {}
        self.R = 8
        self.PREDICTION = PREDICTION
        self.bit_probs = None  # 新增属性存储位级概率


    def create_negative_counts(self, eeg_feature, r=R):
        """
        将脑电数据的2048位二进制编码转换为不可逆的统计形式,即负数据库形式

        :param eeg_feature: 原始脑电信号的二进制字符串
        :param r: 负数据库规模参数（控制生成多少条负数据）

        :return zero_cnt 和 one_cnt,存储着每一位上0或1的计数
        """
        signal_len = len(eeg_feature)  # 特征长度
        num = int(signal_len * r + 0.5)  # 负数据条数

        zero_cnt = [0] * signal_len  # 0的统计
        one_cnt = [0] * signal_len  # 1的统计

        for _ in range(num):
            v = Ent()
            self.generate_random_numbers(v, signal_len)

            rand_pbt = random.random()  # 0~1之间的小数

            if rand_pbt < PREDICTION[0]:
                # 一位不同,两位相同
                u = random.randint(0, 2)
                for i in range(3):
                    v.c[i] = eeg_feature[v.p[i]]
                v.c[u] = '1' if eeg_feature[v.p[u]] == '0' else '0'


            elif rand_pbt < PREDICTION[0] + PREDICTION[1]:
                # 一位相同,两位不同
                u = random.randint(0, 2)
                for i in range(3):
                    v.c[i] = '1' if eeg_feature[v.p[i]] == '0' else '0'
                v.c[u] = eeg_feature[v.p[u]]


            else:
                # 三位都不同
                for i in range(3):
                    v.c[i] = '1' if eeg_feature[v.p[i]] == '0' else '0'

            for i in range(3):
                if v.c[i] == '0':
                    zero_cnt[v.p[i]] += 1
                else:
                    one_cnt[v.p[i]] += 1

        return zero_cnt, one_cnt


    def create_negative_string(self, eeg_feature, r=R):
        """
        将脑电数据的2048位二进制编码转换为不可逆的统计形式,即负数据库形式

        :param eeg_feature: 原始脑电信号的二进制字符串
        :param r: 负数据库规模参数（控制生成多少条负数据）

        :return zero_str 和 one_str,以 "#" 分隔开存储到数据库中的字符串
        """
        signal_len = len(eeg_feature)  # 特征长度
        num = int(signal_len * r + 0.5)  # 负数据条数

        zero_cnt = [0] * signal_len  # 0的统计
        one_cnt = [0] * signal_len  # 1的统计

        for _ in range(num):
            v = Ent()
            self.generate_random_numbers(v, signal_len)

            rand_pbt = random.random()  # 0~1之间的小数

            if rand_pbt < PREDICTION[0]:
                # 一位不同,两位相同
                u = random.randint(0, 2)
                for i in range(3):
                    v.c[i] = eeg_feature[v.p[i]]
                v.c[u] = '1' if eeg_feature[v.p[u]] == '0' else '0'


            elif rand_pbt < PREDICTION[0] + PREDICTION[1]:
                # 一位相同,两位不同
                u = random.randint(0, 2)
                for i in range(3):
                    v.c[i] = '1' if eeg_feature[v.p[i]] == '0' else '0'
                v.c[u] = eeg_feature[v.p[u]]


            else:
                # 三位都不同
                for i in range(3):
                    v.c[i] = '1' if eeg_feature[v.p[i]] == '0' else '0'

            for i in range(3):
                if v.c[i] == '0':
                    zero_cnt[v.p[i]] += 1
                else:
                    one_cnt[v.p[i]] += 1

        return '#'.join(map(str, zero_cnt)), '#'.join(map(str, one_cnt))


    def generate_random_numbers(self, v, length):
        """生成三个不重复的随机位置索引"""
        v.p[0] = random.randint(0, length - 1)

        v.p[1] = random.randint(0, length - 1)
        while v.p[1] == v.p[0]:
            v.p[1] = random.randint(0, length - 1)

        v.p[2] = random.randint(0, length - 1)
        while v.p[2] == v.p[0] or v.p[2] == v.p[1]:
            v.p[2] = random.randint(0, length - 1)

        return 0

    def GetBinaryArray(self, string_zero, string_one):
        """
        从数据库中读取的的字符串string_zero和string_one获取每一位上0或1的数量

        :param string_zero: 格式化的统计0字符串,如"5#8#12#10#7#3#"
        :param string_one: 格式化的统计0字符串,如"5#8#12#10#7#3#"

        :return: 返回列表,0的计数和1的计数
        """
        # 处理0的计数字符串
        zero_parts = string_zero.split('#')[:-1]  # 去掉最后的空字符串
        zero_counts = list(map(int, zero_parts))

        # 处理1的计数字符串
        one_parts = string_one.split('#')[:-1]  # 去掉最后的空字符串
        one_counts = list(map(int, one_parts))

        return zero_counts, one_counts

    # 计算海明距离
    def ndb_hamming(self, query_binary, string_zero, string_one):
        """
        计算查询二进制串与负数据库的海明距离

        :param query_binary: 查询二进制串,一般是提取的脑电波特征
        :param string_zero: 在数据库中存储的统计0的字符串
        :param string_one: 在数据库中存储的统计1的字符串

        :return: distance: 归一化距离值 [0,1]
        """
        # 将数据库的字符串转为统计列表,列表元素均为整数
        zero_counts, one_counts=self.GetBinaryArray(string_zero,string_one)

        query_binary = np.array([int(num) for num in query_binary])

        zero_array = np.array(zero_counts)
        one_array = np.array(one_counts)

        numerator = np.where(query_binary == 0, one_array, zero_array)

        denominator = one_array + zero_array

        distance = np.sum(numerator / denominator) / len(query_binary)

        return distance



    def estimate_original(self, zero_cnt, one_cnt):
        """
        从统计好的 0/1 计数估计原始二进制串的期望值
        参数:
            zero_cnt: 每位上 0 的计数列表（如 [13, 8, 9,...]）
            one_cnt:  每位上 1 的计数列表（如 [11, 16, 15,...]）
        返回:
            bit_probs: 每位为 1 的概率列表
        """
        total_bits = len(zero_cnt)
        bit_probs = []

        for n0, n1 in zip(zero_cnt, one_cnt):
            # 论文公式(2.3)计算P_diff
            P_diff = (sum(p * i for p, i in zip(PREDICTION, range(1, 4))) / 3)

            # 论文公式(2.4)贝叶斯估计
            numerator = (P_diff ** n1) * ((1 - P_diff) ** n0)
            denominator = numerator + (P_diff ** n0) * ((1 - P_diff) ** n1)

            prob_1 = numerator / denominator if denominator != 0 else 0.5
            bit_probs.append(prob_1)

        return bit_probs



def generate_binary_string(length=2048):
    """生成指定位数的随机二进制字符串"""
    return ''.join(random.choice('01') for _ in range(length))


if __name__ == "__main__":
    eeg_system = EEGNegativeDatabase()
    binary_str = generate_binary_string(EEG_SIGNAL_LENGTH)  # 生成测试用2048位二进制串
    binary_str_1 = generate_binary_string(EEG_SIGNAL_LENGTH)


    """
        1.验证生成负数据库是否有效
    """

    # 生成二进制串的负数据库统计字符
    a, b = eeg_system.create_negative_string(binary_str, R)

    # 根据统计字符生成整数数组,记录次数
    c, d = eeg_system.GetBinaryArray(a, b)


    # a是统计0的字符串,b则是统计1的字符串
    print(a,b, sep = '\n')


    print(c,d, sep = '\n')
