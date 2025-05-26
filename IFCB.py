import random
import numpy as np
import pickle
import torch
"""
    该文件作用是对脑电波数据进行加密
    
    首先接受一个脑电波数据特征,将其转为二进制文件形式;
    
    二进制通过密钥随机置换(将0101010101的位置按照密钥随机打乱),然后根据规定的组大小将二进制转为十进制
    (比如16位二进制 0000111100001111 分为4组即 0000 1111 0000 1111,十进制表示为 0 15 0 15)
    
    对这个十进制文件应用反转融合操作,操作的阈值一般是组大小的一半(4个组,组大小为4,则阈值为 2 ** (4-1) - 1 = 7,超过7就反转)
    
    最后对反转的文件进行秩值的计算,用秩值代替原来的反转文件
"""


# 根据个数Key_n和长度Key_length生成密钥
def generate_Keys(Key_nums, Key_length):
    """
    生成 Key_nums 个长度为 Key_length 的随机排列密钥矩阵

    其中密钥长度是和二进制文件长度一致的,密钥本质上是1~n的随机数列,n是二进制文件大小

    参数:
        Key_n: 密钥数量
        Key_length: 每个密钥的长度
    返回:
        binary_array: 2048位的二进制数组
    """
    key = []
    for _ in range(Key_nums):
        # 生成 0 到 m-1 的随机排列
        perm = list(range(Key_length))

        random.shuffle(perm)
        key.append(perm)
    return key


# 将特征值数据转为二进制形式
def feature_to_bits(feature):
    """
    处理脑电波数据特征并生成对应二进制文件

    参数:
        feature: 输入的脑电波数据特征
    返回:
        binary_array: 2048位的二进制数组
    """
    original_shape = feature.shape
    original_dtype = feature.dtype


    cpu_tensor = feature.cpu().numpy()

    binary_data = cpu_tensor.tobytes()
    byte_array = np.frombuffer(binary_data, dtype=np.uint8)
    binary_array = np.unpackbits(byte_array)

    binary_str = ''.join(map(str, binary_array))

    return binary_str, original_shape, original_dtype


def bits_to_feature(binary_str, original_shape = torch.Size([1, 64]), original_dtype = torch.float32):
    """
    将二进制字符串还原为原始特征数据

    参数:
        binary_str: 二进制字符串
        original_shape: 原始形状
        original_dtype: 原始数据类型

    返回:
        reconstructed_feature: 重建的PyTorch Tensor
    """
    # 转换PyTorch dtype为NumPy dtype
    dtype_map = {
        torch.float32: np.float32,
        torch.float64: np.float64,
        torch.int16: np.int16,
        torch.int32: np.int32,
        # 添加其他需要的类型映射
    }

    np_dtype = dtype_map.get(original_dtype, np.float32)  # 默认使用float32

    # 计算预期位数
    expected_bits = np.prod(original_shape) * np.dtype(np_dtype).itemsize * 8

    # 将二进制字符串转为numpy数组
    binary_array = np.array([int(bit) for bit in binary_str], dtype=np.uint8)

    # 截断或填充至预期长度
    if len(binary_array) > expected_bits:
        binary_array = binary_array[:expected_bits]
    elif len(binary_array) < expected_bits:
        binary_array = np.pad(binary_array, (0, expected_bits - len(binary_array)))

    # 重建数据
    byte_array = np.packbits(binary_array)
    reconstructed_data = np.frombuffer(
        byte_array.tobytes(),
        dtype=np_dtype
    ).reshape(original_shape)

    return torch.from_numpy(reconstructed_data).type(original_dtype)


# 将特征根据密钥进行随机置换
def permute_feature(feature, key):
    """
    对脑电波特征数据根据密钥进行随机置换

    参数:
        feature: 脑电波特征数据
        key: 密钥

    返回:
        permuted_bits: 置换后的二进制串
    """
    binary_array = feature_to_bits(feature)  # 转为二进制文件
    permuted_features = []

    for k in range(len(key)):
        current_permutation = binary_array[np.array(key[k])]
        permuted_features.append(current_permutation)

    return permuted_features


# 反转融合函数,将密钥转换的二进制文件按块组转为十进制,并对十进制应用反转融合操作
def inverse_fusion(permuted_features, block_size=8, threshold=None):
    """
    反转融合函数：对置换后的特征进行分块反转融合操作

    参数:
        permuted_features: 密钥置换后的特征列表，每个元素是一个二进制字符串
        block_size: 块大小（默认8位）
        threshold: 反转阈值，默认值为 2^(block_size-1)-1
    返回:
        list: 经过反转融合处理后的十进制值列表
    """

    protected_templates = []
    n = len(permuted_features[0])  # 脑电波特征二进制编码长度
    block_num = n // block_size  # 块数

    if threshold is None:
        threshold = 2 ** (block_size - 1) - 1  # 默认阈值：2^{b-1}-1,即127

    for feature in permuted_features:
        decimal_values = []

        # 分块处理
        for i in range(block_num):
            start = i * block_size
            end = start + block_size
            block = feature[start:end]  # 获取当前块

            # 将二进制数值转十进制
            block_str = ''.join(map(str, block))
            dec_value = int(block_str, 2)

            # 反转融合判断
            if dec_value > threshold:
                # 反转操作：用最大值减去当前值
                inverted_value = (2 ** block_size - 1) - dec_value
                decimal_values.append(inverted_value)
            else:
                decimal_values.append(dec_value)

        protected_templates.append(decimal_values)

    return protected_templates


# 计算反转融合后的秩值,这样可以保护原始数据,避免数据泄露
def localRank(protected_templates, group_size=16):
    """
    对每组进行排序并给出秩值
        参数:
        protected_templates: 反转融合后的十进制数组
        group_size: 块大小
    返回:
        list: 排名数组（保护模板）

    """

    protected_ranks = []
    template_length = len(protected_templates[0])  # 反转融合后的十进制数组长度
    group_num = template_length // group_size  # 组数

    for feature in protected_templates:
        r = np.zeros(template_length, dtype=int)

        for i in range(group_num):
            start = i * group_size
            end = start + group_size
            group = feature[start:end]  # 获取当前组

            # 计算组内排序的秩值
            sorted_indices = np.argsort(group)
            ranks = np.zeros_like(sorted_indices)
            for rank, idx in enumerate(sorted_indices):
                ranks[idx] = rank

            r[start:end] = ranks  # 存入结果

        protected_ranks.append(r)

    return protected_ranks

def sum_IFCB(feature,keys):

    permuted_features =  permute_feature(feature,keys)
    protected_templates = inverse_fusion(permuted_features)
    protected_ranks = localRank(protected_templates)


    return protected_ranks

def serialize_feature(feature: np.ndarray) -> bytes:
    """将NumPy数组序列化为二进制"""

    try:
        return pickle.dumps(feature, protocol=pickle.HIGHEST_PROTOCOL)
    except pickle.PicklingError as e:
        raise ValueError("Failed to serialize array") from e

def deserialize_feature(binary_data: bytes) -> np.ndarray:
    """从二进制反序列化为NumPy数组"""

    try:
        return pickle.loads(binary_data)
    except pickle.UnpicklingError as e:
        raise ValueError("Failed to deserialize data") from e


if __name__ == "__main__" :
    print('0')