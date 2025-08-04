import random
import numpy as np
import pymysql
import torch
from numpy.core.defchararray import equal
from pymysql import Error
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr
import torch.nn.functional as F
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from typing import Tuple, Union

from statsmodels.genmod.families.links import logit

from Model_3 import DCNN
from IFCB import feature_to_bits,generate_Keys,inverse_fusion,localRank,permute_feature,sum_IFCB,serialize_feature,deserialize_feature,bits_to_feature
from NegativeDatabase import EEGNegativeDatabase,Ent

# 国密算法
from ENCRYPT import Encrypt_sm2, Encrypt_sm4, encrypt_sm3
import os



"""
    该文件用于DCNN模型与数据库之间的交互,模型为models文件夹下的DCNN_16x80.pth,数据库连接本地数据库
"""

def load_model_DCNN(model_path : str, num_classes : int) -> Tuple[torch.nn.Module, torch.device]:
    """
        加载模型,并使用GPU设备
        :param model_path: 模型地址
        :param num_classes: 模型分类数
        :return: 返回模型和GPU设备
    """

    model = DCNN(num_classes)

    # 安全加载模型权重
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    return model, device



class DatabaseManager:
    """
        主要做数据库方面的操作,一些普适性的操作
    """
    def __init__(self):
        """
        这里是数据库配置参数,更改参数可以连接其他数据库,这里的配置是连接我本人本地的数据库
        """
        # 配置连接本地数据库,后续考虑连接如其他服务器
        self.db_config = {
            'host': 'localhost',
            'user': 'root',
            'password': '123456',
            'database': 'eeg',
            'port': 3306,
            'charset': 'utf8mb4'
        }
        self.connection = None

    def connect(self):
        """
            连接数据库,运行后才能进行后续增删改查操作
        """
        try:
            self.connection = pymysql.connect(
                cursorclass=pymysql.cursors.DictCursor,
                **self.db_config
            )
            print("数据库连接成功")
        except Error as e:
            print(f"数据库连接失败: {e}")
            raise

    def execute_query(self, query_sql, params=None):
        """
        执行查询操作
        接收 "query_sql":查询SQL语句
        返回查询结果
        """
        cursor = None
        try:
            with self.connection.cursor() as cursor:
                if params:
                    cursor.execute(query_sql, params)
                else:
                    cursor.execute(query_sql)
                result = cursor.fetchall()
                return result
        except Error as e:
            print(f"查询数据错误: '{e}'")
            raise
        finally:
            if cursor:
                cursor.close()

    def insert_data(self, table, data):
        """
        执行插入数据操作,主要用于注册用户操作
        "table":插入的表名称
        "data":插入数据,其中插入数据以字典形式保留,如 {'user': 'user_1','pwd':'123456'}

        返回插入操作的id,主要用于表达插入的次数
        """
        if not data:
            raise ValueError("插入数据不能为空")

        # 对data分割操作
        columns = ', '.join(data.keys())
        placeholders = ', '.join(['%s'] * len(data))

        # 正则化表达
        sql = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"

        cursor = None
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(sql, tuple(data.values()))

                self.connection.commit()

                last_id = cursor.lastrowid

                print(f"数据插入成功，ID: {last_id}")
                return last_id
        except Error as e:
            self.connection.rollback()
            print(f"插入数据错误: '{e}'")
            raise
        finally:
            if cursor:
                cursor.close()

    def execute_update(self, update_sql, params=None):
        """
        执行更新操作(INSERT/UPDATE/DELETE)
        接收 "update_sql":更新SQL语句
        返回受影响的行数
        """
        cursor = None
        try:
            with self.connection.cursor() as cursor:
                if params:
                    affected_rows = cursor.execute(update_sql, params)
                else:
                    affected_rows = cursor.execute(update_sql)

                self.connection.commit()
                print(f"操作成功，受影响行数: {affected_rows}")
                return affected_rows
        except Error as e:
            self.connection.rollback()
            print(f"数据操作错误: '{e}'")
            raise
        finally:
            if cursor:
                cursor.close()

    def update_data(self, table, data, condition):
        """
        执行更新数据操作
        接收 "table":插入的表名称, "data":更新数据,其中更新数据以字典形式表示,如 {'user': 'user_1'} 和 "condition":更新条件
        返回插入操作的id,主要用于表达插入的次数
        """
        if not data:
            raise ValueError("更新数据不能为空")
        if not condition:
            raise ValueError("更新条件不能为空")

        set_clause = ', '.join([f"{k} = %s" for k in data.keys()])
        where_clause = ' AND '.join([f"{k} = %s" for k in condition.keys()])
        sql = f"UPDATE {table} SET {set_clause} WHERE {where_clause}"

        params = tuple(data.values()) + tuple(condition.values())

        return self.execute_update(sql, params)

    def delete_data(self, table, condition):
        """
        删除数据(便捷方法)
        :param table: 表名
        :param condition: 删除条件 {列名: 值}
        :return: 受影响的行数
        """
        if not condition:
            raise ValueError("删除条件不能为空")

        where_clause = ' AND '.join([f"{k} = %s" for k in condition.keys()])
        sql = f"DELETE FROM {table} WHERE {where_clause}"

        return self.execute_update(sql, tuple(condition.values()))


class EEGProcessor_DCNN:
    """
        主要是模型与数据之间的操作
    """
    def __init__(self, model_path):
        self.model, self.device = load_model_DCNN(model_path, num_classes=109)
        self.templates = {}  # 格式: {user_id: [feature_vec1, feature_vec2,...]}

    def extract_features(self, npz_path):
        """
        从NPZ文件中随机提取一个EEG片段的特征,返回的是PyTorch CUDA张量
        """
        data = np.load(npz_path)

        # 获取所有有效的EEG片段key（形状为16x80）
        available_keys = [k for k in data.files if data[k].shape == (16, 80)]
        if not available_keys:
            raise ValueError(f"NPZ文件中没有形状为(16, 80)的EEG数据: {npz_path}")

        # 随机选择一个片段
        selected_key = random.choice(available_keys)
        eeg = data[selected_key]  # 形状 (16, 80)

        eeg_tensor = torch.from_numpy(eeg.astype(np.float32)).unsqueeze(0).to(self.device)  # (1,16,80)

        try:
            # 提取特征
            with torch.no_grad():
                features = self.model.extract_features(eeg_tensor)  # 形状 (1, 64)

            return features

        except Exception as e:
            print(f"处理片段 {selected_key} 时出错: {str(e)}")
            raise


class EEGAuthSystem_DCNN:
    """
        将前面两个类:数据库和模型数据处理放一起
    """
    # 初始化函数
    def __init__(self, model_path, sm2_public_key, sm2_private_key):
        self.processor = EEGProcessor_DCNN(model_path)
        self.db = DatabaseManager()
        self.neg_db = EEGNegativeDatabase()


        # 初始化加密模块
        self.sm2 = None
        if sm2_public_key and sm2_private_key:
            self.sm2 = Encrypt_sm2(private_key=sm2_private_key, public_key=sm2_public_key)


    # 实际使用的注册方法
    def register_user_negdb(self, user_id, npz_path):
        """应用了负数据的用户注册流程(再次应用则更新数据)"""
        try:
            # 提取特征数据
            feature = self.processor.extract_features(npz_path)

            # 转为二进制文件,并保存原始形状,好后续再从数据库读取转回特征数据形式
            binary_feature, original_shape, original_dtype = feature_to_bits(feature)

            # 负数据库字符串
            str_zero, str_one = self.neg_db.create_negative_string(binary_feature)

            # SM3国密算法获取密码哈希,检验数据是否被篡改
            combined_str = str_zero + str_one
            data_hash = encrypt_sm3(combined_str)

            # 负数据库字符串
            encrypted_zero, encrypted_one = None, None
            encrypted_key = None

            if self.sm2:
                # 生成SM4密钥
                sm4_key = os.urandom(16)
                sm4 = Encrypt_sm4(sm4_key) #sm4加密类

                # 加密负数据库字符串
                iv_zero, encrypted_zero_bytes = sm4.encrypt(str_zero)
                iv_one, encrypted_one_bytes = sm4.encrypt(str_one)

                print(f"iv_zero 类型: {type(iv_zero)}, 长度: {len(iv_zero)}")
                print(f"encrypted_zero_bytes 类型: {type(encrypted_zero_bytes)}, 长度: {len(encrypted_zero_bytes)}")
                print(f"iv_one 类型: {type(iv_one)}, 长度: {len(iv_one)}")
                print(f"encrypted_one_bytes 类型: {type(encrypted_one_bytes)}, 长度: {len(encrypted_one_bytes)}")

                # 组合IV和密文用于存储
                encrypted_zero = f"{iv_zero}:{encrypted_zero_bytes}"
                encrypted_one = f"{iv_one}:{encrypted_one_bytes}"

                # 将字节密钥转换为字符串表示并加密
                encrypted_key = self.sm2.encrypt(sm4_key.hex())


            else:
                return {"status": "error", "message": "sm2类不存在"}


            # 检查用户是否存在（任选一条记录判断）
            existing_record = self.db.execute_query(
                "SELECT * FROM user_info_2_temp WHERE user = %s LIMIT 1",
                (user_id,)
            )


            data_to_store_0 = {
                "pwd": encrypted_zero,
                "encrypted_key": encrypted_key,
                "data_hash": data_hash  # 添加SM3哈希值
            }
            data_to_store_1 = {
                "pwd": encrypted_one
            }

            # 存在则更新，不存在则插入
            if existing_record:
                # 更新type=0的记录
                self.db.update_data(
                    table="user_info_2_temp",
                    data=data_to_store_0,
                    condition={"user": user_id, "type": 0}
                )
                # 更新type=1的记录
                self.db.update_data(
                    table="user_info_2_temp",
                    data=data_to_store_1,
                    condition={"user": user_id, "type": 1}
                )

            else:
                # 插入新记录,即注册
                self.db.insert_data(
                    table="user_info_2_temp",
                    data={
                        "user": user_id,
                        "pwd": encrypted_zero,
                        "type": 0,
                        "encrypted_key": encrypted_key,
                        "data_hash": data_hash
                    }
                )
                self.db.insert_data(
                    table="user_info_2_temp",
                    data={
                        "user": user_id,
                        "pwd": encrypted_one,
                        "type": 1
                    }
                )

            return {
                "status": "success",
                "user_id": user_id,
                "digital_info": [original_shape, original_dtype],
                "operation": "updated" if existing_record else "inserted"
            }

        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }



    # 实际使用的验证登录方法
    def verify_neg_db(self, user_id, npz_path):
        """
            用于登录界面:用户输入id,检索数据库中id对应密码,经过解密后传入模型判断是否是本人;

            若成功,返回用户名称(id);失败返回"unknown"
        """



        # 根据 user_id 和 type获取负数据库存储的数据
        # type为 0 即 0 的统计次数
        user_info_0 = self.db.execute_query(
            "SELECT pwd, encrypted_key, data_hash FROM user_info_2_temp WHERE user = %s AND type = 0",
            (user_id,)
        )

        # type为 1 即为 1 的统计次数
        user_info_1 = self.db.execute_query(
            "SELECT pwd FROM user_info_2_temp WHERE user = %s AND type = 1",
            (user_id,)
        )

        if not user_info_0 or not user_info_1:
            return "用户信息不完整,缺少type为0或1的数据"


        # 获取加密数据和SM2加密的SM4密钥
        encrypted_zero = user_info_0[0]['pwd']
        encrypted_one = user_info_1[0]['pwd']
        encrypted_key = user_info_0[0]['encrypted_key']
        stored_hash = user_info_0[0]['data_hash']

        print("type of encrypted_zero",type(encrypted_zero))
        print("len of encrypted_zero",len(encrypted_zero))

        if encrypted_key and self.sm2:
            # 使用SM2解密
            sm4_key_str = self.sm2.decrypt(encrypted_key)
            sm4_key = bytes.fromhex(sm4_key_str)

            # 解析加密数据
            iv_zero, ciphertext_zero = encrypted_zero.split(':', 1)
            iv_one, ciphertext_one = encrypted_one.split(':', 1)

            # 使用SM4解密（直接传入十六进制字符串）
            sm4 = Encrypt_sm4(sm4_key)
            zero_bytes = sm4.decrypt(iv_zero, ciphertext_zero)
            one_bytes = sm4.decrypt(iv_one, ciphertext_one)

            # 将字节转换回字符串
            zero_str = zero_bytes
            one_str = one_bytes

        else:
            return {"status": "error", "message": "加密密钥无效或sm2类不存在"}

        # SM3国密算法验证数据完整性
        combined_bytes = zero_bytes + one_bytes
        current_hash = encrypt_sm3(combined_bytes)

        if stored_hash != current_hash:
            return {"status": "error", "message": "数据完整性验证失败，可能被篡改"}

        # 提取数据库里的0与1的计数
        zero_cnt, one_cnt = self.neg_db.GetBinaryArray(zero_bytes, one_bytes)

        bit_probs = self.neg_db.estimate_original(zero_cnt, one_cnt)
        binary_pred = ['1' if prob < 0.5 else '0' for prob in bit_probs]
        binary_str = ''.join(map(str, binary_pred))


        # 数据库中存的密码
        pre_feature = bits_to_feature(binary_str)
        pre_feature = pre_feature.to(self.processor.model.fc.weight.device)


        # 输入的特征
        input_feature = self.processor.extract_features(npz_path)
        binary_data = feature_to_bits(input_feature)
        input_feature = input_feature.to(self.processor.model.fc.weight.device)
        logits_user = self .processor.model.fc(input_feature)
        probs_user = F.softmax(logits_user, dim=1)
        pre_user = torch.argmax(probs_user).item()


        # 使用模型预测类别
        logits = self.processor.model.fc(pre_feature)
        probs = F.softmax(logits, dim=1)
        pred_class = torch.argmax(probs).item()

        # 加 1是因为从0开始计数
        pre_usernum = f"user_{pre_user+1:03d}"
        verify_user = f"user_{pred_class+1:03d}"

        valid = equal(pre_usernum, verify_user)

        if valid:
            return "登录成功"
        else:
            return "登录失败"



    # 被弃用
    def register_user(self, user_id, npz_path):
        """用户注册流程,原始模板,后续应用加密算法"""
        try:
            # 提取特征数据
            feature = self.processor.extract_features(npz_path)

            # 将压缩后的特征数据存储到数据库
            record_id = self.db.insert_data(  # record_id代表插入的次数,一般不相干
                table="user_info",
                data={
                    "user": user_id,
                    "pwd": feature,
                    "type": 0
                }
            )

            return {
                "status": "success",
                "user_id": user_id,
            }

        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }

    # 之前的思路是提取两个特征之间的相似度,如果大于某个值则认为是,但是需要改变思路,将原本的数据加入后,进行一些模糊化的特征处理
    # 已被弃用
    def verify(self, npz_path, threshold=0.85):
        """
            用于验证传入EEG数据eeg_sample是否在数据库中,身份验证;置信度为0.85,可更改
            若成功,返回用户名称(id);失败返回"unknown"

            注意:这里传入的是npz文件而不是eeg文件,需要更正,暂且将名字改掉,然后固定输入npz文件的第一个数组用于验证,后续考虑随机输入的更改
        """

        feature = self.processor.extract_features(npz_path).squeeze(0)

        # 查询数据库获取所有用户特征
        user_info = self.db.execute_query(
            "SELECT user, pwd FROM user_info_2_temp"
        )

        if not user_info:
            return "数据库中没有数据!"

        #  寻找最佳匹配
        best_match = None
        best_score = -1


        for info in user_info:
            try:
                # 从数据库读取存储的特征
                db_feature = np.frombuffer(info['pwd'], dtype=np.float32)

                # 计算余弦相似度
                score = cosine_similarity([feature], [db_feature])[0][0]

                # 更新最佳匹配
                if score > best_score:
                    best_score = score
                    best_match = info['user']

            except Exception as e:
                print(f"处理用户 {info['user']} 时出错: {str(e)}")
                continue


        if best_score >= threshold:
            return best_match
        else:
            return "Unknown"

    # 反转融合,未使用
    def register_user_IFCB(self, user_id, npz_path, keys):
        """完整的用户注册流程"""
        try:
            # 提取特征数据
            feature = self.processor.extract_features(npz_path)

            pwd = sum_IFCB(feature, keys)
            pwd = serialize_feature(pwd)

            # 将压缩后的特征数据存储到数据库
            record_id = self.db.insert_data(  # record_id代表插入的次数,一般不相干
                table="user_info_3",
                data={
                    "user": user_id,
                    "pwd": pwd
                }
            )

            return {
                "status": "success",
                "user_id": user_id,
            }


        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }

    # 未使用
    def verify_IFCB(self, user_id, npz_path, keys, threshold=0.40, ):
        """
            用于登录界面:用户输入id,检索数据库中id对应密码,比较海明距离验证是否是本人;置信度为0.40,可更改

            暂时返回score,即海明距离
            若成功,返回用户名称(id);失败返回"unknown"
        """

        feature = self.processor.extract_features(npz_path)
        binary_data = feature_to_bits(feature)
        binary_data = sum_IFCB(binary_data, keys)

        # 根据 user_id 和 type 查询用户的密码
        user_info = self.db.execute_query(
            "SELECT pwd FROM user_info_3 WHERE user = %s",
            (user_id,)
        )

        if not user_info:
            return "数据库中没有找到文本!"

        pwd = user_info[0]['pwd']
        pwd = deserialize_feature(pwd)


        score, _ = spearmanr(binary_data[0], pwd[0])

        # if score <= threshold:
        #     return "验证成功"
        # else:
        #     return "Unknown"
        return score





# 常量定义（使用大写命名）
NUM_SAMPLES = 20  # 样本个数
SUBJECT_RANGE = (1, 110)  # S编号范围
RECORD_RANGE = (1, 15)  # R编号范围
ARR_INDEX_RANGE = (0, 240)  # arr_索引范围
DATA_DIR = "16_channels_seg"



def main():
    try:
        # # 1. 使用模型和连接数据库
        # model_path = "models/DCNN_16x80.pth"
        # auth_system = EEGAuthSystem_DCNN(model_path)
        # # 连接数据库
        # auth_system.db.connect()

        # # 2. 创建已注册用户列表（这里假设已注册的用户ID）
        # registered_users = [f"user_{i:03d}" for i in range(1, 109) if i not in [88, 92, 100]]
        # print(f"已注册用户数量: {len(registered_users)}")
        #
        # # 3. 参数设置
        # NUM_TESTS = 200  # 随机测试次数
        # SUBJECT_RANGE = (1, 100)  # 用户ID范围
        # RECORD_RANGE = (1, 14)    # 记录ID范围
        # ARR_INDEX_RANGE = (0, 240) # 数组索引范围
        #
        # # 4. 初始化评估指标
        # y_true = []  # 真实标签 (1=合法用户, 0=非法用户)
        # y_pred = []  # 预测标签 (1=验证通过, 0=验证失败)
        # test_details = []  # 存储每次测试的详细信息
        #
        #
        # TP = 0
        # TN = 0
        # FP = 0
        # FN = 0
        # times = 0
        #
        # # 5. 随机验证循环
        # print(f"\n开始随机验证测试 ({NUM_TESTS}次)...")
        # for i in range(NUM_TESTS):
        #     try:
        #         # 5.1 随机生成测试场景
        #         # 随机选择注册用户 (从已注册用户中选取)
        #         registered_user = random.choice(registered_users)
        #
        #         # 5.2 随机选择脑电数据文件
        #         subject_id = np.random.randint(*SUBJECT_RANGE)
        #         record_id = np.random.randint(*RECORD_RANGE)
        #         arr_index = np.random.randint(*ARR_INDEX_RANGE)
        #
        #         # 跳过无效数据
        #         if subject_id in {88, 92, 100} and record_id in range(3, 15):
        #             continue
        #         if record_id in range(1, 3) and arr_index in range(120, 241):
        #             continue
        #
        #         # 构建文件路径
        #         npz_file = f"16_channels_seg/S{subject_id:03d}R{record_id:02d}.npz"
        #
        #         # 5.3 执行验证
        #         result = auth_system.verify_neg_db(f"user_{subject_id:03d}", npz_file)
        #
        #         times += 1
        #         if result == "登录成功":
        #             TP += 1
        #         else:
        #             FN += 1
        #
        #
        #         print(f"召回率:", TP/times)
        #
        #     except Exception as e:
        #         print(f"测试 {i} 出错: {str(e)}")
        #
        # return 0
        #

        # SM2 密钥配置
        SM2_PUBLIC_KEY = "AEC6B9C5455745BB2A6A1DE0C7C984491C75A30F2D0EE42C2047936472F936B05F1EB909D96DDA4E41A0F433D7DB870763145CBE678D5E9D674ADE740EB1A66F"
        SM2_PRIVATE_KEY = "3610724A9AD7B7CEDBE8C2D6FC858D8426AE919114AA8A5BCBFCB7E75BC6EE29"



        # 1. 使用模型和连接数据库
        model_path = "models/DCNN_16x80.pth"
        auth_system = EEGAuthSystem_DCNN(
            model_path,
            sm2_public_key=SM2_PUBLIC_KEY,
            sm2_private_key=SM2_PRIVATE_KEY
        )
        # 连接数据库
        auth_system.db.connect()


        """
            第一步,注册用户,执行完后注释掉

            注意 user 和 type 两者作为主键,所以再次注册是不可以的,只可以更新
        """
        # result = auth_system.register_user_negdb("user_001", "16_channels_seg/S001R07.npz")
        # if result["status"] == "success":
        #     print(f"注册成功，ID: {result['user_id']}")
        # else:
        #     print("注册失败!")

        """
            第二步,验证用户

            验证流程: 1.输入处理npz文件;2.获取npz文件的数组提取一个特征;3.对比特征与数据库用户.
        """
        npz_file = f"16_channels_seg/S001R02.npz"
        result = auth_system.verify_neg_db('user_001', npz_file)
        print("验证结果:", result)


    except Exception as e:
        print(f"程序运行失败: {e}")

        return 0

if __name__ == "__main__":
    main()

