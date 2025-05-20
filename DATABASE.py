import numpy as np
import torch
from Models import VeCNNNet
from sklearn.metrics.pairwise import cosine_similarity
import random
import pymysql
from pymysql import Error
import zlib

"""
    主要模型还是DCNN模型,所以这个可以不管
"""

"""
    该文件用于ve_cnn模型与数据库之间的交互,模型为models文件夹下的DCNN_16x80.pth,数据库连接本地数据库
"""


def load_model(model_path, num_classes):
    """
        加载模型,并使用GPU设备
        返回模型和GPU设备
    """
    model = VeCNNNet(input_channels=1, num_classes=num_classes)

    # 安全加载模型权重
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    return model, device

def preprocess_npz(npz_path, required_shape=(16, 80)):
    """
        加载NPZ文件中的EEG数据并对其进行预处理
    """
    # 加载NPZ文件
    data = np.load(npz_path)

    # 获取EEG片段
    valid_keys = [k for k in data.files if data[k].shape == required_shape]
    if not valid_keys:
        raise ValueError(f"NPZ文件中没有形状为{required_shape}的数据")

    # 取一个固定值,后续考虑随机
    eeg_data = data[valid_keys[0]]  # 取第一个符合条件的片段

    # 数据预处理
    return preprocess_data(eeg_data)

def preprocess_data(eeg_data):
    """
        对EEG数据进行预处理
    """
    # 确保输入是(16, 80)的numpy数组
    assert eeg_data.shape == (16, 80), f"需要(16,80)的输入，但得到{eeg_data.shape}"

    # 添加batch和channel维度 -> (1, 1, 16, 80)
    return torch.tensor(eeg_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

def compre_feature(feature):
    """
        压缩特征,否则数据库无法存储,后续考虑其他解决方法
        接收feature,返回16进制的压缩字节
    """
    original_data = feature.tobytes()

    compressed_data = zlib.compress(original_data)

    return compressed_data

def decompre_feature(compre_data):
    """
        进行解压缩,与compre_feature搭配使用
        接收压缩的data,返回解压的特征值
    """
    decompre_data = zlib.decompress(compre_data)

    return decompre_data


class DatabaseManager:
    def __init__(self):
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

    def insert_data(self, table, data):
        """
        执行插入数据操作,主要用于注册用户操作
        接收 "table":插入的表名称和 "data":插入数据,其中插入数据以字典形式保留,如 {'user': 'user_1','pwd':'123456'}
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


class EEGProcessor:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _ = load_model(model_path,num_classes=109)
        self.templates = {}  # 格式: {user_id: [feature_vec1, feature_vec2,...]}

    def extract_features(self, npz_path):
        """
            从npz文件提取特征,默认提取五个特征
        """
        data = np.load(npz_path)
        features = []
        available_keys = [k for k in data.files if data[k].shape == (16, 80)]

        for key in random.sample(available_keys, 5):
            try:
                eeg = data[key]
                tensor = torch.from_numpy(eeg).float().unsqueeze(0).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    feature = self.model.extract_features(tensor).cpu().numpy()[0]
                    features.append(feature)
            except Exception as e:
                print(f"处理片段 {key} 时出错: {str(e)}")

        if not features:
            raise ValueError(f"未找到有效EEG片段: {npz_path}")
        return np.mean(features, axis=0)


class EEGAuthSystem:
    """身份验证系统（整合层）"""
    def __init__(self, model_path):
        self.processor = EEGProcessor(model_path)
        self.db = DatabaseManager()

    def register_user(self, user_id, npz_path):
        """完整的用户注册流程"""
        try:
            # 提取特征数据
            feature = self.processor.extract_features(npz_path)
            feature = compre_feature(feature)
            # 将压缩后的特征数据存储到数据库
            record_id = self.db.insert_data( # record_id代表插入的次数,一般不相干
                table="user_info",
                data={
                    "user": user_id,
                    "pwd": feature
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

    def verify(self, npz_path, threshold=0.85):
        """
            用于验证传入EEG数据eeg_sample是否在数据库中,身份验证;置信度为0.85,可更改
            若成功,返回用户名称(id);失败返回"unknown"

            注意:这里传入的是npz文件而不是eeg文件,需要更正,暂且将名字改掉,然后固定输入npz文件的第一个数组用于验证,后续考虑随机输入的更改
        """

        features = self.processor.extract_features(npz_path)

        # 查询用户
        user_info = self.db.execute_query(
            "SELECT user, pwd FROM user_info"
        )
        if not user_info:
            return "数据库中没有数据!"

        best_match, best_score = None, -1
        for info in user_info:
            try:
                # 解压数据
                decom_data = decompre_feature(info["pwd"])
                feature = np.frombuffer(decom_data, dtype=np.float32)

                # 计算相似度得分
                score = cosine_similarity([features], [feature])[0][0]
                if score > best_score:
                    best_score, best_match = score, info['user']

            except Exception as process_error:
                print(f"处理用户 {info['user']} 时出错: {str(process_error)}")
                continue


        if best_score >= threshold:
            return best_match
        else:
            return "Unknown"


if __name__ == "__main__":

    # 初始化系统
    auth_system = EEGAuthSystem(
        model_path="models/ve_cnn_16x80_sec.pth",
    )
    # 连接数据库
    auth_system.db.connect()

    # 注册新用户
    result = auth_system.register_user("user_009", "16_channels_seg/S009R07.npz")
    if result["status"] == "success":
        print(f"注册成功，ID: {result['user_id']}")
    else:
        print("注册失败!")



    """
        验证流程:1.输入处理npz文件;2.随机获取npz文件的几个(16,80)数组提取特征;3.获取数据库所有数据然后解压,转换;4.对比特征与数据库用户.
    """
    # 验证用户
    npz_file = "16_channels_seg/S009R07.npz"

    result = auth_system.verify(npz_file)
    # 打印验证结果
    print("验证结果:", result)

