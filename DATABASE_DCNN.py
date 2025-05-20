import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
import random
import pymysql
from pymysql import Error
from Model_3 import DCNN

"""
    该文件用于DCNN模型与数据库之间的交互,模型为models文件夹下的DCNN_16x80.pth,数据库连接本地数据库
"""


def load_model_DCNN(model_path, num_classes):
    """
        加载模型,并使用GPU设备
        返回模型和GPU设备
    """
    model = DCNN(num_classes=num_classes)

    # 安全加载模型权重
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    return model, device

def preprocess_data(eeg_data):
    """
        对EEG数据进行预处理,预处理为数据加维度,这样才能作为特征(或说密码)提取的对象
    """
    # 确保输入是(16, 80)的numpy数组
    assert eeg_data.shape == (16, 80), f"需要(16,80)的输入，但得到{eeg_data.shape}"

    return torch.tensor(eeg_data, dtype=torch.float32).unsqueeze(0)

def preprocess_npz(npz_path, required_shape=(16, 80)):
    """
        加载NPZ文件中的EEG数据并对其进行预处理
        返回可以被特征提取(batch_size, 16, 80)的形状的数据
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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _ = load_model_DCNN(model_path,num_classes=109)
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
                tensor = torch.from_numpy(eeg).float().to(self.device).unsqueeze(0)


                with torch.no_grad():
                    feature = self.model.extract_features(tensor).cpu().numpy()[0]
                    features.append(feature)
            except Exception as e:
                print(f"处理片段 {key} 时出错: {str(e)}")

        if not features:
            raise ValueError(f"未找到有效EEG片段: {npz_path}")
        return np.mean(features, axis=0)


class EEGAuthSystem_DCNN:
    """
        将前面两个类:数据库和模型数据处理放一起
    """
    def __init__(self, model_path):
        self.processor = EEGProcessor_DCNN(model_path)
        self.db = DatabaseManager()

    def register_user(self, user_id, npz_path):
        """完整的用户注册流程"""
        try:
            # 提取特征数据
            feature = self.processor.extract_features(npz_path)


            # 将压缩后的特征数据存储到数据库
            record_id = self.db.insert_data( # record_id代表插入的次数,一般不相干
                table="user_info_2",
                data={
                    "user": user_id,
                    "pwd": feature.tobytes()
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

        features = self.processor.extract_features(npz_path).squeeze()

        # 查询用户
        user_info = self.db.execute_query(
            "SELECT user, pwd FROM user_info_2"
        )
        if not user_info:

            return "数据库中没有数据!"

        best_match, best_score = None, -1
        for info in user_info:
            try:
                feature = np.frombuffer(info['pwd'], dtype=np.float32)

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
    auth_system = EEGAuthSystem_DCNN(
        model_path="models/DCNN_16x80.pth",
    )


    # 连接数据库
    auth_system.db.connect()


    """
        第一步,注册用户,执行完后注释掉
    """
    # # 注册新用户
    # result_1 = auth_system.register_user("user_001", "16_channels_seg/S001R07.npz")
    # if result_1["status"] == "success":
    #     print(f"注册成功，ID: {result_1['user_id']}")
    # else:
    #     print("注册失败!")
    #
    # result_2 = auth_system.register_user("user_002", "16_channels_seg/S002R07.npz")
    # if result_2["status"] == "success":
    #     print(f"注册成功，ID: {result_2['user_id']}")
    # else:
    #     print("注册失败!")
    #
    # result_3 = auth_system.register_user("user_003", "16_channels_seg/S003R07.npz")
    # if result_3["status"] == "success":
    #     print(f"注册成功，ID: {result_3['user_id']}")
    # else:
    #     print("注册失败!")




    """
        第二步,验证用户
        
        验证流程:1.输入处理npz文件;2.随机获取npz文件的几个(16,80)数组提取特征;3.对比特征与数据库用户.
    """
    # 验证用户
    npz_file = f"16_channels_seg/S003R02.npz"

    result = auth_system.verify(npz_file)
    # 打印验证结果
    print("验证结果:", result)




