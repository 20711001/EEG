import os
from gmssl.sm2 import CryptSM2
from gmssl.sm3 import sm3_hash
from gmssl.sm4 import CryptSM4, SM4_ENCRYPT, SM4_DECRYPT

"""
    国密算法的实现,sm_2,sm_3和sm_4算法
"""

"""
    国密算法SM_2,使用时需给定公钥和私钥,传入公钥和私钥初始化模型,使用模型的encrypt和decrypt函数进行加密和解密
"""


class Encrypt_sm2:
    def __init__(self, private_key, public_key):
        """
        初始化SM2加密器
        """
        self.crypt = CryptSM2(public_key=public_key, private_key=private_key)

    def encrypt(self, plaintext: str) -> str:
        """
        加密数据

        :param plaintext: 要加密的明文
        :return: 十六进制格式的密文
        """
        # 加密
        encrypted = self.crypt.encrypt(plaintext.encode("utf-8"))

        return encrypted

    def decrypt(self, ciphertext: str) -> str:
        """
        解密

        :param ciphertext: 十六进制格式的密文
        :return: 解密后的明文字符串
        """
        encrypted = bytes(ciphertext)
        decrypted = self.crypt.decrypt(encrypted)

        return decrypted.decode('utf-8')


"""
    国密算法SM3,单向不可解密
"""


def encrypt_sm3(message):
    """
    计算 SM3 哈希值

    :param message: 输入字符
    :return: 64字符的十六进制哈希值
    """
    # 将字符串编码为 UTF-8 字节
    message_bytes = message.encode("utf-8")

    # 转换为 bytearray（因为 sm3_hash 需要bytearray形式）
    message_bytearray = bytearray(message_bytes)

    hash_result = sm3_hash(message_bytearray)
    return hash_result


"""
    国密算法SM_4,使用时需给定密钥(必须是16字节),传入密钥初始化模型,使用模型的encrypt和decrypt函数进行加密和解密
"""


class Encrypt_sm4:
    def __init__(self, key: bytes):
        """
        初始化SM4加密器
        """
        if len(key) != 16:
            raise ValueError("SM4密钥必须是16字节长度")

        self.key = key
        self.crypt = CryptSM4()
        self.crypt.set_key(self.key, SM4_ENCRYPT)
        self.iv = os.urandom(16)

    def encrypt(self, plaintext: str) -> str:
        """
        加密数据

        :param plaintext: 要加密的明文
        :return: 十六进制格式的密文
        """
        # 将明文转换为字节
        plaintext_bytes = plaintext.encode('utf-8')

        # 计算填充长度
        pad_len = 16 - (len(plaintext_bytes) % 16)
        print(plaintext_bytes)

        # 填充数据到16字节倍数
        padded_data = plaintext_bytes + bytes([pad_len]) * pad_len
        print(padded_data)

        # 加密
        encrypted = self.crypt.crypt_cbc(self.iv, padded_data)

        return encrypted.hex()

    def decrypt(self, ciphertext: str) -> str:
        """
        解密

        :param ciphertext: 十六进制格式的密文
        :return: 解密后的明文字符串
        """
        encrypted = bytes.fromhex(ciphertext)

        # CBC模式解密
        crypt = CryptSM4()
        crypt.set_key(self.key, SM4_DECRYPT)
        decrypted = crypt.crypt_cbc(self.iv, encrypted)  # 使用CBC模式

        pad_len = decrypted[-1]
        if pad_len < 1 or pad_len > 16:
            raise ValueError("Invalid padding length")
        return decrypted[:-pad_len].decode('utf-8')


# 使用示例
if __name__ == "__main__":
    print("以下为各个算法的测试用例,取消注释即可使用")

    # sm2算法测试用例
    private_key = "3610724A9AD7B7CEDBE8C2D6FC858D8426AE919114AA8A5BCBFCB7E75BC6EE29"
    public_key = "AEC6B9C5455745BB2A6A1DE0C7C984491C75A30F2D0EE42C2047936472F936B05F1EB909D96DDA4E41A0F433D7DB870763145CBE678D5E9D674ADE740EB1A66F"

    # 初始化加密器
    sm2 = Encrypt_sm2(private_key, public_key)

    # 测试加密解密
    plaintext = "Hello, SM2! 这是一个测试消息。"
    print("原始数据:", plaintext)

    encrypted = sm2.encrypt(plaintext)
    print("加密结果:", encrypted)

    decrypted = sm2.decrypt(encrypted)
    print("解密结果:", decrypted)

    # sm3算法测试用例
    data = "Hello, SM3! 你好，国密哈希！"
    hash_value = encrypt_sm3(data)
    print(f"SM3 哈希值: {hash_value}")

    # sm4算法测试用例
    sm4key = b'*`\xc0\x17j\xef\xa2{\x8c!\xd9\xb6\xf9}a\xbb'  # 16字节密钥
    sm4 = Encrypt_sm4(sm4key)

