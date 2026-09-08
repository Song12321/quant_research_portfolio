import tushare as ts
from tushare.pro.client import DataApi

from quant_lib.tushare.tushare_token_manager.token_manager import load_token_from_local, refresh_token

# Pro 客户端和 ts.pro_bar 内部创建的客户端使用同一个代理地址。
DataApi._DataApi__http_url = "https://t.xiaodefa.top/dataapi"

##
#
# 主要就读取本地json。拿到token，用于初始化tusahre。拿到ts 、pro对象。便于调用thshare的api#
class TushareClient:
    _pro = None  # 私有类变量，用于存储全局唯一的pro实例
    _ts = None  # 私有类变量，用于存储全局唯一的ts实例

    @classmethod
    def get_pro(cls):
        """获取全局唯一的Tushare Pro API实例。如果不存在则创建。"""
        if cls._pro is None:
            token = load_token_from_local()
            if not token:
                raise ConnectionError("本地缓存未配置 Token，无法初始化 API。")
            print("正在初始化Tushare API实例...")
            cls._pro = ts.pro_api(token)
            print("Tushare API实例初始化成功。")
        return cls._pro

    @classmethod
    def get_ts(cls):
        """获取全局唯一的Tushare Pro API实例。如果不存在则创建。"""
        if cls._ts is None:
            token = load_token_from_local()
            if not token:
                raise ConnectionError("本地缓存未配置 Token，无法初始化 API。")
            print("_ts正在初始化Tushare API实例...")
            ts.set_token(token)
            cls._ts = ts
            print("_tsTushare API实例初始化成功。")
        return cls._ts

    @classmethod
    def refresh_pro(cls):
        """核心的自我修复逻辑：获取新Token，并用它替换掉旧的API实例。"""
        print("检测到Token失效，启动刷新流程...")
        try:
            new_token = refresh_token()
            if new_token:
                cls._pro = ts.pro_api(new_token)  # <-- 看这里！用新Token创建全新的实例
                print("Token刷新成功，API实例已更新！")
                return True
            else:
                print("未能找到有效的新Token。")
                return False
        except Exception as e:
            print(f"刷新Token时发生未知错误: {e}")
            return False

    @classmethod
    def refresh_ts(cls):
        """核心的自我修复逻辑：获取新Token，并用它替换掉旧的API实例。"""
        print("_ts检测到Token失效，启动刷新流程...")
        try:
            new_token = refresh_token()
            if new_token:
                cls._ts = ts.set_token(new_token)  # <-- 看这里！用新Token创建全新的实例
                print("_tsToken刷新成功，API实例_ts已更新！")
                return True
            else:
                print("_ts未能找到有效的新Token。")
                return False
        except Exception as e:
            print(f"_ts刷新Token时发生未知错误: {e}")
            return False
