import time
import pandas as pd

from quant_lib.tushare.tushare_client import TushareClient
from quant_lib.config.logger_config import setup_logger, log_warning

# 配置日志
logger = setup_logger(__name__)

# --- 1. 中央速率控制器 ---
##
# 迫于tushare 老是限制速率！#
class RateLimiter:
    """
    一个简单的、用于控制API调用频率的类。
    """

    def __init__(self, calls_per_minute=380):
        """
        :param calls_per_minute: 每分钟最大调用次数 (为了安全，比官方上限略低)
        """
        self.min_interval = 60.0 / calls_per_minute
        self.last_call_time = 0

    def wait(self):
        """
        检查离上次调用的时间，如有必要则暂停等待。
        """
        elapsed = time.time() - self.last_call_time
        if elapsed < self.min_interval:
            sleep_time = self.min_interval - elapsed
            # print(f"Rate limiting: sleeping for {sleep_time:.3f} seconds...") # 调试时可以打开
            time.sleep(sleep_time)
        # 更新最后调用时间
        self.last_call_time = time.time()


# --- 2. 创建一个全局共享的限速器实例 ---
# 所有的API调用函数都将使用这同一个实例
shared_rate_limiter = RateLimiter(calls_per_minute=2000)


# --- 3. 你的两个API调用函数 (已集成中央限速) ---

# None 表示当前官方文档未公布单次返回上限，不猜测、不套用其他接口的限制。
_API_ROW_LIMITS = {
    'adj_factor': None,
    'balancesheet_vip': None,
    'cashflow_vip': None,
    'daily': 6000,
    'daily_basic': 6000,
    'dividend': None,
    'fina_indicator': 100,
    'fina_indicator_vip': None,
    'hm_detail': 2000,
    'income_vip': None,
    'index_basic': 8000,
    'index_daily': None,
    'index_member': None,
    'index_member_all': 2000,
    'index_weight': None,
    'margin_detail': 6000,
    'namechange': None,
    'pro_bar': None,
    'stk_limit': 5800,
    'stock_basic': 6000,
    'suspend_d': None,
    'sw_daily': 4000,
    'trade_cal': None,
}


def reach_limit(func_name: str, df: pd.DataFrame) -> bool:
    if not isinstance(func_name, str) or not func_name:
        raise TypeError(f"Tushare接口名类型错误: actual={func_name!r}, expected=非空str")
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Tushare接口返回类型错误: actual={type(df).__name__}, expected=DataFrame")
    if func_name not in _API_ROW_LIMITS:
        raise ValueError(f"Tushare接口未配置单次返回上限: api={func_name!r}, expected=根据官方文档明确登记")

    row_limit = _API_ROW_LIMITS[func_name]
    return row_limit is not None and len(df) >= row_limit


def is_token_invalid_error(error_message):
    # 根据你的实现来判断
    return 'token' in error_message or '权限' in error_message or 'ERROR.' in error_message


def call_pro_tushare_api(func_name: str, max_retries=3, **kwargs):
    """
    调用 pro 实例方法的封装 (带限速和自我修复)
    """
    for i in range(max_retries):
        # 核心改动：在调用前，先请求中央限速器进行等待
        shared_rate_limiter.wait()

        try:
            pro = TushareClient.get_pro()
            api_func = getattr(pro, func_name)
            df = api_func(**kwargs)

            if reach_limit(func_name, df):
                # 这个错误非常严重，直接抛出，让上层程序知道数据不完整
                raise ValueError(
                    f"API '{func_name}' 返回条数已达官方单次上限: "
                    f"rows={len(df)}, limit={_API_ROW_LIMITS[func_name]}，数据可能不完整！"
                )
            return df

        except Exception as e:
            # ... (错误处理和Token刷新逻辑保持不变) ...
            error_message = str(e)
            logger.error(f"call_pro_tushare_api调用'{func_name}'失败: {error_message}")
            if is_token_invalid_error(error_message):
                raise ValueError("token问题，无法调用tushare api")
            if i < max_retries - 1:
                log_warning(f"非token导致的报错！！正在进行第 {i + 1}/{max_retries} 次重试...")
                time.sleep(60)

    logger.error(f"call_pro_tushare_api调用'{func_name}'在 {max_retries} 次尝试后彻底失败。")
    raise ValueError("访问api重度异常！！！！立即停止")


def call_ts_tushare_api(func_name: str, max_retries=3, **kwargs):
    """
    调用 ts 实例方法的封装 (带限速)
    根据你的要求，保留此函数用于调用 pro_bar
    """
    for i in range(max_retries):
        # 核心改动：同样请求同一个中央限速器
        shared_rate_limiter.wait()

        try:
            ts = TushareClient.get_ts()
            api_func = getattr(ts, func_name)
            df = api_func(**kwargs)
            if reach_limit(func_name, df):
                # 这个错误非常严重，直接抛出，让上层程序知道数据不完整
                raise ValueError(
                    f"API '{func_name}' 返回条数已达官方单次上限: "
                    f"rows={len(df)}, limit={_API_ROW_LIMITS[func_name]}，数据可能不完整！"
                )
            return df
        except Exception as e:
            # ... (错误处理和Token刷新逻辑保持不变) ...
            error_message = str(e)
            logger.error(f"call_ts_tushare_api调用'{func_name}'失败: {error_message}")
            if is_token_invalid_error(error_message):
                if TushareClient.refresh_pro():
                    logger.info("Token已刷新，正在立即重试...")
                    continue
                else:
                    logger.error("Token刷新失败，终止此API调用。")
                    break
            if i < max_retries - 1:
                log_warning(f"非token导致的报错！！正在进行第 {i + 1}/{max_retries} 次重试...")
                time.sleep(60)

    logger.error(f"call_ts_tushare_api调用'{func_name}'在 {max_retries} 次尝试后彻底失败。")
    raise ValueError(f"访问Tushare ts接口失败: func_name={func_name}, retries={max_retries}")
