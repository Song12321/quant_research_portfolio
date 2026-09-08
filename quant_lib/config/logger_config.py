import logging
import os
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

from quant_lib.config.symbols_constants import SUCCESS, WARNING, FAIL, RUNNING


def setup_logger(
        name: str = None,
        console_level: str = 'DEBUG',
        log_dir: str = r'D:\lqs\codeAbout\py\Quantitative\import_file\quant_research_portfolio\log',
        file_level: str = 'DEBUG'
) -> logging.Logger:
    """
    设置一个功能强大的日志记录器，支持双通道输出、日志轮转和差异化级别。

    :param name: 日志记录器的名称，通常是 __name__。
    :param console_level: 控制台输出的日志级别。
    :param log_dir: 日志文件存放的目录。
    :param file_level: 文件记录的日志级别。
    :return: 配置好的 logger 对象。
    """
    logger_name = name or 'quant_lib'
    logger = logging.getLogger(logger_name)

    # 如果logger已经有handlers，直接返回，避免重复配置
    if logger.handlers:
        return logger

    # 设置logger的最低处理级别，这是所有handler的“总开关”
    # 必须设置为console和file中更低的级别，否则低级别的日志会被直接过滤掉
    lowest_level = min(getattr(logging, console_level.upper()), getattr(logging, file_level.upper()))
    logger.setLevel(lowest_level)

    # ------------------- 1. 配置控制台 Handler -------------------
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, console_level.upper()))
    console_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s',  # 控制台格式可以简洁一些
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # ------------------- 2. 配置文件 Handler (新增部分) -------------------
    # 确保日志目录存在
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # 定义日志文件名
    # 每个进程独立写入，避免 Windows 下轮转重命名被其他进程占用。
    log_file = Path(log_dir) / f"{logger_name}.{os.getpid()}.log"

    # 使用 TimedRotatingFileHandler 实现日志按天轮转
    # when='D': 按天轮转; interval=1: 每天一个新文件
    # backupCount=30: 保留最近30天的日志文件
    file_handler = TimedRotatingFileHandler(
        filename=log_file,
        when='D',
        interval=1,
        backupCount=30,
        encoding='utf-8'
    )
    file_handler.setLevel(getattr(logging, file_level.upper()))
    file_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s',  # 文件格式可以更详细
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # 防止向上传播（避免重复输出），保持不变
    logger.propagate = False

    return logger

logger = setup_logger(__name__, console_level='DEBUG', file_level='DEBUG')
def log_flow_start(msg): logger.info(f"{RUNNING} {msg}")
def log_success(msg): logger.info(f"{SUCCESS} {msg}")
def log_warning(msg): logger.info(f"{WARNING} {msg}")
def log_notice(msg): logger.info(f"{FAIL} {msg}")
def log_error(msg): logger.info(f"{FAIL} {msg}")
