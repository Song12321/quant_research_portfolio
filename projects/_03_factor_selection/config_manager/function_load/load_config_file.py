import datetime
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
import yaml

from projects._03_factor_selection.config_manager.function_load.local_config_file_definition import \
    _massive_test_ZZ800_profile, pool_for_massive_test_MICROSTRUCTURE_profile, generate_dynamic_config, \
    CSI300_most_basic_profile, CSI300_none_FFF_most_basic_profile, CSI300_more_filter_profile, \
    CSI500_none_FFF_most_basic_profile, EVAL_SETTING_FULL, EVAL_SETTING_FAST, \
    dongbei_SETTING, fast_hs300_profile, ZZ1000_more_filter_profile, ZZ1000_no_filter_profile, fast_eva_SETTING, \
    HS300_no_filter_profile, ALL_none_FFF_most_basic_profile, fast_ZZ800_profile, ZZ500_more_filter_profile, \
    东北_zz500_profile, HS300_fast_profile
from quant_lib import logger
from quant_lib.config.logger_config import log_warning

fast_periods = ('20190328', '20190612')
fast_train_period = ('20190328', '20200328')
period_东北研报 = ('20220101','20250710')
fast_periods_2 = ('20240301', '20250710')
period_six_year = ('20190710', '20250710')
period_four_year = ('20210710', '20250710')
period_behind_three_year = ('20220710', '20250710')
really_train_period = ('20190328', '20231231')
period_pre_three_year = ('20190710', '20220710')
period_two_year = ('20230601', '20250710')
period_one_year = ('20230601', '20240710')
period_half_year = ('20250101', '20250710')
temp_half_year = ('20200102', '20200810')
test_half_year = ('20230928', '20231231')

longest_periods = ('20190328', '20250710')

massive_test_ZZ800_train_mode = {
    'mode': 'massive_test',
    'pools': {
        **_massive_test_ZZ800_profile
    },
    'period': really_train_period,
    'evaluation': EVAL_SETTING_FULL,  # <--- 【新增】
    'desc': '海量测试环境 zz800股票池+必要过滤  （这是最真实的环境'
}
fast_mode = {
    'mode': 'fast',
    'pools': {
        **fast_ZZ800_profile
    },
    'period': period_one_year,
    'evaluation': fast_eva_SETTING,  # <--- 【新增】
    'desc': '但是只用了沪深300股票池（） ，没有任何过滤 fast'
}

CSI300_most_basic_mode = {
    'mode': 'CSI300_most_basic_profile',
    'pools': {
        **CSI300_most_basic_profile
    },
    'period': period_东北研报,
    'desc': '但是只用了沪深300股票池（）只有普适性过滤，除此之外，没有任何过滤'
}




fast_mode_two_pools = {
    'mode': 'fast',
    'pools': {
        **CSI300_none_FFF_most_basic_profile,
        **CSI500_none_FFF_most_basic_profile
    },
    'period': fast_train_period,
    'desc': 'fast_mode_two_pools ，没有任何过滤 fast'
}

CSI300_more_filter_mode = {
    'mode': 'CSI300_most_basic_profile',
    'pools': {
        **CSI300_more_filter_profile
    },
    'period': period_东北研报,
    'desc': '但是只用了沪深300股票池（）普适性过滤+流动率过滤'
}

东北证券_CSI300_more_filter_mode = {
    'mode': 'CSI300_most_basic_profile',
    'pools': {
        **CSI300_more_filter_profile
    },
    'period': period_东北研报,
    'desc': '但是只用了沪深300股票池（）普适性过滤+流动率过滤'
}

东北证券_CSI1000_more_filter_mode = {
    'mode': '东北证券_CSI1000_more_filter_mode',
    'pools': {
        **ZZ1000_more_filter_profile
    },
    'period': period_东北研报,
    'evaluation': dongbei_SETTING,  # <--- 【新增】
    'desc': 'CSI1000（）普适性过滤+流动率过滤'
}

东北证券_ZZ1000_no_filter_mode = {
    'mode': '东北证券_ZZ1000_no_filter_mode',
    'pools': {
        **ZZ1000_no_filter_profile
    },
    'period': period_two_year,
    'evaluation': dongbei_SETTING,  # <--- 【新增】
    'desc': '东北证券_ZZ1000_no_filter_mode'
}



东北证券_HS300_no_filter_mode = {
    'mode': '东北证券_HS300_no_filter_mode',
    'pools': {
        **HS300_no_filter_profile
    },
    'period': period_one_year,
    'evaluation': fast_eva_SETTING,  # <--- 【新增】
    'desc': '东北证券_HS300_no_filter_mode'
}
CSI300_FFF_most_basic_mode = {
    'mode': 'CSI300_FFF_most_basic_mode',
    'pools': {
        **CSI300_none_FFF_most_basic_profile
    },
    'period': period_东北研报,
    'desc': '但是只用了沪深300股票池（）无普适性过滤，，没有任何过滤'
}
ALL_FFF_most_basic_mode = {
    'mode': 'ALL_none_FFF_most_basic_profile',
    'pools': {
        **ALL_none_FFF_most_basic_profile
    },
    'evaluation': fast_eva_SETTING,  # <--- 【新增】
    'period': period_behind_three_year,#一年单调性就正常
    'desc': 'ALL_none_FFF_most_basic_profile（）无普适性过滤，，没有任何过滤'
}

def check_backtest_periods(start_date, end_date):
    if pd.to_datetime(end_date) - pd.to_datetime(start_date) < datetime.timedelta(days=110):
        raise ValueError("回测时间太短")

################################################################################################################



trans_pram =  {
    'mode': 'massive_test',
    'pools': {
        **_massive_test_ZZ800_profile
    },
    'period': really_train_period,
    'evaluation': EVAL_SETTING_FULL,  # <--- 【新增】
    'desc': '海量测试环境 zz800股票池+必要过滤  （这是最真实的环境'
}


trans_pram =  {
    'mode': 'massive_test',
    'pools': {
        **HS300_fast_profile
    },
    'period': temp_half_year,
    'evaluation': EVAL_SETTING_FAST,  # <--- 【新增】
    'desc': 'fast临时'
}
#
# trans_pram = {
#     'mode': 'fast',
#     'pools': {
#         **fast_ZZ800_profile
#     },
#     'period': test_half_year,
#     'evaluation': fast_eva_SETTING,  # <--- 【新增】
#     'desc': '测试环境'
# }

# 使用包含1日期间的完整测试模式
is_debug = False


def _load_file(config_path: str) -> Dict[str, Any]:
    # confirm_production_mode(massive_test_mode)
    """加载配置文件"""
    config_file = Path(config_path)
    if config_file.exists():
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"配置文件加载成功: {config_path}")
    else:
        raise RuntimeError("未找到config文件")
    return config


def _load_local_config_functional(config_path: str) -> Dict[str, Any]:
    # confirm_production_mode(massive_test_mode)
    """加载配置文件"""
    config = _load_file(config_path)

    # 根据debug模式 修改内容
    # 在这里，根据总开关来决定你的过滤器配置
    log_warning(f"【信息】当前处于  模式，desp: {trans_pram['desc']}。")

    start, end = trans_pram['period']

    dynamic_config = generate_dynamic_config(
        start_date=start, end_date=end,
        pool_profiles=trans_pram['pools']  # 直接取用 dict
    )
    config['backtest']['start_date'] = start
    config['backtest']['end_date'] = end

    config['stock_pool_profiles'] = dynamic_config['stock_pool_profiles']

    # --- 【核心新增】动态更新 evaluation 配置 ---
    if 'evaluation' in trans_pram:
        logger.info("  > 发现模式中的 [evaluation] 配置，正在应用...")
        # 使用 .update() 可以灵活地覆盖YAML中的部分或全部设置
        # 比如模式中只定义了 forward_periods，则只会更新它，n_groups等会保持YAML中的默认值
        config['evaluation'].update(trans_pram['evaluation'])
    return config


def confirm_production_mode(is_debug_mode: bool, task_name: str = "批量因子测试"):
    """
    一个安全确认函数，防止在调试模式下运行生产级任务。
    """
    if is_debug_mode:
        warning_message = f"""
#################################################################
#                                                               #
#   警告! 警告! 警告!  (WARNING! WARNING! WARNING!)             #
#                                                               #
#   您正准备以【调试模式】(DEBUG MODE)运行 '{task_name}'！        #
#                                                               #
#   在此模式下，ST股、停牌股、新股等关键过滤器已被禁用！            #
#   产出的结果将是【失真】且【不可信】的，仅供快速代码调试使用！    #
#                                                               #
#   如要运行正式测试，请在主脚本中设置 IS_DEBUG_MODE = False      #
#                                                               #
#################################################################
"""
        print(warning_message)
        print("程序将在5秒后继续，以便您有时间终止。")
        seconds = 1
        print("如果您确认要在调试模式下继续，请等待...")
        try:
            for i in range(seconds, 0, -1):
                print(f"...{i}")
                time.sleep(1)
        except KeyboardInterrupt:
            print("\\n操作已由用户终止。")
            exit()  # 直接退出程序
        print("继续执行调试模式任务...")


if __name__ == '__main__':
    config = _load_local_config_functional(
        'D:\\lqs\\codeAbout\\py\\Quantitative\\quant_research_portfolio\\projects\\_03_factor_selection\\factory\\config.yaml')
