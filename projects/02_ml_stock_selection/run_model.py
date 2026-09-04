"""
机器学习选股模型脚本

执行机器学习选股模型的训练、预测和回测流程。
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import logging
import datetime
from pathlib import Path
import argparse
import joblib

# 添加项目根目录到路径，以便导入自定义模块
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from quant_lib.data_loader import DataLoader
from quant_lib.ml_utils import (
    create_model_trainer,
    TimeSeriesFeatureGenerator
)
from quant_lib.backtesting import create_backtest_engine
from quant_lib.utils.file_utils import (
    ensure_dir_exists,
    save_to_csv,
    save_to_pickle,
    load_from_yaml
)
from quant_lib.config.constant_config import (
    MODEL_DIR,
    RESULT_DIR
)
from quant_lib.config.logger_config import setup_logger

# 配置日志
logger = setup_logger(__name__)


def load_config(config_path: str) -> dict:
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置字典
    """
    logger.info(f"加载配置文件: {config_path}")
    return load_from_yaml(config_path)


# def load_data(config_manager: dict) -> dict:
#     """
#     加载数据
#
#     Args:
#         config_manager: 配置字典
#
#     Returns:
#         数据字典
#     """
#     logger.info("开始加载数据...")
#
#     # 创建数据加载器
#     data_loader = DataLoader()
#
#     # 确定需要加载的字段
#     fields = []
#
#     # 添加价格特征
#     fields.extend(config_manager['features']['price_features'])
#
#     # 添加基本面特征
#     fields.extend(config_manager['features']['fundamental_features'])
#
#     # 加载数据
#     data_dict = data_loader.get_raw_dfs_by_require_fields(fields=fields, start_date=config_manager['start_date'],
#                                                           end_date=config_manager['end_date'])
#
#     logger.info(f"数据加载完成，共加载 {len(data_dict)} 个字段")
#     return data_dict


def generate_features(config: dict, data_dict: dict) -> tuple:
    """
    生成特征
    
    Args:
        config: 配置字典
        data_dict: 数据字典
        
    Returns:
        (特征DataFrame, 目标变量Series)
    """
    logger.info("开始生成特征...")
    
    # 创建特征生成器
    feature_generator = TimeSeriesFeatureGenerator()
    
    # 获取价格数据作为基础数据
    price_df = data_dict['close'].copy()
    
    # 创建特征DataFrame
    feature_df = pd.DataFrame(index=price_df.index)
    
    # 添加价格特征
    for field in config['features']['price_features']:
        if field in data_dict:
            feature_df[field] = data_dict[field].mean(axis=1)
    
    # 添加基本面特征
    for field in config['features']['fundamental_features']:
        if field in data_dict:
            feature_df[field] = data_dict[field].mean(axis=1)
    
    # 计算技术指标
    for tech_feature in config['features']['technical_features']:
        if tech_feature['name'] == 'ma':
            for window in tech_feature['windows']:
                feature_df[f'ma_{window}'] = price_df.mean(axis=1).rolling(window=window).mean()
        elif tech_feature['name'] == 'rsi':
            for window in tech_feature['windows']:
                # 简化的RSI计算
                price_change = price_df.mean(axis=1).diff()
                gain = price_change.where(price_change > 0, 0).rolling(window=window).mean()
                loss = -price_change.where(price_change < 0, 0).rolling(window=window).mean()
                rs = gain / loss
                feature_df[f'rsi_{window}'] = 100 - (100 / (1 + rs))
    
    # 生成时间序列特征
    time_series_config = config['features']['time_series']
    
    # 选择一个目标列（这里使用收盘价均值）
    target_col = 'close_mean'
    feature_df[target_col] = data_dict['close'].mean(axis=1)
    
    # 生成时间序列特征
    X, _ = feature_generator.generate_features(
        df=feature_df,
        target_col=target_col,
        lag_periods=time_series_config['lag_periods'],
        rolling_windows=time_series_config['rolling_windows'],
        date_features=time_series_config['date_features']
    )
    
    # 生成目标变量
    target_config = config['target']
    forward_period = target_config['forward_period']
    
    if target_config['type'] == 'return':
        # 计算未来收益率
        y = data_dict['close'].mean(axis=1).shift(-forward_period) / data_dict['close'].mean(axis=1) - 1
    else:  # classification
        # 计算未来收益率并转换为分类标签
        future_return = data_dict['close'].mean(axis=1).shift(-forward_period) / data_dict['close'].mean(axis=1) - 1
        threshold = target_config['classification_threshold']
        y = (future_return > threshold).astype(int)
    
    # 对齐数据
    common_index = X.index.intersection(y.dropna().index)
    X = X.loc[common_index]
    y = y.loc[common_index]
    
    logger.info(f"特征生成完成，共 {X.shape[1]} 个特征，{len(y)} 个样本")
    return X, y


def train_model(config: dict, X: pd.DataFrame, y: pd.Series, model_dir: Path) -> tuple:
    """
    训练模型
    
    Args:
        config: 配置字典
        X: 特征DataFrame
        y: 目标变量Series
        model_dir: 模型保存目录
        
    Returns:
        (模型, 评估指标)
    """
    logger.info("开始训练模型...")
    
    # 获取模型配置
    model_config = config['model']
    
    # 创建模型训练器
    trainer = create_model_trainer(
        model_type=model_config['type'],
        task_type=model_config['task_type']
    )
    
    # 训练模型
    metrics = trainer.train(
        X=X,
        y=y,
        test_size=model_config['test_size'],
        cv=model_config['cv'],
        tune_hyperparams=model_config['tune_hyperparams'],
        param_grid=model_config['param_grid']
    )
    
    # 保存模型
    model_path = model_dir / 'model.pkl'
    trainer.save_model(model_path)
    
    # 绘制特征重要性
    plt.figure(figsize=(12, 8))
    trainer.plot_feature_importance()
    plt.savefig(model_dir / 'feature_importance.png')
    
    logger.info(f"模型训练完成，评估指标: {metrics}")
    return trainer, metrics


def run_backtest(config: dict, data_dict: dict, model, X: pd.DataFrame, result_dir: Path) -> dict:
    """
    执行回测
    
    Args:
        config: 配置字典
        data_dict: 数据字典
        model: 训练好的模型
        X: 特征DataFrame
        result_dir: 结果保存目录
        
    Returns:
        回测结果
    """
    # logger.info("开始执行回测...")
    
    # 获取回测参数
    backtest_config = config['backtest']
    
    # 预测
    predictions = model.predict(X)
    
    # 创建因子DataFrame
    factor_df = pd.DataFrame(
        data=predictions,
        index=X.index,
        columns=data_dict['close'].columns
    )
    
    # 创建回测引擎
    engine = create_backtest_engine(
        start_date=config['start_date'],
        end_date=config['end_date'],
        stack_pool=list(data_dict['close'].columns),
        rebalance_freq=backtest_config['rebalance_freq'],
        n_stocks=backtest_config['n_stocks'],
        fee_rate=backtest_config['fee_rate'],
        slippage=backtest_config['slippage'],
        benchmark=backtest_config['benchmark'],
        capital=backtest_config['capital']
    )
    
    # 准备基准收益率数据
    # 实际项目中应该从数据源加载基准指数数据
    # 这里简化处理，使用等权组合作为基准
    benchmark_returns = data_dict['close'].pct_change().mean(axis=1)
    
    # 执行回测
    result = engine.run(
        price_df=data_dict['close'],
        factor_df=factor_df,
        benchmark_returns=benchmark_returns
    )
    
    logger.info("回测完成")
    return result


def save_results(result, metrics: dict, result_dir: Path) -> None:
    """
    保存回测结果
    
    Args:
        result: 回测结果对象
        metrics: 模型评估指标
        result_dir: 结果保存目录
    """
    logger.info("保存回测结果...")
    
    # 保存模型评估指标
    pd.DataFrame(metrics, index=[0]).to_csv(result_dir / 'model_metrics.csv')
    
    # 保存性能指标
    metrics_df = result.summary()
    save_to_csv(metrics_df, result_dir / 'performance_metrics.csv')
    
    # 保存收益率序列
    returns_df = pd.DataFrame({
        'portfolio': result.portfolio_returns,
        'benchmark': result.benchmark_returns
    })
    save_to_csv(returns_df, result_dir / 'returns.csv')
    
    # 保存持仓信息
    save_to_csv(result.positions, result_dir / 'positions.csv')
    
    # 保存换手率
    save_to_csv(result.turnover.to_frame('turnover'), result_dir / 'turnover.csv')
    
    # 绘制收益曲线
    result.plot_returns()
    plt.savefig(result_dir / 'cumulative_returns.png')
    
    # 绘制回撤曲线
    result.plot_drawdown()
    plt.savefig(result_dir / 'drawdown.png')
    
    # 绘制月度收益热力图
    result.plot_monthly_returns()
    plt.savefig(result_dir / 'monthly_returns.png')
    
    logger.info(f"回测结果已保存至: {result_dir}")


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='机器学习选股模型')
    parser.add_argument('--config_manager', type=str, default='config.yaml', help='配置文件路径')
    args = parser.parse_args()
    
    # 获取当前脚本所在目录
    current_dir = Path(__file__).parent
    
    # 加载配置
    config_path = current_dir / args.config
    config = load_config(config_path)
    
    # 创建结果目录
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    model_dir = MODEL_DIR / 'ml_stock_selection' / timestamp
    result_dir = RESULT_DIR / 'ml_stock_selection' / timestamp
    ensure_dir_exists(model_dir)
    ensure_dir_exists(result_dir)
    
    # 保存配置副本
    with open(result_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # 加载数据
    data_dict = load_data(config)
    
    # 生成特征
    X, y = generate_features(config, data_dict)
    
    # 训练模型
    model, metrics = train_model(config, X, y, model_dir)
    
    # 执行回测
    result = run_backtest(config, data_dict, model, X, result_dir)
    
    # 保存结果
    save_results(result, metrics, result_dir)
    
    # 打印性能摘要
    print("\n=== 回测性能摘要 ===")
    print(result.summary())


if __name__ == '__main__':
    main()
