"""
并发执行器 - 用于批量因子计算的高效并发处理

支持功能：
1. 多线程并发执行
2. 进度监控和日志记录
3. 异常处理和重试机制
4. 资源管理和内存控制
5. 可配置的并发参数
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Callable, Any, Dict, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import logging
from quant_lib.config.logger_config import setup_logger

logger = setup_logger(__name__)


@dataclass
class ConcurrentConfig:
    """并发配置"""
    max_workers: int = 4  # 最大工作线程数
    chunk_size: int = 1  # 每个任务的批次大小
    timeout: Optional[float] = 60000  # 单个任务超时时间(秒)
    retry_count: int = 1  # 失败重试次数
    log_interval: int = 10  # 进度日志间隔(秒)


class ConcurrentExecutor:
    """通用并发执行器"""
    
    def __init__(self, config: Optional[ConcurrentConfig] = None):
        self.config = config or ConcurrentConfig()
        self._completed_count = 0
        self._failed_count = 0
        self._total_count = 0
        self._lock = threading.Lock()
        self._start_time = None
        
    def execute_batch(
        self,
        target_function: Callable,
        task_list: List[Any],
        task_name: str = "批量任务",
        **kwargs
    ) -> Tuple[List[Any], List[Tuple[Any, Exception]]]:
        """
        批量并发执行任务
        
        Args:
            target_function: 目标执行函数
            task_list: 任务参数列表
            task_name: 任务名称(用于日志)
            **kwargs: 传递给目标函数的额外参数
            
        Returns:
            (successful_results, failed_tasks): 成功结果列表和失败任务列表
        """
        self._total_count = len(task_list)
        self._completed_count = 0
        self._failed_count = 0
        self._start_time = time.time()
        
        logger.info(f"🚀 开始{task_name}，共{self._total_count}个任务，并发度{self.config.max_workers}")
        
        successful_results = []
        failed_tasks = []
        
        # 启动进度监控线程
        progress_thread = threading.Thread(target=self._progress_monitor, daemon=True)
        progress_thread.start()
        
        try:
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                # 提交所有任务
                future_to_task = {
                    executor.submit(self._execute_with_retry, target_function, task, **kwargs): task
                    for task in task_list
                }
                
                # 收集结果
                for future in as_completed(future_to_task, timeout=self.config.timeout * len(task_list)):
                    task = future_to_task[future]
                    
                    try:
                        result = future.result()
                        successful_results.append(result)
                        
                        with self._lock:
                            self._completed_count += 1
                            
                    except Exception as e:
                        failed_tasks.append((task, e))
                        logger.error(f"任务失败: {task} - {e}")
                        
                        with self._lock:
                            self._failed_count += 1
                            
        except Exception as e:
            logger.error(f"批量执行异常: {e}")
            
        finally:
            # 最终统计
            elapsed_time = time.time() - self._start_time
            success_rate = (self._completed_count / self._total_count) * 100 if self._total_count > 0 else 0
            
            logger.info(f"✅ {task_name}完成")
            logger.info(f"📊 成功: {self._completed_count}, 失败: {self._failed_count}, 成功率: {success_rate:.1f}%")
            logger.info(f"⏱️ 总耗时: {elapsed_time:.1f}秒, 平均每任务: {elapsed_time/self._total_count:.1f}秒")
            
        return successful_results, failed_tasks
    
    def _execute_with_retry(self, target_function: Callable, task: Any, **kwargs) -> Any:
        """带重试机制的任务执行"""
        last_exception = None
        
        for attempt in range(self.config.retry_count + 1):
            try:
                return target_function(task, **kwargs)
                
            except Exception as e:
                last_exception = e
                if attempt < self.config.retry_count:
                    logger.warning(f"任务 {task} 第{attempt+1}次尝试失败，准备重试: {e}")
                    time.sleep(0.5 * (attempt + 1))  # 递增延迟
                    
        raise last_exception
    
    def _progress_monitor(self):
        """进度监控线程"""
        while self._completed_count + self._failed_count < self._total_count:
            time.sleep(self.config.log_interval)
            
            with self._lock:
                completed = self._completed_count
                failed = self._failed_count
                progress = ((completed + failed) / self._total_count) * 100
                elapsed = time.time() - self._start_time
                
            logger.info(f"📈 进度: {completed}/{self._total_count} ({progress:.1f}%), "
                       f"失败: {failed}, 耗时: {elapsed:.0f}秒")


class FactorCalculationExecutor(ConcurrentExecutor):
    """因子计算专用并发执行器"""
    
    def __init__(self, config: Optional[ConcurrentConfig] = None):
        # 因子计算通常CPU密集，适当减少并发数
        if config is None:
            config = ConcurrentConfig(
                max_workers=3,  # 因子计算CPU密集，不宜过多线程
                timeout=12000,   # 因子计算可能较耗时
                retry_count=2   # 增加重试次数
            )
        super().__init__(config)
    
    def execute_factor_batch(
        self,
        factor_names: List[str],
        snapshot_config_id: str,
        target_function: Callable = None
    ) -> Tuple[List[Any], List[Tuple[str, Exception]]]:
        """
        批量执行因子计算
        
        Args:
            factor_names: 因子名称列表
            snapshot_config_id: 快照配置ID
            target_function: 目标函数(默认为rolling_ic计算)
            
        Returns:
            (successful_results, failed_factors): 成功结果和失败因子
        """
        if target_function is None:
            from projects._03_factor_selection.factor_manager.ic_manager.rolling_ic_manager import \
                run_cal_and_save_rolling_ic_by_snapshot_config_id
            target_function = run_cal_and_save_rolling_ic_by_snapshot_config_id
        
        logger.info(f"🔬 开始批量因子计算: {len(factor_names)}个因子")
        logger.info(f"📋 快照配置ID: {snapshot_config_id}")
        
        # 包装执行函数
        def execute_single_factor(factor_name: str):
            logger.info(f"🧮 开始计算因子: {factor_name}")
            result = target_function(snapshot_config_id, [factor_name])
            logger.info(f"✅ 因子 {factor_name} 计算完成")
            return result
        
        return self.execute_batch(
            target_function=execute_single_factor,
            task_list=factor_names,
            task_name="因子IC计算"
        )
    
    def execute_chunked_factors(
        self,
        factor_names: List[str],
        snapshot_config_id: str,
        chunk_size: int = 3,
        target_function: Callable = None
    ) -> Tuple[List[Any], List[Tuple[List[str], Exception]]]:
        """
        分组并发执行因子计算(适合内存密集型任务)
        
        Args:
            factor_names: 因子名称列表
            snapshot_config_id: 快照配置ID
            chunk_size: 每组因子数量
            target_function: 目标函数
            
        Returns:
            (successful_results, failed_chunks): 成功结果和失败分组
        """
        if target_function is None:
            from projects._03_factor_selection.factor_manager.ic_manager.rolling_ic_manager import \
                run_cal_and_save_rolling_ic_by_snapshot_config_id
            target_function = run_cal_and_save_rolling_ic_by_snapshot_config_id
        
        # 分组
        factor_chunks = [
            factor_names[i:i + chunk_size]
            for i in range(0, len(factor_names), chunk_size)
        ]
        
        logger.info(f"📦 因子分组: {len(factor_chunks)}组，每组最多{chunk_size}个因子")
        
        # 包装执行函数
        def execute_factor_chunk(factor_chunk: List[str]):
            logger.info(f"🎯 开始计算因子组: {factor_chunk}")
            result = target_function(snapshot_config_id, factor_chunk)
            logger.info(f"✅ 因子组计算完成: {len(factor_chunk)}个因子")
            return result
        
        return self.execute_batch(
            target_function=execute_factor_chunk,
            task_list=factor_chunks,
            task_name="分组因子IC计算"
        )


# 便捷函数
def run_concurrent_factors(
    factor_names: List[str],
    snapshot_config_id: str,
    max_workers: int = 3,
    execution_mode: str = "single"  # "single" 或 "chunked"
) -> Tuple[List[Any], List[Tuple[Any, Exception]]]:
    """
    便捷的并发因子计算函数
    
    Args:
        factor_names: 因子名称列表
        snapshot_config_id: 快照配置ID
        max_workers: 最大并发数
        execution_mode: 执行模式 ("single": 单因子并发, "chunked": 分组并发)
        
    Returns:
        (successful_results, failed_tasks): 成功结果和失败任务
    """
    config = ConcurrentConfig(max_workers=max_workers)
    executor = FactorCalculationExecutor(config)
    
    if execution_mode == "single":
        return executor.execute_factor_batch(factor_names, snapshot_config_id)
    elif execution_mode == "chunked":
        return executor.execute_chunked_factors(factor_names, snapshot_config_id, chunk_size=3)
    else:
        raise ValueError(f"不支持的执行模式: {execution_mode}")


if __name__ == "__main__":
    # 示例用法
    import pandas as pd
    
    # 读取因子列表
    # df = pd.read_csv(r'D:\lqs\codeAbout\py\Quantitative\quant_research_portfolio\projects\_03_factor_selection\factor_manager\selector\o2o_v3.csv')
    # factor_names = df['factor_name'].unique().tolist()
    factor_names = ['vwap_deviation_20d','roe_change_q','roa_ttm','turnover_rate',  'large_trade_ratio_10d','beta','revenue_growth_ttm']
    #
    snapshot_config_id = '20250906_045625_05e460ab'
    
    # # 方式1: 单因子并发
    # logger.info("=== 单因子并发模式 ===")
    # successful, failed = run_concurrent_factors(
    #     factor_names=factor_names[:6],  # 测试前10个因子
    #     snapshot_config_id=snapshot_config_id,
    #     max_workers=6,
    #     execution_mode="single"
    # )
    
    # 方式2: 分组并发
    logger.info("=== 分组并发模式 ===")
    successful, failed = run_concurrent_factors(
        factor_names=factor_names,
        snapshot_config_id=snapshot_config_id,
        max_workers=3,
        execution_mode="chunked"
    )