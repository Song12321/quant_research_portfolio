"""
配置快照管理器 - 测试配置的版本控制系统

核心功能：
1. 自动保存测试时的配置快照
2. 配置快照去重和版本管理
3. 测试结果与配置关联
4. 配置回溯和对比分析

设计理念：
- 每次测试后自动保存当前配置
- 通过哈希值避免重复存储
- 提供完整的配置追踪链路
- 支持配置差异对比
"""

import json
import hashlib
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import copy

from quant_lib.config.logger_config import setup_logger

logger = setup_logger(__name__)


@dataclass
class ConfigSnapshot:
    """配置快照数据结构"""
    snapshot_id: str                    # 快照唯一标识
    timestamp: str                      # 创建时间戳
    config_hash: str                    # 配置内容哈希
    snapshot_name: str                  # 快照名称/描述
    config_content: Dict[str, Any]      # 配置内容
    metadata: Dict[str, Any]            # 元数据
    

@dataclass 
class TestConfigReference:
    """测试结果的配置引用"""
    factor_name: str
    stock_pool: str
    calc_type: str
    version: str
    snapshot_id: str
    test_timestamp: str
    test_description: str = ""


class ConfigSnapshotManager:
    """配置快照管理器"""
    
    def __init__(self):
        # 动态定位workspace路径 - 修复硬编码路径问题
        self.workspace_root =Path(r'D:\lqs\codeAbout\py\Quantitative\import_file\quant_research_portfolio\workspace')
        self.snapshots_dir = self.workspace_root / "config_snapshots"
        self.snapshots_storage = self.snapshots_dir / "snapshots"
        self.index_file = self.snapshots_dir / "snapshot_index.json"
        
        # 创建目录结构
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self.snapshots_storage.mkdir(parents=True, exist_ok=True)
        
        # 需要保存的配置部分（排除factor_definition）
        self.tracked_config_sections = [
            'preprocessing',
            'evaluation', 
            'stock_pool_profiles',
            'research_window',
            'factor_selection',
            'factor_combination',
            'other_backtest',
            'target_factors_for_evaluation'  # 可能会变化
        ]
        
        # 加载现有索引
        self._index = self._load_index()
    
    def create_snapshot(
        self, 
        config: Dict[str, Any], 
        snapshot_name: str = "",
        test_context: Optional[Dict] = None
    ) -> str:
        """
        创建配置快照
        
        Args:
            config: 完整配置字典
            snapshot_name: 快照名称/描述
            test_context: 测试上下文信息
            
        Returns:
            str: 快照ID
        """
        logger.info(f"🔄 开始创建配置快照: {snapshot_name}")
        
        # 1. 提取需要跟踪的配置部分
        tracked_config = self._extract_tracked_config(config)
        
        # 2. 计算配置哈希
        config_hash = self._calculate_config_hash(tracked_config)
        
        # 3. 检查是否已存在相同配置
        existing_snapshot_id = self._find_existing_snapshot(config_hash)
        if existing_snapshot_id:
            logger.info(f"📥 发现相同配置的快照: {existing_snapshot_id}")
            return existing_snapshot_id
        
        # 4. 创建新快照
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        snapshot_id = f"{timestamp}_{config_hash[:8]}"
        
        # 5. 构建快照对象
        snapshot = ConfigSnapshot(
            snapshot_id=snapshot_id,
            timestamp=datetime.now().isoformat(),
            config_hash=config_hash,
            snapshot_name=snapshot_name or f"Config_{timestamp}",
            config_content=tracked_config,
            metadata={
                'created_at': datetime.now().isoformat(),
                'test_context': test_context or {},
                'config_sections': list(tracked_config.keys()),
                'total_size': len(json.dumps(tracked_config))
            }
        )
        
        # 6. 保存快照
        self._save_snapshot(snapshot)
        
        # 7. 更新索引
        self._update_index(snapshot)
        
        logger.info(f"✅ 配置快照创建完成: {snapshot_id}")
        return snapshot_id
    
    def link_test_result(
        self,
        snapshot_id: str,
        factor_name: str,
        stock_pool: str, 
        calc_type: str = 'o2o',
        version: str = '20190328_20231231',
        test_description: str = ""
    ) -> bool:
        """
        将测试结果与配置快照关联
        
        Args:
            snapshot_id: 快照ID
            factor_name: 因子名称
            stock_pool: 股票池
            calc_type: 计算类型
            version: 版本
            test_description: 测试描述
            
        Returns:
            bool: 关联成功与否
        """
        try:
            # 构建测试结果路径
            result_dir = self.workspace_root / "result" / stock_pool / factor_name / calc_type / version
            
            if not result_dir.exists():
                logger.warning(f"⚠️ 测试结果目录不存在: {result_dir}")
                return False
            
            # 创建配置引用
            config_ref = TestConfigReference(
                factor_name=factor_name,
                stock_pool=stock_pool,
                calc_type=calc_type,
                version=version,
                snapshot_id=snapshot_id,
                test_timestamp=datetime.now().isoformat(),
                test_description=test_description
            )
            
            # 保存配置引用
            ref_file = result_dir / "config_reference.json"
            with open(ref_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(config_ref), f, ensure_ascii=False, indent=2)
            
            logger.info(f"✅ 测试结果已关联配置快照: {factor_name} -> {snapshot_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ 关联配置快照失败: {e}")
            return False
    def get_snapshot_config_content_details(self, snapshot_id: str):
        config_content = self.get_snapshot(snapshot_id).config_content
        s = config_content['research_window']['start_date']
        e = config_content['research_window']['end_date']
        pool_index =list(config_content['stock_pool_profiles'].values())[0]['index_filter']['index_code']
        return pool_index,s,e ,config_content['evaluation']

    def get_snapshot(self, snapshot_id: str) -> Optional[ConfigSnapshot]:
        """获取配置快照"""
        try:
            snapshot_dir = self.snapshots_storage / snapshot_id
            snapshot_file = snapshot_dir / "config_snapshot.json"
            
            if not snapshot_file.exists():
                logger.warning(f"快照不存在: {snapshot_id}")
                return None
            
            with open(snapshot_file, 'r', encoding='utf-8') as f:
                snapshot_dict = json.load(f)
            
            return ConfigSnapshot(**snapshot_dict)
            
        except Exception as e:
            logger.error(f"加载快照失败 {snapshot_id}: {e}")
            return None
    
    def get_test_config(
        self,
        factor_name: str,
        stock_pool: str,
        calc_type: str = 'o2o',
        version: str = '20190328_20231231'
    ) -> Optional[Dict[str, Any]]:
        """获取测试结果对应的配置"""
        try:
            # 获取配置引用
            result_dir = self.workspace_root / "result" / stock_pool / factor_name / calc_type / version
            ref_file = result_dir / "config_reference.json"
            
            if not ref_file.exists():
                logger.warning(f"配置引用不存在: {ref_file}")
                return None
            
            with open(ref_file, 'r', encoding='utf-8') as f:
                config_ref = json.load(f)
            
            # 获取快照内容
            snapshot = self.get_snapshot(config_ref['snapshot_id'])
            if snapshot:
                return snapshot.config_content
            else:
                return None
                
        except Exception as e:
            logger.error(f"获取测试配置失败: {e}")
            return None
    
    def compare_configs(
        self, 
        snapshot_id1: str, 
        snapshot_id2: str
    ) -> Dict[str, Any]:
        """比较两个配置快照的差异"""
        snapshot1 = self.get_snapshot(snapshot_id1)
        snapshot2 = self.get_snapshot(snapshot_id2)
        
        if not snapshot1 or not snapshot2:
            return {"error": "无法获取快照"}
        
        differences = {}
        
        # 找出所有配置节
        all_sections = set(snapshot1.config_content.keys()) | set(snapshot2.config_content.keys())
        
        for section in all_sections:
            config1_section = snapshot1.config_content.get(section)
            config2_section = snapshot2.config_content.get(section)
            
            if config1_section != config2_section:
                differences[section] = {
                    'snapshot1': config1_section,
                    'snapshot2': config2_section
                }
        
        return {
            'snapshot1_id': snapshot_id1,
            'snapshot2_id': snapshot_id2,
            'differences': differences,
            'total_differences': len(differences)
        }
    
    def list_snapshots(
        self, 
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """列出配置快照"""
        snapshots = []
        
        for snapshot_id, metadata in self._index.items():
            # 时间过滤
            if start_date and metadata['timestamp'] < start_date:
                continue
            if end_date and metadata['timestamp'] > end_date:
                continue
            
            snapshots.append({
                'snapshot_id': snapshot_id,
                'snapshot_name': metadata['snapshot_name'],
                'timestamp': metadata['timestamp'],
                'config_hash': metadata['config_hash'],
                'config_sections': metadata.get('config_sections', [])
            })
        
        # 按时间倒序排列
        snapshots.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return snapshots[:limit]
    
    def cleanup_old_snapshots(self, keep_days: int = 90):
        """清理过期的配置快照"""
        from datetime import timedelta
        
        cutoff_date = datetime.now() - timedelta(days=keep_days)
        cutoff_str = cutoff_date.isoformat()
        
        removed_count = 0
        snapshots_to_remove = []
        
        for snapshot_id, metadata in self._index.items():
            if metadata['timestamp'] < cutoff_str:
                snapshots_to_remove.append(snapshot_id)
        
        for snapshot_id in snapshots_to_remove:
            try:
                # 删除快照文件
                snapshot_dir = self.snapshots_storage / snapshot_id
                if snapshot_dir.exists():
                    import shutil
                    shutil.rmtree(snapshot_dir)
                
                # 从索引中移除
                del self._index[snapshot_id]
                removed_count += 1
                
            except Exception as e:
                logger.error(f"删除快照失败 {snapshot_id}: {e}")
        
        # 保存更新后的索引
        self._save_index()
        
        logger.info(f"清理完成，删除 {removed_count} 个过期快照")
    
    def _extract_tracked_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """提取需要跟踪的配置部分"""
        tracked_config = {}
        
        for section in self.tracked_config_sections:
            if section in config:
                # 深拷贝避免修改原配置
                tracked_config[section] = copy.deepcopy(config[section])
        
        return tracked_config
    
    def _calculate_config_hash(self, config: Dict[str, Any]) -> str:
        """计算配置哈希值"""
        # 确保配置的序列化是确定性的
        config_str = json.dumps(config, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(config_str.encode('utf-8')).hexdigest()
    
    def _find_existing_snapshot(self, config_hash: str) -> Optional[str]:
        """查找现有的相同配置快照"""
        for snapshot_id, metadata in self._index.items():
            if metadata.get('config_hash') == config_hash:
                return snapshot_id
        return None
    
    def _save_snapshot(self, snapshot: ConfigSnapshot):
        """保存配置快照到文件"""
        snapshot_dir = self.snapshots_storage / snapshot.snapshot_id
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存快照内容
        snapshot_file = snapshot_dir / "config_snapshot.json"
        with open(snapshot_file, 'w', encoding='utf-8') as f:
            json.dump(asdict(snapshot), f, ensure_ascii=False, indent=2)
        
        # 保存元数据
        metadata_file = snapshot_dir / "metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(snapshot.metadata, f, ensure_ascii=False, indent=2)
        
        logger.debug(f"快照已保存: {snapshot_file}")
    
    def _update_index(self, snapshot: ConfigSnapshot):
        """更新快照索引"""
        self._index[snapshot.snapshot_id] = {
            'snapshot_id': snapshot.snapshot_id,
            'snapshot_name': snapshot.snapshot_name,
            'timestamp': snapshot.timestamp,
            'config_hash': snapshot.config_hash,
            'config_sections': list(snapshot.config_content.keys()),
            'metadata': snapshot.metadata
        }
        
        self._save_index()
    
    def _load_index(self) -> Dict[str, Any]:
        """加载快照索引"""
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"加载快照索引失败: {e}")
                return {}
        else:
            return {}
    
    def _save_index(self):
        """保存快照索引"""
        try:
            with open(self.index_file, 'w', encoding='utf-8') as f:
                json.dump(self._index, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存快照索引失败: {e}")
    
    def print_snapshot_summary(self, snapshot_id: str):
        """打印快照摘要"""
        snapshot = self.get_snapshot(snapshot_id)
        if not snapshot:
            print(f"❌ 快照不存在: {snapshot_id}")
            return
        
        print(f"\n{'='*60}")
        print(f"📸 配置快照摘要: {snapshot_id}")
        print(f"{'='*60}")
        print(f"🏷️  快照名称: {snapshot.snapshot_name}")
        print(f"⏰ 创建时间: {snapshot.timestamp}")
        print(f"🔍 配置哈希: {snapshot.config_hash[:16]}...")
        print(f"📊 配置节数: {len(snapshot.config_content)}")
        
        print(f"\n📋 包含的配置节:")
        for section in snapshot.config_content.keys():
            print(f"  ✓ {section}")
        
        print(f"\n💾 元数据:")
        for key, value in snapshot.metadata.items():
            if key != 'test_context':
                print(f"  {key}: {value}")
        
        print(f"{'='*60}")


def load_config_from_yaml(config_path: str) -> Dict[str, Any]:
    """从YAML文件加载配置"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)
#
#
# if __name__ == "__main__":
#     # 示例用法
#     config_path = r"/projects/_03_factor_selection/factory/config.yaml"
#
#     # 创建配置管理器
#     manager = ConfigSnapshotManager()
#
#     # 加载配置
#     config = load_config_from_yaml(config_path)
#
#     # 创建配置快照
#     snapshot_id = manager.create_snapshot(
#         config,
#         snapshot_name="测试动量因子配置V1.0",
#         test_context={
#             'test_type': 'single_factor',
#             'factors': ['momentum_120d', 'volatility_90d'],
#             'stock_pools': ['institutional_stock_pool']
#         }
#     )
#
#     # 关联测试结果
#     manager.link_test_result(
#         snapshot_id=snapshot_id,
#         factor_name="momentum_120d",
#         stock_pool="000300",
#         test_description="动量因子单因子测试"
#     )
#
#     # 查看快照
#     manager.print_snapshot_summary(snapshot_id)
#
#     # 列出所有快照
#     snapshots = manager.list_snapshots(limit=10)
#     print(f"\n📝 最近的配置快照:")
#     for snapshot in snapshots:
#         print(f"  {snapshot['snapshot_id']}: {snapshot['snapshot_name']}")





if __name__ == '__main__':
    factors_tested =  [
        "operating_accruals",
        "earnings_stability",
        "pead",
        "quality_momentum",
        "rsi",
        "cci",
        "log_total_mv",
        "log_circ_mv",
        "bm_ratio",
        "ep_ratio",
        "sp_ratio",
        "cfp_ratio",
        "roe_change_q",
        "roe_ttm",
        "gross_margin_ttm",
        "debt_to_assets",
        "net_profit_growth_yoy",
        "total_revenue_growth_yoy",
        "sharpe_momentum_60d",
        "momentum_12_1",
        "momentum_20d",
        "momentum_120d",
        "momentum_pct_60d",
        "volatility_120d",
        "volatility_90d",
        "volatility_40d",
        "reversal_21d",
        "reversal_5d",
        "turnover_rate_90d_mean",
        "ln_turnover_value_90d",
        "turnover_t1_div_t20d_avg",
        "turnover_rate_monthly_mean",
        "amihud_liquidity",
        "value_composite",
        "vwap_deviation_20d",
        "sw_l1_momentum_21d"
    ]
    manager = ConfigSnapshotManager()
    #20190328_20231231
    for name in factors_tested:
       manager.link_test_result(
            snapshot_id = '20250906_045625_05e460ab',
            factor_name=name,
            stock_pool="000906",
            test_description="o2o_v3"
        )
