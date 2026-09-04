import numpy as np
import pandas as pd
from bisect import bisect_right

from quant_lib.config.constant_config import get_market_data_path
from quant_lib.config.constant_config import permanent__day


class PointInTimeIndustryMap:
    """
    一个高效的、支持即时查询(Point-in-Time)的行业地图管理器。
    """

    def __init__(self,raw_industry_df=None):
        """
        通过原始的、包含in_date和out_date的成员关系DataFrame进行初始化。
        这个过程会进行一次性预处理，构建高效的查询结构。
        """
        print("正在预处理历史行业数据，构建Point-in-Time地图...")
        if raw_industry_df is None:
            self._raw_data = get_industry_record_df_processed()
        else:
            self._raw_data = raw_industry_df

        # 1. 获取所有行业变动的“事件日”
        event_dates = pd.unique(np.concatenate([
            self._raw_data['in_date'],
            self._raw_data['out_date'] + pd.Timedelta(days=1)  # out_date当天失效，第二天变更 （理解：我们要的是状态变更生效的哪一天！
        ])).astype('datetime64[ns]')
        #... 快照A ... [2023-11-15] ... 快照B ... [2023-12-29] ... 快照C ... [2024-02-10] ... 快照D ... [2024-03-16] ... 快照E ...
        # 核心思想就是 快照B的start end 都来自于某天某只股票的事件（生效or剔除） 在整个快照period，可以理解为 这整个时期 所有行业都是稳定未变化的！
        #下面 遍历每个事件行动日！
        ###比如遍历到快照B的start日，20231115
        #### 注意期间的不需要遍历啊，这就是此设计的唯一的性能亮点 （为什么可以做到不需要遍历：见上面说的核心思想
        ##然后遍历快照B的end日， 20231229
        self._event_dates = sorted([d for d in event_dates if d < pd.Timestamp(permanent__day)])

        # 2. 为每个事件日生成一个行业地图快照
        self._maps_on_event_dates = {}
        for date in self._event_dates:
            # 筛选出在 `date` 当天有效的成员关系
            current_map_df = self._raw_data[
                (self._raw_data['in_date'] <= date) &
                (self._raw_data['out_date'] >= date)
                ]
            # 只保留需要的列，并设置索引
            self._maps_on_event_dates[date] = current_map_df[['ts_code', 'l1_code', 'l2_code']].set_index('ts_code')

        print(f"预处理完成！共生成 {len(self._event_dates)} 个历史快照。")
    #ok
    def get_map_for_date(self, query_date: pd.Timestamp) -> pd.DataFrame:
        """
        高效获取指定日期的行业地图。
        :param query_date: 需要查询的日期
        :return: 一个以ts_code为索引的DataFrame，包含l1_code和l2_code
        """
        # 使用二分查找找到正确的事件日索引
        # bisect_right 会找到 query_date 应该插入的位置
        # 它之前的那个事件日，就是我们需要的快照日期
        idx = bisect_right(self._event_dates, query_date)
        #_event_dates:快照起始日：  快照起始日0101 快照起始日0104 . 快照起始日（0110）
        #query_date（0110）
        #return idx:3 这个索引上的快照起始日，是query_Date无法触摸的坎 so后面-1

        if idx == 0:
            # 如果查询日期比最早的事件日还早
            raise ValueError("查询日期比最早的事件日还早 肯定有问题！") #return  pd.DataFrame(columns=['l1_code', 'l2_code'])

        # 获取对应的历史快照
        target_event_date = self._event_dates[idx - 1]
        return self._maps_on_event_dates[target_event_date]

def get_industry_record_df_processed():
    df = pd.read_parquet(get_market_data_path('industry_record.parquet'))
    df['in_date'] = pd.to_datetime(df['in_date'])
    df['out_date'] = pd.to_datetime(df['out_date'])

    # 按 ts_code + in_date + out_date 排序，NaT 默认在最后
    df = df.sort_values(by=['ts_code', 'in_date', 'out_date'])


    # 分组处理
    def resolve_timeline_conflicts(group: pd.DataFrame) -> pd.DataFrame:
        """
        为一个股票分组，解决时间线冲突（重叠或NaT），但尊重空窗期。
        """
        group = group.sort_values(by='in_date').reset_index(drop=True)

        if len(group) <= 1:
            if not group.empty and pd.isna(group.loc[0, 'out_date']):
                group.loc[0, 'out_date'] = permanent__day
            return group

        for i in range(len(group) - 1):
            current_out = group.loc[i, 'out_date']
            next_in_date = group.loc[i + 1, 'in_date']  # 正确的写法

            # 情况一：当前 out_date 为空，必须填充
            if pd.isna(current_out):
                group.loc[i, 'out_date'] = next_in_date - pd.Timedelta(days=1)
            # 情况二：当前 out_date 不为空，但与下一个 in_date 构成了重叠
            elif current_out >= next_in_date:
                # 强制修正，确保没有重叠
                group.loc[i, 'out_date'] = next_in_date - pd.Timedelta(days=1)
            # 情况三：current_out < next_in_date，是合理的空窗期，不做任何事

        # 单独处理最后一条记录的 out_date
        if pd.isna(group.loc[len(group) - 1, 'out_date']):
            group.loc[len(group) - 1, 'out_date'] = permanent__day

        return group

    df = df.groupby('ts_code').apply(resolve_timeline_conflicts)
    df=df.drop_duplicates(subset=['ts_code', 'in_date', 'out_date'], keep='first', inplace=False)

    return df.reset_index(drop=True)
def _simple_insdustry_record():
    ret = get_industry_record_df_processed()
    mask = (ret['in_date'] >= pd.to_datetime('2022-01-01') )&  (ret['out_date'] <= pd.to_datetime('2024-12-31'))
    ret = ret[mask]
    ret = ret[ret['ts_code'].isin(['301217.SZ','688592.SH'])]
    ret = ret.sort_values(by=['ts_code', 'in_date', 'out_date'])

    return ret
