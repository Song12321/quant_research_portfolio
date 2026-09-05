"""A 股日频数据侦察：按交易日找出应有股票，检查缺行和整行全空。"""

from datetime import datetime
from pathlib import Path

import pandas as pd

from quant_lib.config.constant_config import MARKET_DATA_ROOT, ROOT_DIR, get_market_data_path


# ===== 用户显式填写：填好日期后直接运行本文件 =====
START_DATE = ""
END_DATE = ""

DAILY_DATASETS = ("daily", "daily_hfq", "daily_basic", "stk_limit")
ISSUE_COLUMNS = ("dataset", "observation_date", "ts_code", "issue_type")


class DataReconManager:
    """只检查四张日频表，基准信息仅用于确定哪些股票当天应有数据。"""

    def __init__(self, data_root: Path = MARKET_DATA_ROOT):
        self.data_root = Path(data_root)

    def run(self, start_date: str, end_date: str, output_dir: Path) -> pd.DataFrame:
        # 日期只转换一次；留空或区间颠倒时停止，避免无意检查错误区间。
        start, end = pd.Timestamp(start_date), pd.Timestamp(end_date)
        if pd.isna(start) or pd.isna(end) or start > end:
            raise ValueError("必须填写有效的起止日期，且开始日期不得晚于结束日期")

        dates, stocks, suspend = self._load_baselines(start, end)
        issues = []
        counts = []
        for year in sorted(dates.year.unique()):
            # 四张表每年各读一次；日内查询用分组，避免每天重新读文件。
            yearly_data = self._load_year(year)
            for date in dates[dates.year == year]:
                expected_stocks = self._expected_stocks(stocks, suspend, date)
                for dataset, grouped in yearly_data.items():
                    # 当天完全没有行也是正常的检查分支，所有应有股票都会记为缺失。
                    daily = grouped.obj.iloc[:0]
                    if date in grouped.groups:
                        daily = grouped.get_group(date)
                    missing, empty = self._check_day(daily, expected_stocks)
                    day_text = date.strftime("%Y-%m-%d")
                    issues.extend((dataset, day_text, code, "missing_expected_row") for code in sorted(missing))
                    issues.extend((dataset, day_text, code, "empty_payload") for code in sorted(empty))
                    counts.append((dataset, len(expected_stocks), len(expected_stocks) - len(missing), len(missing), len(empty)))

        # 检查结束后统一保存；汇总直接从每日结果求和，无需维护多套共享计数器。
        details = pd.DataFrame(issues, columns=ISSUE_COLUMNS)
        summary = pd.DataFrame(counts, columns=[
            "dataset", "expected_rows", "observed_rows", "missing_rows", "invalid_rows",
        ]).groupby("dataset").sum().reindex(DAILY_DATASETS, fill_value=0).reset_index()
        summary["issue_records"] = summary["missing_rows"] + summary["invalid_rows"]
        summary["status"] = summary["issue_records"].map(lambda count: "PASS" if count == 0 else "FAIL")
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=False)
        details.to_csv(output_path / "issues.csv", index=False, encoding="utf-8")
        summary.to_csv(output_path / "summary.csv", index=False, encoding="utf-8")
        return summary

    def _load_baselines(self, start: pd.Timestamp, end: pd.Timestamp):
        # 文件不存在、必需列不存在或日期格式错误时，直接由读取/转换操作抛错。
        calendar = pd.read_parquet(get_market_data_path("trade_cal.parquet", self.data_root))
        stocks = pd.read_parquet(get_market_data_path("stock_basic.parquet", self.data_root))
        suspend = pd.read_parquet(get_market_data_path("suspend_d.parquet", self.data_root))
        calendar["cal_date"] = pd.to_datetime(calendar["cal_date"])
        stocks["list_date"] = pd.to_datetime(stocks["list_date"])
        stocks["delist_date"] = pd.to_datetime(stocks["delist_date"])
        suspend["trade_date"] = pd.to_datetime(suspend["trade_date"])

        # 基准日期缺失会直接改变“应有股票日”，不能把它当成没有上市或没有停牌。
        if calendar[["cal_date", "is_open"]].isna().any().any():
            raise ValueError("trade_cal 的 cal_date/is_open 不能为空")
        if stocks[["ts_code", "list_date"]].isna().any().any():
            raise ValueError("stock_basic 的 ts_code/list_date 不能为空")
        if suspend[["ts_code", "trade_date", "suspend_type"]].isna().any().any():
            raise ValueError("suspend_d 的 ts_code/trade_date/suspend_type 不能为空")
        if calendar["cal_date"].min() > start or calendar["cal_date"].max() < end or calendar.empty:
            raise ValueError("交易日历未覆盖指定的起止日期")

        mask = calendar["cal_date"].between(start, end) & calendar["is_open"].eq(1)
        dates = pd.DatetimeIndex(calendar.loc[mask, "cal_date"].unique()).sort_values()
        stocks = stocks.loc[stocks["ts_code"].str.fullmatch(r"\d{6}\.(SH|SZ|BJ)")]
        return dates, stocks, suspend.groupby("trade_date")

    def _expected_stocks(self, stocks: pd.DataFrame, suspend, date: pd.Timestamp) -> set: #ok
        # 上市日包含，退市日不包含；退市日期为空表示尚未退市。ST 不影响这里的要求。
        active = stocks["list_date"].le(date) & (
            stocks["delist_date"].isna() | stocks["delist_date"].gt(date)
        )
        expected = set(stocks.loc[active, "ts_code"])
        if date in suspend.groups:
            events = suspend.get_group(date)
            timing = events["suspend_timing"]
            # 判断每一行是否填写了日内停牌时间段，例如 "09:30-10:00"。
            # notna()：排除 None/NaN。
            # str.strip().ne("")：去掉首尾空格后，内容不能是空字符串。
            # 两个条件都满足，intraday 才为 True。
            intraday = timing.notna() & timing.str.strip().ne("")

            # 找出“停牌类型为 S，并且没有日内时间段”的股票代码。
            # ~intraday 表示取反，即“不是日内停牌记录”。
            # 这些股票暂时作为全日停牌候选；后面还要检查同日其他记录。
            full_day = set(
                events.loc[
                    events["suspend_type"].eq("S") & ~intraday,
                    "ts_code",
                ]
            )

            # 找出当天有复牌记录 R，或者有日内停牌时间段的股票。
            # 当前规则认为这些股票当天仍应有日线，不能获得全日停牌豁免。
            # 注意：resumed 这个名字不够准确，它也包含日内停牌股票，
            # 并不意味着这些股票都存在一条 R 记录。
            resumed = set(
                events.loc[
                    events["suspend_type"].eq("R") | intraday,
                    "ts_code",
                ]
            )

            # 集合相减：
            # full_day - resumed = 只有全日停牌记录、没有上述反向证据的股票。
            # 再从“当天应有行情的股票集合”中移除这些股票。
            #
            # 例如：
            # full_day = {A, B}
            # resumed  = {B, C}
            # 最终只移除 A；B 有同日复牌或日内停牌记录，仍要求行情。
            expected -= full_day - resumed
        return expected

    def _load_year(self, year: int) -> dict:
        result = {}
        for dataset in DAILY_DATASETS:
            # 只使用项目现有的年度分区路径，不添加备用路径或读取失败后的替代结果。
            path = get_market_data_path(dataset, self.data_root) / f"year={year}"
            frame = pd.read_parquet(path)
            frame["trade_date"] = pd.to_datetime(frame["trade_date"])
            if frame[["trade_date", "ts_code"]].isna().any().any():
                raise ValueError(f"{dataset}: trade_date/ts_code 不能为空")
            result[dataset] = frame.groupby("trade_date")
        return result

    def _check_day(self, daily: pd.DataFrame, expected: set) -> tuple[set, set]:
        # 只看应有股票；主键、分区列和索引残留不算业务数据。
        business_columns = daily.columns.difference([
            "ts_code", "trade_date", "year", "__index_level_0__",
        ])
        present = set(daily["ts_code"])
        has_data = set(daily.loc[daily[business_columns].notna().any(axis=1), "ts_code"])
        # 保留原来的简单口径：有任意业务值就通过；空壳行与完全缺行分别列出。
        return expected - present, (expected & present) - has_data


def main():
    # 只在入口读取顶部参数，核心函数通过显式参数调用。
    output_dir = ROOT_DIR / "results" / "data_recon" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    summary = DataReconManager().run(START_DATE, END_DATE, output_dir)
    print(summary.to_string(index=False))
    print(f"报告目录: {output_dir}")


if __name__ == "__main__":
    main()
