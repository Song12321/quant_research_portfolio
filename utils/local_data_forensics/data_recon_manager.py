"""A 股日频数据侦察：以 daily 成交额为依据，检查另外三表的一致性。"""

from datetime import datetime
from pathlib import Path

import pandas as pd

from quant_lib.config.constant_config import MARKET_DATA_ROOT, ROOT_DIR, get_market_data_path




DAILY_DATASETS = ("daily", "daily_hfq", "daily_basic", "stk_limit")
ISSUE_COLUMNS = ("dataset", "observation_date", "ts_code", "issue_type")


class DataReconManager:
    """daily 提供成交证据；无成交证据仅标记未知，不推断停牌。"""

    def __init__(self, data_root: Path = MARKET_DATA_ROOT):
        self.data_root = Path(data_root)

    def run(self, start_date: str, end_date: str, output_dir: Path) -> pd.DataFrame:
        # 日期只转换一次；留空或区间颠倒时停止，避免无意检查错误区间。
        start, end = pd.Timestamp(start_date), pd.Timestamp(end_date)
        if pd.isna(start) or pd.isna(end) or start > end:
            raise ValueError("必须填写有效的起止日期，且开始日期不得晚于结束日期")

        dates, stocks = self._load_baselines(start, end)
        issues = []
        counts = []
        for year in sorted(dates.year.unique()):
            # 四张表每年各读一次；日内查询用分组，避免每天重新读文件。
            yearly_data = self._load_year(year)
            for date in dates[dates.year == year]:
                frames = {
                    dataset: grouped.get_group(date) if date in grouped.groups else grouped.obj.iloc[:0]
                    for dataset, grouped in yearly_data.items()
                }
                daily = frames["daily"]
                # 成交证据独立于上市日期；沪深范围保持不变。
                traded = set(daily.loc[
                    daily["ts_code"].str.fullmatch(r"\d{6}\.(SH|SZ)") & daily["amount"].gt(0),
                    "ts_code",
                ])
                active = self._active_stocks(stocks, date)
                unknown = active - traded
                day_text = date.strftime("%Y-%m-%d")
                issues.extend(("daily", day_text, code, "trading_status_unknown") for code in sorted(unknown))
                counts.append(("daily", len(active | traded), len(traded), 0, 0, len(unknown)))
                for dataset in DAILY_DATASETS[1:]:
                    missing, empty = self._check_day(frames[dataset], traded)
                    issues.extend((dataset, day_text, code, "missing_expected_row") for code in sorted(missing))
                    issues.extend((dataset, day_text, code, "empty_payload") for code in sorted(empty))
                    counts.append((dataset, len(traded), len(traded) - len(missing), len(missing), len(empty), 0))

        # 检查结束后统一保存；汇总直接从每日结果求和，无需维护多套共享计数器。
        details = pd.DataFrame(issues, columns=ISSUE_COLUMNS)
        summary = pd.DataFrame(counts, columns=[
            "dataset", "expected_rows", "observed_rows", "missing_rows", "invalid_rows", "unknown_rows",
        ]).groupby("dataset").sum().reindex(DAILY_DATASETS, fill_value=0).reset_index()
        summary["issue_records"] = summary["missing_rows"] + summary["invalid_rows"] + summary["unknown_rows"]
        summary["status"] = "PASS"
        summary.loc[summary["unknown_rows"].gt(0), "status"] = "UNKNOWN"
        summary.loc[(summary["missing_rows"] + summary["invalid_rows"]).gt(0), "status"] = "FAIL"
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=False)
        details.to_csv(output_path / "issues.csv", index=False, encoding="utf-8")
        summary.to_csv(output_path / "summary.csv", index=False, encoding="utf-8")
        return summary

    def _load_baselines(self, start: pd.Timestamp, end: pd.Timestamp):
        # 文件不存在、必需列不存在或日期格式错误时，直接由读取/转换操作抛错。
        calendar = pd.read_parquet(get_market_data_path("trade_cal.parquet", self.data_root))
        stocks = pd.read_parquet(get_market_data_path("stock_basic.parquet", self.data_root))
        calendar["cal_date"] = pd.to_datetime(calendar["cal_date"])
        stocks["list_date"] = pd.to_datetime(stocks["list_date"])
        stocks["delist_date"] = pd.to_datetime(stocks["delist_date"])

        # 基准日期缺失会直接改变“应有股票日”，不能把它当成没有上市或没有停牌。
        if calendar[["cal_date", "is_open"]].isna().any().any():
            raise ValueError("trade_cal 的 cal_date/is_open 不能为空")
        if stocks[["ts_code", "list_date"]].isna().any().any():
            raise ValueError("stock_basic 的 ts_code/list_date 不能为空")
        if calendar["cal_date"].min() > start or calendar["cal_date"].max() < end or calendar.empty:
            raise ValueError("交易日历未覆盖指定的起止日期")

        mask = calendar["cal_date"].between(start, end) & calendar["is_open"].eq(1)
        dates = pd.DatetimeIndex(calendar.loc[mask, "cal_date"].unique()).sort_values()
        # 当前只侦察沪深股票，北交所股票不进入应有数据集合。
        stocks = stocks.loc[stocks["ts_code"].str.fullmatch(r"\d{6}\.(SH|SZ)")]
        return dates, stocks

    def _active_stocks(self, stocks: pd.DataFrame, date: pd.Timestamp) -> set:
        # 上市日包含，退市日不包含；仅用于界定需要标记未知的股票。
        active = stocks["list_date"].le(date) & (
            stocks["delist_date"].isna() | stocks["delist_date"].gt(date)
        )
        return set(stocks.loc[active, "ts_code"])

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
    # ===== 用户显式填写：填好日期后直接运行本文件 =====
    START_DATE = "20220401"
    END_DATE = "20220501"
    summary = DataReconManager().run(START_DATE, END_DATE, output_dir)
    print(summary.to_string(index=False))
    print(f"报告目录: {output_dir}")


if __name__ == "__main__":
    main()
