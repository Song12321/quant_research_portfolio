import pandas as pd
import tushare as ts
from tushare.pro.client import DataApi


TOKEN = "75daea70a87e1c5a50c2344440fb7a69685986e25ba8a592e08684ff"
TS_CODE = "600023.SH"


def _revision_example(data: pd.DataFrame) -> str:
    """返回同一报告键下 update_flag=0/1 的首个数值差异。"""
    if "update_flag" not in data.columns:
        return ""

    key_columns = [
        column
        for column in [
            "ts_code",
            "end_date",
            "ann_date",
            "report_type",
            "comp_type",
            "end_type",
        ]
        if column in data.columns
    ]
    value_columns = [
        column for column in data.columns
        if column not in key_columns + ["f_ann_date", "update_flag"]
    ]

    for _, group in data.groupby(key_columns, dropna=False):
        flags = group["update_flag"].astype(str)
        if not {"0", "1"}.issubset(set(flags)):
            continue
        original = group.loc[flags == "0"].iloc[0]
        updated = group.loc[flags == "1"].iloc[0]
        for column in value_columns:
            if pd.isna(original[column]) and pd.isna(updated[column]):
                continue
            if original[column] != updated[column]:
                return (
                    f"key={tuple(original[column_name] for column_name in key_columns)}, "
                    f"f_ann_date0={original.get('f_ann_date')!r}, "
                    f"f_ann_date1={updated.get('f_ann_date')!r}, "
                    f"field={column}, flag0={original[column]!r}, flag1={updated[column]!r}"
                )
    return "none"


def inspect_report_api(pro, api_name: str) -> None:
    print(f"\n===== {api_name}: report_type 1..12 =====")
    for report_type in range(1, 13):
        data = getattr(pro, api_name)(
            ts_code=TS_CODE,
            report_type=str(report_type),
            fields="",
        )
        if data.empty:
            print(f"report_type={report_type}: empty")
            continue

        flags = data["update_flag"].astype(str).value_counts().to_dict()
        date_differences = int(
            (data["ann_date"].astype(str) != data["f_ann_date"].astype(str)).sum()
        )
        print(
            f"report_type={report_type}: rows={len(data)}, "
            f"end_dates={data['end_date'].nunique()}, flags={flags}, "
            f"ann_vs_fann_diff={date_differences}, "
            f"revision_example={_revision_example(data)}"
        )


def main() -> None:
    pro = ts.pro_api(TOKEN)

    for api_name in ("income", "balancesheet", "cashflow"):
        inspect_report_api(pro, api_name)

    print("\n===== fina_indicator =====")
    data = pro.fina_indicator(ts_code=TS_CODE, fields="")
    flags = data["update_flag"].astype(str).value_counts().to_dict()
    print(
        f"rows={len(data)}, end_dates={data['end_date'].nunique()}, "
        f"flags={flags}, revision_example={_revision_example(data)}"
    )


def temp():
    pro = ts.pro_api(TOKEN)


    df = pro.suspend_d(ts_code='000670.SZ')

if __name__ == "__main__":
    temp()
