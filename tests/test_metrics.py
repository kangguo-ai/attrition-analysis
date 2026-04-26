import pandas as pd
import pytest
from src.metrics import (
    attrition_rate,
    attrition_by_department,
    attrition_by_overtime,
    average_income_by_attrition,
    satisfaction_summary,
)


@pytest.fixture
def sample_df():
    # Sales: 2 leavers / 2 total = 100%
    # HR:    1 leaver  / 2 total = 50%
    # Eng:   0 leavers / 2 total = 0%
    # Overtime Yes: 2/3 = 66.67%,  No: 1/3 = 33.33%
    # Satisfaction 1: 2/2 = 100%,  2: 1/2 = 50%,  3: 0/2 = 0%
    # Income leavers avg: (4000+6000+5000)/3 = 5000.0
    # Income stayers avg: (7000+8000+10000)/3 = 8333.33
    return pd.DataFrame(
        {
            "employee_id": [1, 2, 3, 4, 5, 6],
            "department": ["Sales", "Sales", "HR", "HR", "Engineering", "Engineering"],
            "monthly_income": [4000, 6000, 5000, 7000, 8000, 10000],
            "job_satisfaction": [1, 1, 2, 2, 3, 3],
            "overtime": ["Yes", "Yes", "No", "No", "Yes", "No"],
            "attrition": ["Yes", "Yes", "Yes", "No", "No", "No"],
        }
    )


# ---------------------------------------------------------------------------
# attrition_rate
# ---------------------------------------------------------------------------


def test_attrition_rate_returns_expected_percent():
    df = pd.DataFrame(
        {
            "employee_id": [1, 2, 3, 4],
            "department": ["Sales", "Sales", "HR", "HR"],
            "attrition": ["Yes", "No", "No", "Yes"],
        }
    )
    assert attrition_rate(df) == 50.0


def test_attrition_rate_zero_attrition():
    df = pd.DataFrame({"employee_id": [1, 2], "attrition": ["No", "No"]})
    assert attrition_rate(df) == 0.0


def test_attrition_rate_full_attrition():
    df = pd.DataFrame({"employee_id": [1, 2], "attrition": ["Yes", "Yes"]})
    assert attrition_rate(df) == 100.0


# ---------------------------------------------------------------------------
# attrition_by_department
# ---------------------------------------------------------------------------


def test_attrition_by_department_returns_expected_columns():
    df = pd.DataFrame(
        {
            "employee_id": [1, 2, 3, 4],
            "department": ["Sales", "Sales", "HR", "HR"],
            "attrition": ["Yes", "No", "No", "Yes"],
        }
    )
    result = attrition_by_department(df)
    assert list(result.columns) == ["department", "employees", "leavers", "attrition_rate"]


def test_attrition_by_department_correct_rates(sample_df):
    result = attrition_by_department(sample_df)
    rates = result.set_index("department")["attrition_rate"]
    assert rates["Sales"] == 100.0
    assert rates["HR"] == 50.0
    assert rates["Engineering"] == 0.0


def test_attrition_by_department_sorted_descending(sample_df):
    result = attrition_by_department(sample_df)
    rates = result["attrition_rate"].tolist()
    assert rates == sorted(rates, reverse=True)


# ---------------------------------------------------------------------------
# attrition_by_overtime
# ---------------------------------------------------------------------------


def test_attrition_by_overtime_returns_expected_columns(sample_df):
    result = attrition_by_overtime(sample_df)
    assert list(result.columns) == ["overtime", "employees", "leavers", "attrition_rate"]


def test_attrition_by_overtime_correct_rates(sample_df):
    result = attrition_by_overtime(sample_df)
    rates = result.set_index("overtime")["attrition_rate"]
    assert rates["Yes"] == round(2 / 3 * 100, 2)
    assert rates["No"] == round(1 / 3 * 100, 2)


# ---------------------------------------------------------------------------
# average_income_by_attrition
# ---------------------------------------------------------------------------


def test_average_income_by_attrition_returns_expected_columns(sample_df):
    result = average_income_by_attrition(sample_df)
    assert list(result.columns) == ["attrition", "avg_monthly_income"]


def test_average_income_by_attrition_correct_values(sample_df):
    result = average_income_by_attrition(sample_df)
    avgs = result.set_index("attrition")["avg_monthly_income"]
    assert avgs["Yes"] == round((4000 + 6000 + 5000) / 3, 2)
    assert avgs["No"] == round((7000 + 8000 + 10000) / 3, 2)


# ---------------------------------------------------------------------------
# satisfaction_summary
# ---------------------------------------------------------------------------


def test_satisfaction_summary_returns_expected_columns(sample_df):
    result = satisfaction_summary(sample_df)
    assert list(result.columns) == ["job_satisfaction", "total_employees", "leavers", "attrition_rate"]


def test_satisfaction_summary_correct_rates(sample_df):
    # Rates are per-group headcount, not share of total leavers.
    result = satisfaction_summary(sample_df)
    rates = result.set_index("job_satisfaction")["attrition_rate"]
    assert rates[1] == 100.0  # 2 leavers / 2 employees
    assert rates[2] == 50.0   # 1 leaver  / 2 employees
    assert rates[3] == 0.0    # 0 leavers / 2 employees


def test_satisfaction_summary_sorted_ascending(sample_df):
    result = satisfaction_summary(sample_df)
    levels = result["job_satisfaction"].tolist()
    assert levels == sorted(levels)
