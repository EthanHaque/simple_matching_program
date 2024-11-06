import pytest
import pulp
from matching.matching import (
    create_availability_dict,
    define_decision_variables,
    add_constraints,
)


@pytest.fixture
def basic_test_data():
    mentors = ["M1", "M2"]
    mentees = ["E1", "E2"]
    time_slots = ["T1", "T2"]
    target_group_size = len(mentees) / len(mentors)
    max_mentees_per_group = 1
    max_delta_group_size = 0

    mentor_availability = {"M1": ["T1"], "M2": ["T2"]}
    mentee_availability = {"E1": ["T1"], "E2": ["T2"]}

    availability = create_availability_dict(
        mentors, mentees, time_slots, mentor_availability, mentee_availability
    )

    return (
        mentors,
        mentees,
        time_slots,
        availability,
        target_group_size,
        max_mentees_per_group,
        max_delta_group_size,
    )


def test_optimal_schedule_single_solution(basic_test_data):
    (
        mentors,
        mentees,
        time_slots,
        availability,
        target_group_size,
        max_mentees_per_group,
        max_delta_group_size,
    ) = basic_test_data

    model = pulp.LpProblem("Mentor_Mentee_Scheduling", pulp.LpMinimize)
    x, y, deviation = define_decision_variables(mentors, mentees, time_slots)

    model += pulp.lpSum(deviation[m] for m in mentors)

    add_constraints(
        model,
        mentors,
        mentees,
        time_slots,
        x,
        y,
        deviation,
        availability,
        max_mentees_per_group,
        target_group_size,
        max_delta_group_size,
    )

    model.solve()
    assert pulp.LpStatus[model.status] == "Optimal"

    assert pulp.value(x["M1", "E1", "T1"]) == 1
    assert pulp.value(x["M2", "E2", "T2"]) == 1
    assert pulp.value(x["M1", "E2", "T2"]) == 0
    assert pulp.value(x["M2", "E1", "T1"]) == 0

    assert pulp.value(y["M1", "T1"]) == 1
    assert pulp.value(y["M2", "T2"]) == 1
    assert pulp.value(y["M1", "T2"]) == 0
    assert pulp.value(y["M2", "T1"]) == 0


def test_balanced_groups(basic_test_data):
    (
        mentors,
        mentees,
        time_slots,
        availability,
        target_group_size,
        max_mentees_per_group,
        max_delta_group_size,
    ) = basic_test_data

    model = pulp.LpProblem("Mentor_Mentee_Scheduling", pulp.LpMinimize)
    x, y, deviation = define_decision_variables(mentors, mentees, time_slots)

    model += pulp.lpSum(deviation[m] for m in mentors)

    add_constraints(
        model,
        mentors,
        mentees,
        time_slots,
        x,
        y,
        deviation,
        availability,
        max_mentees_per_group,
        target_group_size,
        max_delta_group_size,
    )

    model.solve()
    assert pulp.LpStatus[model.status] == "Optimal"

    for m in mentors:
        assigned_mentees = sum(
            pulp.value(x[m, e, t]) for e in mentees for t in time_slots
        )
        assert abs(assigned_mentees - target_group_size) <= max_delta_group_size
