import random
import pulp
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib

# QT is not working on my ARM workstation.
matplotlib.use("Agg")


def generate_random_availability(people, time_slots, num_available_times):
    """Generate random availability for each person."""
    return {person: random.sample(time_slots, num_available_times) for person in people}


def create_availability_dict(
    mentors, mentees, time_slots, mentor_availability, mentee_availability
):
    """Create availability dictionary indicating shared availability between mentors and mentees."""
    return {
        (m, e, t): int(t in mentor_availability[m] and t in mentee_availability[e])
        for m in mentors
        for e in mentees
        for t in time_slots
    }


def define_decision_variables(mentors, mentees, time_slots):
    """Define decision variables for the optimization model."""
    x = pulp.LpVariable.dicts(
        "x",
        ((m, e, t) for m in mentors for e in mentees for t in time_slots),
        cat="Binary",
    )
    y = pulp.LpVariable.dicts(
        "y", ((m, t) for m in mentors for t in time_slots), cat="Binary"
    )
    deviation = pulp.LpVariable.dicts("deviation", mentors, lowBound=0)
    return x, y, deviation


def add_constraints(
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
):
    """Add constraints to the optimization model."""
    # 1. Deviation constraints for group size balancing
    for m in mentors:
        total_mentees = pulp.lpSum(x[m, e, t] for e in mentees for t in time_slots)
        model += deviation[m] >= total_mentees - target_group_size
        model += deviation[m] >= target_group_size - total_mentees

    # 2. Availability constraint
    for (m, e, t), is_available in availability.items():
        if not is_available:
            model += x[m, e, t] == 0

    # 3. Single time slot per mentor
    for m in mentors:
        model += pulp.lpSum(y[m, t] for t in time_slots) == 1

    # 4. Mentors can only have a session if mentees are assigned
    for m in mentors:
        for t in time_slots:
            model += (
                pulp.lpSum(x[m, e, t] for e in mentees)
                <= max_mentees_per_group * y[m, t]
            )
            model += y[m, t] <= pulp.lpSum(x[m, e, t] for e in mentees)

    # 5. Each mentee is assigned to one mentor at a single time slot
    for e in mentees:
        model += pulp.lpSum(x[m, e, t] for m in mentors for t in time_slots) == 1

    # Auxiliary variables for absolute differences between mentor group sizes
    abs_diff = pulp.LpVariable.dicts(
        "abs_diff",
        [(m1, m2) for m1 in mentors for m2 in mentors if m1 != m2],
        lowBound=0,
    )

    # 6. Balance group sizes across mentors
    for m1 in mentors:
        for m2 in mentors:
            if m1 != m2:
                model += abs_diff[(m1, m2)] >= pulp.lpSum(
                    x[m1, e, t] for e in mentees for t in time_slots
                ) - pulp.lpSum(x[m2, e, t] for e in mentees for t in time_slots)
                model += abs_diff[(m1, m2)] >= pulp.lpSum(
                    x[m2, e, t] for e in mentees for t in time_slots
                ) - pulp.lpSum(x[m1, e, t] for e in mentees for t in time_slots)
                model += abs_diff[(m1, m2)] <= max_delta_group_size


def solve_and_display_schedule(model, mentors, mentees, time_slots, x):
    model.solve()
    matching = []
    if pulp.LpStatus[model.status] == "Optimal":
        schedule_data = []
        for t in time_slots:
            for m in mentors:
                mentees_assigned = [e for e in mentees if pulp.value(x[m, e, t]) == 1]
                if mentees_assigned:
                    schedule_data.append(
                        {
                            "Time Slot": t,
                            "Mentor": m,
                            "Mentees": ", ".join(mentees_assigned),
                        }
                    )
                    for e in mentees_assigned:
                        matching.append((m, e, t))

        schedule_df = pd.DataFrame(schedule_data)
        print("\nOptimal Schedule:\n")
        print(schedule_df)
        visualize_matching(matching)
    else:
        print("No optimal solution found.")


def visualize_matching(matching):
    """Visualize mentor-mentee matches with mentors on the left and mentees on the right, sorted and spaced to reduce overlap."""
    B = nx.Graph()

    matched_mentors = sorted({m for m, _, _ in matching})
    matched_mentees = sorted({e for _, e, _ in matching})
    matching_edges = [(m, e) for m, e, _ in matching]
    B.add_nodes_from(matched_mentors, bipartite=0)
    B.add_nodes_from(matched_mentees, bipartite=1)
    B.add_edges_from(matching_edges)

    mentor_x, mentee_x = -1, 1

    mentee_spacing = 1
    mentor_spacing = (
        mentee_spacing * (len(matched_mentees) / len(matched_mentors))
        if len(matched_mentors) > 1
        else 1
    )
    pos = {}

    for i, mentor in enumerate(matched_mentors):
        pos[mentor] = (
            mentor_x,
            i * mentor_spacing - (mentor_spacing * (len(matched_mentors) - 1)) / 2,
        )

    sorted_mentees = sorted(
        matched_mentees,
        key=lambda e: next(
            (i for i, m in enumerate(matched_mentors) if (m, e) in matching_edges),
            len(matched_mentors),
        ),
    )

    for i, mentee in enumerate(sorted_mentees):
        pos[mentee] = (
            mentee_x,
            i * mentee_spacing - (mentee_spacing * (len(sorted_mentees) - 1)) / 2,
        )

    plt.figure(figsize=(15, 10))
    nx.draw_networkx_nodes(
        B,
        pos,
        nodelist=matched_mentors,
        node_color="lightblue",
        label="Mentors",
        node_size=500,
    )
    nx.draw_networkx_nodes(
        B,
        pos,
        nodelist=sorted_mentees,
        node_color="lightgreen",
        label="Mentees",
        node_size=500,
    )
    nx.draw_networkx_edges(
        B, pos, edgelist=matching_edges, width=2.0, edge_color="blue"
    )

    edge_labels = {(m, e): t for m, e, t in matching}
    nx.draw_networkx_edge_labels(
        B, pos, edge_labels=edge_labels, font_color="red", label_pos=0.5
    )

    nx.draw_networkx_labels(B, pos, font_size=10)

    plt.legend(scatterpoints=1)
    plt.title("Mentor-Mentee Matching")
    plt.axis("off")
    plt.savefig("matching.png")


def main():
    # Define parameters
    mentors = [f"M{i}" for i in range(5)]
    mentees = [f"E{i}" for i in range(20)]
    time_slots = [f"T{i+1}" for i in range(16)]
    num_available_times = 10
    max_mentees_per_group = len(mentees)
    target_group_size = len(mentees) / len(mentors)
    max_delta_group_size = 1

    mentor_availability = generate_random_availability(
        mentors, time_slots, num_available_times
    )
    mentee_availability = generate_random_availability(
        mentees, time_slots, num_available_times
    )
    availability = create_availability_dict(
        mentors, mentees, time_slots, mentor_availability, mentee_availability
    )

    model = pulp.LpProblem("Mentor_Mentee_Scheduling", pulp.LpMinimize)
    x, y, deviation = define_decision_variables(mentors, mentees, time_slots)

    # Objective function: Minimize deviations from the target group size
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

    solve_and_display_schedule(model, mentors, mentees, time_slots, x)


if __name__ == "__main__":
    main()
