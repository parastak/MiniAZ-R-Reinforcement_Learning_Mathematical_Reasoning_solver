#tasks/deduction.py
"""
This module generates deduction-style math problems for self-play.
A deduction task is a reasoning problem where the agent starts from a set of premises (e.g., a math problem)
 and tries to arrive at a valid conclusion (e.g., the solution), using logical or mathematical rules.

In deduction tasks, the agent is given a complete equation (e.g., "2*x + 3 = 7")
and must solve for the unknown variable (typically 'x').
"""


# All necessary libraries to import here
import random
import logging
from typing import List
from core.environment import solve_equation
from utils import Task, task_to_log_line


# Generating a solvable linear equation for agent
def generate_linear_equation_deduction() -> str:
    """this function generate random linear equation with one variable as deduction task for agent to find the
    value x in the given equation"""
    a = random.randint(1, 10)
    b = random.randint(1, 10)
    x = random.randint(1, 10)
    y = a * x + b
    return f"{a} * x + {b} = {y}"


# generating a solvable quadratic equation for agent
def generate_quadratic_equation_deduction() -> str:
    """this function generate random quadratic equation with single variable as deduction task for agent to find the
    values of x in the given equation """
    # Choosing random roots r1 and r2
    r1 = random.randint(1, 5)
    r2 = random.randint(1, 5)
    a = 1  # Keeping leading coefficient simple for starting phase
    b = -(r1 + r2)
    c = r1 * r2
    return f"x**2 + ({b})*x + ({c}) = 0"


# creating deduction task and return it as a Task object
def create_deduction_task(task_id: int, difficulty: str = "easy") -> Task:
    """
    this function Create a single deduction task and return it as a Task object.
    """
    if difficulty == "easy":
        eq = generate_linear_equation_deduction()
    else:
        eq = generate_quadratic_equation_deduction()

    # now let put this eq into solve_equation for ground truth
    solution = solve_equation(eq)

    if isinstance(solution, dict) and "error" in solution:
        logging.warning(f"Invalid task generated: {eq} | Error: {solution['error']}")
        # Optionally: regenerate instead of returning
        return create_deduction_task(task_id, difficulty)

    task = Task(
        task_id=task_id,
        equation_str=eq,
        reasoning_type="deduction",
        ground_truth_solution=solution,
    )

    logging.info(task_to_log_line(task))
    return task


# Main block for testing environment functions
if __name__ == "__main__":
    from utils import setup_logger
    setup_logger()

    for i in range(5):
        create_deduction_task(task_id=i, difficulty='easy')
        create_deduction_task(task_id=i+100, difficulty='hard')
