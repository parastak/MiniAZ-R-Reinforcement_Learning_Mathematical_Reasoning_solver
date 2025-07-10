#agent/proposer.py
"""
this proposer agent mainly produce new equations
"""

#importing all necessary modules and libraries
import random


def generate_equation(difficulty: str = "easy") -> str:
    """
    Generates an algebraic equation based on difficulty level.
    Easy: x + b = c
    Medium: a*x + b = c
    Hard: x^2 + bx + c = 0
    """

    if difficulty == "easy":
        x = random.randint(1, 10)
        b = random.randint(1, 10)
        c = x + b
        return f" x + {b} = {c}"

    elif difficulty == "medium":
        x = random.randint(1, 10)
        a = random.randint(1, 5)
        b = random.randint(1, 10)
        c = a * x + b
        return f"{a}*x + {b} = {c}"

    elif difficulty == "hard":
        r1 = random.randint(1, 5)
        r2 = random.randint(1, 5)
        a = 1
        b = -(r1 + r2)
        c = r1 * r2
        return f"x**2 + ({b})*x + ({c}) = 0"

    else:
        raise ValueError("Unknown difficulty level: " + difficulty)
