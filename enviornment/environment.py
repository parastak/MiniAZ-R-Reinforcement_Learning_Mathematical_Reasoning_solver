# core/environment.py
"""
This module serves as the Ground Truth Oracle for our mathematical environment.
It does not define the state space for an agent, but rather provides two critical services:
1.  Verifies if a proposed mathematical task is valid and solvable.
2.  Provides the correct "ground truth" solution for a given task, which is used
    to generate the reward signal for the Solver agent.
"""

# All necessary libraries to import here
from sympy import symbols, solve, SympifyError
from sympy.parsing.sympy_parser import parse_expr


def solve_equation(equation_str: str, var: str = 'x'):
    """
    Solves a given algebraic equation for a specified variable.

    This function can handle linear, quadratic, and other polynomial equations
    that are solvable by SymPy.

    :param equation_str: A string representing the equation (e.g., "2*x + 3 = 7").
    :param var: A string representing the variable to solve for (e.g., 'x').
    :return: A list of numerical solutions, or a dictionary with an error key if failed.
    """
    try:
        # Define the variable/symbol
        variable = symbols(var)

        # Split the equation into left and right sides
        left_side, right_side = equation_str.split('=')

        # Create a SymPy expression representing 'left_side - right_side = 0' parse_expr is safer and more flexible than eval()
        expression = parse_expr(left_side.strip(), local_dict={var: variable}) - \
                     parse_expr(right_side.strip(), local_dict={var: variable})

        # Solve the expression for the variable
        solutions = solve(expression, variable)
        return [float(s) for s in solutions]

    except (ValueError, SympifyError) as e:
        # Catches errors from bad equation formats (e.g., no '=') or from SymPy failing to parse the expression.
        return {"error": f"Invalid equation format or syntax: {str(e)}"}
    except Exception as e:
        # Catches other unexpected errors.
        return {"error": f"An unexpected error occurred: {str(e)}"}


# Main block for testing environment functions
if __name__ == "__main__":
    # --- Deduction Task Examples ---
    # The environment is given a full equation and provides the solution.
    print(f"Solving '2*x + 3 = 7': {solve_equation('2*x + 3 = 7')}")
    print(f"Solving '20*x + 30 = 910': {solve_equation('20*x + 30 = 910')}")
    print(f"Solving 'x**2 - 5*x + 6 = 0': {solve_equation('x**2 - 5*x + 6 = 0')}")
    # Note: SymPy can handle this, but the float conversion might be tricky.
    # For now, this is fine. We can refine later if needed.
    print(f"Solving 'x**2 + x + 1 = 0': {solve_equation('x**2 + x + 1 = 0')}")
    # Example of an invalid equation
    print(f"Solving '5x bananas': {solve_equation('5x bananas')}")
