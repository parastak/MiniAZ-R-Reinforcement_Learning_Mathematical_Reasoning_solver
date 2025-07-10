#agent/learned_solver.py

"""
This module contains the logic for a solver that uses a trained
neural network policy to predict solutions.
"""



# importing all necessary files and modules
import torch
from sympy import symbols
from sympy.parsing.sympy_parser import parse_expr


def extract_cofficients(equation: str, var: str = 'x') -> list[float]:
    """
        Parses a linear equation of the form 'ax + b = c' and returns [a, b, c].
        This implementation is robust and handles many formats.
    """
    try:
        x = symbols(var)
        left, right = equation.split("=")

        # parse both side of the equation
        left_expr = parse_expr(left.strip(), local_dict={var: x})
        right_expr = parse_expr(right.strip(), local_dict={var: x})

        # creating the full equation expression
        expr = left_expr - right_expr
        coeffs = expr.as_coefficients_dict()

        a = float(coeffs.get(x**2, 0))
        b = float(coeffs.get(x, 0))
        c = float(coeffs.get(1, 0))

        if a == 0:
            a = 1e-6  # to prevent div by 0

        discriminant = b**2 - 4*a*c
        x_vertex = -b / (2*a)
        num_real_roots = float(discriminant >= 0)

        return [a, b, c, b / a, float(a > 0), float(b > 0), abs(b) / (abs(a) + 1e-6), discriminant, x_vertex,
                num_real_roots, float(a != 0)]

    except Exception as e:
        print(f"[Extract cofficients] Parse error : {e}")
        return [0.0] * 11


def predict_with_model(equation: str, model: torch.nn.Module) -> list[float]:
    """
    Uses a trained neural network to predict the solution.
    """
    # step 1 convert the equation string into numerical features
    features = extract_cofficients(equation)
    input_tensor = torch.tensor([features], dtype=torch.float32)

    # step 2 model prediction
    model.eval()
    with torch.no_grad():
        output_tensor = model(input_tensor)

    # returning the prediction as a list
    return [output_tensor.item()]
