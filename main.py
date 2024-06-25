# the standard way to import PySAT:
from pysat.formula import CNF
from pysat.solvers import Solver

# xyz -> na y,z je cislo x

# [111, -112, -113, -114]


class Variable:
    def __init__(self, sign: int, x: int, y: int, z: int):
        self.sign, self.x, self.y, self.z = sign, x, y, z

    def __str__(self):
        return f"{self.x} {self.y} {self.z}"

    def convert_to_int(self):
        # 100 -> 0-0+1
        # 101 -> 1-0+1
        # ...
        # 108 -> 8-0+1

        # 110 -> 10-1+1
        # ...
        # 118 -> 18-1+1
        # 120 -> 20-2+1
        # 188 -> 88-8+1

        # 200 -> (2-1)*81 + 0-0+1
        # 700 -> (7-1)*81 + 0-0+1

        return (self.sign * 2 - 1) * (
            (self.x - 1) * 81 + (self.y * 10 + self.z) - self.y + 1
        )


# clausule connected by & or |
class Clausule:
    def __init__(self, var_list: list[int]):
        self.var_list = var_list

    def add_variable(self, v: Variable):
        self.var_list.append(v)

    def add_variables(self, v_list: list[Variable]):
        self.var_list.extend(v_list)

    def get_valid_clausule(self):
        return [num.convert_to_int() for num in self.var_list]


def get_inverse_variable(value: int) -> Variable:
    sign: int = 0 if value < 0 else 1
    value = abs(value)
    x: int = value // 81 + 1
    rem: int = value % 81
    y: int = (rem - 1) // 9
    z: int = (rem - 1) % 9
    return Variable(sign, x, y, z)

    # 1 -> 00
    # 2 -> 01
    # ...
    # 9 -> 08
    # 10 -> 10
    # ...
    # 18 -> 18
    # 19 -> 20


n: int = 3

clausule_list: list[Clausule] = []

# something at each position
for row in range(n**2):
    for col in range(n**2):
        c: Clausule = Clausule([])
        c.add_variables([Variable(1, num, row, col) for num in range(1, n**2 + 1)])
        clausule_list.append(c)

for num in range(1, n**2 + 1):
    for row in range(n**2):
        for col in range(n**2):
            # if v_i_a_b => !v_j_a_b
            clausule_list.extend(
                [
                    Clausule(
                        [Variable(0, num, row, col), Variable(0, other_num, row, col)]
                    )
                    for other_num in range(1, n**2 + 1)
                    if other_num != num
                ]
            )
            # if v_i_a_b => !v_i_a+k_b
            clausule_list.extend(
                [
                    Clausule([Variable(0, num, row, col), Variable(0, num, v_row, col)])
                    for v_row in range(n**2)
                    if v_row != row
                ]
            )
            # if v_i_a_b => !v_i_a_b+k
            clausule_list.extend(
                [
                    Clausule([Variable(0, num, row, col), Variable(0, num, row, v_col)])
                    for v_col in range(n**2)
                    if v_col != col
                ]
            )
            # if v_i_a_b => !v_i_a_b in a block
            area_row = row // n
            area_col = col // n
            for a_row in range(area_row * n, area_row * n + n):
                for a_col in range(area_col * n, area_col * n + n):
                    if not (a_row == row and a_col == col):
                        clausule_list.append(
                            Clausule(
                                [
                                    Variable(0, num, row, col),
                                    Variable(0, num, a_row, a_col),
                                ]
                            )
                        )


# create a satisfiable CNF formula "(-x1 ∨ x2) ∧ (-x1 ∨ -x2)":
cnf = CNF(from_clauses=[c.get_valid_clausule() for c in clausule_list])

# [[-1, 2], [-1, -2]])

# create a SAT solver for this formula:
with Solver(bootstrap_with=cnf) as solver:
    # 1.1 call the solver for this formula:
    print("formula is", f'{"s" if solver.solve() else "uns"}atisfiable')

    # 1.2 the formula is satisfiable and so has a model:
    print("and the model is:", solver.get_model())

    model = solver.get_model()

    solved_sudoku: list[list[int]] = [[0 for i in range(n**2)] for j in range(n**2)]

    for num in model:
        if num > 0:
            v: Variable = get_inverse_variable(num)
            solved_sudoku[v.y][v.z] = v.x

    for row in solved_sudoku:
        for num in row:
            print(num, end=" ")
        print("")
