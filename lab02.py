"""
This program computes the approximative value of the series.
A point for computations and an accuracy are entered by the user.
Input should be real and belong to the fixed range.
On error print a message and show the diagnostics.
"""


A = 0 #LowerBound
B = 1 #UpperBound


def s(x: float, eps: float) -> float:
    """
    Return the approximative value of the series.

    Key arguments:
    x - a point for computations.
    eps - an accuracy.
    pre: x and eps should belong to the range.

    Don`t check corectness of the arguments.
    """
    a = 1.0 #the first member of the sequence
    S = a #primary sum
    k = 1 #step of the cycle
    x = x * x * x * x
    while a >= eps or a <= -eps:
        a *= x / k / k / (k + 1) / (k + 1)
        S += a
        k += 2
    return S


def _check_x(var: float) -> bool:
    """Check belonging of the point to the segment."""
    return A <= var <= B


def _check_eps(var: float) -> bool:
    """Check the accuracy for being positive."""
    return var > 0


def _float_input(prompt=None) -> float:
    """Obtain and convert data from the user."""
    try:
        var = input(prompt)
    except Exception:
        raise Exception('The input is unreadable. Try something simple!')
    try:
        var = float(var)
    except Exception:
        raise Exception('It doesn\'t look like a number. Use digits, not letters!')
    return var

def _input_with_check(prompt, check) -> float:
    """Check correctness of the input ."""
    var = _float_input(prompt)
    if not check(var):
        raise ValueError('This number isn\'t allowed. Read instructions attentively!')
    return var


def final_input(prompt, check, varname=None) -> float:
    """
    Return correct input.
    In other cases create diagnostic.
    """
    try:
        var = _input_with_check(prompt, check)
        return var
    except Exception as h:
        raise Exception(f'All went wrong because of {varname}', f'{h}')

def main() -> None:
    """
    Output the result of computations.
    On error print a message and corresponding diagnostics.
    """
    try:
        x = final_input('Enter a real number x from [0, 1] here: ', _check_x, 'x')
        eps = final_input('\nEnter a positive real number eps here: ', _check_eps, 'eps')
        print('\n***** do calculations ...', end = ' ')
        result = s(x, eps)
        print('done')
        print(f"for {x = :.5f}")
        print(f"for {eps = :.4E}")
        print(f"{result = :.9f}")
    except Exception as e:
        print('\n***** error')
        print(e)


print("""Hello! Follow intructions or get a bunch of weird text. The author is Mykyta Kharin.
This program calculates the approximative value of the series, which exact measure equals ... Variant 89""")
try:
    main()
except KeyboardInterrupt:
    print('\nprogram aborted')
print('See you next time, bye!')
