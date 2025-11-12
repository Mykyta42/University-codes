import math


def f(x):
    """
    Return the value of corresponding expression.

    Arguments:
    x - a real number
    pre: x should be at least 7 and not equal 11 or 12.

    Don`t check belonging of arguments to the domain.
    On error raise an exception.
    """
    return math.cos(15 / 49) - 10 * math.pi - 58 * math.e * 9 / ((x + 13) * (x - 11)) + 12 * math.cos(x - 8) + (
            5 + math.sqrt(x - 7)) / (x - 12)


def check_domain(x):
    """
    Check belonging of input to the domain of given expression.
    If do, return True. Else return False.

    Arguments:
    x - a real number.

    On error raise an exception.
    """
    return x >= 7 and x != 11 and x != 12


def main():
    """
    Output the main part of the program.
    """
    try:
        x = float(input("Enter a real number x here: "))
        print('\n***** do calculations ...', end=' ')
        if check_domain(x):
            answer = f"{f(x):.8f}"
        else:
            answer = "undefined"
        print('done')
        print(f"for {x = :.8f}")
        print(f"result = {answer}")
    except Exception:
        print('\nwrong input')


print("""I'm here to make your life easier. My creator is Mykyta Kharin.
This program computes the expression by given variable x. Variant 89""")
try:
    main()
except KeyboardInterrupt:
    print('\nprogram aborted')
print('Thank you for using me, bye!')
