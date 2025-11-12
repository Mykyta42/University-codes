"""
The author of the function is Mykyta Kharin.
The variant is 89.
"""


def _more_5_div(n: int) -> bool:
    """Check the number of positive divisors being more than 5."""
    num = 0
    for i in range(1, abs(n)+1):
        if n % i == 0:
            num += 1
    return n == 0 or num > 5


def _compare(p: int, q: int) -> int:
    """Number equivalent to result of the comparison."""
    if p > q:
        return 1
    elif p == q:
        return 2
    else:
        return 3


def subsequence(t: tuple) -> list:
    """
    Return a list with the subsequence.
    Argument:
    t - a tuple with the sequence.
    """
    s_size = 1
    d_size = 1
    s_from = 0
    s_to = -1
    d_from = 0
    d_to = 0
    state = 0
    help_i = None
    for i in range(len(t)):
        if _more_5_div(t[i]):
            if help_i is None:
                help_i, d_from = t[i], i
                continue
            if state == 0 and t[i] <= help_i:
                d_from = i
            dif = _compare(t[i], help_i) - state
            if 0 <= dif <= 1:
                state += dif
                d_size += 1
                d_to = i
            else:
                if state == 3 and s_size < d_size:
                    s_size, s_from, s_to = d_size, d_from, d_to
                state, d_size, d_from = 0, 1, i
                if t[i] > help_i:
                    d_from, d_size, state = d_to, 2, 1
            help_i = t[i]
    if state == 3 and s_size < d_size:
        s_from, s_to = d_from, d_to
    s = [i for i in t[s_from:s_to + 1] if _more_5_div(i)]
    return s
