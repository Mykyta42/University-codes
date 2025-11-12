# The first part. Problem 16.
# Write function, which adds one to even values of the dictionary.


def add_one_to_even(s):
    try:
        for el in s.keys():
            if not isinstance(s[el], int):
                raise Exception
            if s[el] % 2 == 0:
                s[el] += 1
        return s
    except:
        return -1  # Inappropriate input


d = 'not a dictionary'  # Examples.
print(add_one_to_even(d))  # -1 is expected
d = {0: 1, 1: 0.5}
print(add_one_to_even(d))  # -1 is expected
d = {}
print(add_one_to_even(d))  # {} is expected
d = {1: 1}
print(add_one_to_even(d))  # {1: 1} is expected
d = {0: 1, 1: 6, 2: 1, 3: 2, 4: 6}
print(add_one_to_even(d))  # {0: 1, 1: 1, 2: 1, 3: 3, 4: 7} is expected
print("The end of the first part. \n")
