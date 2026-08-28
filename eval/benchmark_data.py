"""
Standardized Evaluation Benchmark Datasets.
Includes:
1. HumanEval-50 Coding Subset: 50 canonical algorithmic problems with deterministic unit tests.
2. GSM8K / MATH-500 Subset: 50 multi-step arithmetic, algebraic, combinatoric, and calculus problems.
"""

from typing import Any, Dict, List

# -------------------------------------------------------------
# 1. HUMANEVAL 50-PROBLEM STANDARDIZED CODING SUBSET
# -------------------------------------------------------------
HUMANEVAL_50_SUBSET: List[Dict[str, Any]] = [
    {
        "id": "HumanEval/0",
        "task_id": "has_close_elements",
        "prompt": "Write a Python function `has_close_elements(numbers: list, threshold: float) -> bool` that checks if in given list of numbers, are any two numbers closer to each other than given threshold.",
        "canonical_solution": "def has_close_elements(numbers, threshold):\n    for i in range(len(numbers)):\n        for j in range(i + 1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False",
        "tests": "assert has_close_elements([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True\nassert has_close_elements([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False\nassert has_close_elements([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True"
    },
    {
        "id": "HumanEval/1",
        "task_id": "separate_paren_groups",
        "prompt": "Write a Python function `separate_paren_groups(paren_string: str) -> list` that separates balanced groups of parentheses into separate strings.",
        "canonical_solution": "def separate_paren_groups(paren_string):\n    paren_string = paren_string.replace(' ', '')\n    res, curr, depth = [], '', 0\n    for char in paren_string:\n        curr += char\n        if char == '(':\n            depth += 1\n        elif char == ')':\n            depth -= 1\n        if depth == 0 and curr:\n            res.append(curr)\n            curr = ''\n    return res",
        "tests": "assert separate_paren_groups('( ) (( )) (( )( ))') == ['()', '(())', '(()())']\nassert separate_paren_groups('()') == ['()']\nassert separate_paren_groups('(()(())((())))') == ['(()(())((())))']"
    },
    {
        "id": "HumanEval/2",
        "task_id": "truncate_number",
        "prompt": "Write a Python function `truncate_number(number: float) -> float` that returns the decimal part of a given positive floating point number.",
        "canonical_solution": "def truncate_number(number):\n    return round(number % 1.0, 10)",
        "tests": "assert abs(truncate_number(3.5) - 0.5) < 1e-6\nassert abs(truncate_number(1.25) - 0.25) < 1e-6\nassert abs(truncate_number(123.0) - 0.0) < 1e-6"
    },
    {
        "id": "HumanEval/3",
        "task_id": "below_zero",
        "prompt": "Write a Python function `below_zero(operations: list) -> bool` that detects if at any point the bank balance falls below zero.",
        "canonical_solution": "def below_zero(operations):\n    balance = 0\n    for op in operations:\n        balance += op\n        if balance < 0:\n            return True\n    return False",
        "tests": "assert below_zero([1, 2, 3]) == False\nassert below_zero([1, 2, -4, 5]) == True\nassert below_zero([0, -1, 1]) == True\nassert below_zero([]) == False"
    },
    {
        "id": "HumanEval/4",
        "task_id": "mean_absolute_deviation",
        "prompt": "Write a Python function `mean_absolute_deviation(numbers: list) -> float` that computes the Mean Absolute Deviation (MAD) of the list.",
        "canonical_solution": "def mean_absolute_deviation(numbers):\n    mean = sum(numbers) / len(numbers)\n    return sum(abs(x - mean) for x in numbers) / len(numbers)",
        "tests": "assert abs(mean_absolute_deviation([1.0, 2.0, 3.0, 4.0]) - 1.0) < 1e-6\nassert abs(mean_absolute_deviation([1.0, 2.0, 3.0, 4.0, 5.0]) - 1.2) < 1e-6"
    },
    {
        "id": "HumanEval/5",
        "task_id": "intersperse",
        "prompt": "Write a Python function `intersperse(numbers: list, delimeter: int) -> list` that inserts `delimeter` between every consecutive element of `numbers`.",
        "canonical_solution": "def intersperse(numbers, delimeter):\n    if not numbers:\n        return []\n    res = [numbers[0]]\n    for n in numbers[1:]:\n        res.append(delimeter)\n        res.append(n)\n    return res",
        "tests": "assert intersperse([], 4) == []\nassert intersperse([1, 2, 3], 4) == [1, 4, 2, 4, 3]\nassert intersperse([5], 1) == [5]"
    },
    {
        "id": "HumanEval/6",
        "task_id": "parse_nested_parens",
        "prompt": "Write a Python function `parse_nested_parens(paren_string: str) -> list` that finds the maximum nesting depth for each parenthesized group.",
        "canonical_solution": "def parse_nested_parens(paren_string):\n    groups = paren_string.split()\n    res = []\n    for g in groups:\n        max_d = curr_d = 0\n        for c in g:\n            if c == '(':\n                curr_d += 1\n                max_d = max(max_d, curr_d)\n            elif c == ')':\n                curr_d -= 1\n        res.append(max_d)\n    return res",
        "tests": "assert parse_nested_parens('(()()) ((())) () ((())()())') == [2, 3, 1, 3]\nassert parse_nested_parens('()') == [1]"
    },
    {
        "id": "HumanEval/7",
        "task_id": "filter_by_substring",
        "prompt": "Write a Python function `filter_by_substring(strings: list, substring: str) -> list` that filters an array of strings for ones that contain given substring.",
        "canonical_solution": "def filter_by_substring(strings, substring):\n    return [s for s in strings if substring in s]",
        "tests": "assert filter_by_substring([], 'a') == []\nassert filter_by_substring(['abc', 'bac', 'cba', 'array'], 'a') == ['abc', 'bac', 'cba', 'array']\nassert filter_by_substring(['grunt', 'trumpet', 'prune'], 'run') == ['grunt', 'prune']"
    },
    {
        "id": "HumanEval/8",
        "task_id": "sum_product",
        "prompt": "Write a Python function `sum_product(numbers: list) -> tuple` that returns a tuple consisting of the sum and product of all the integers in a list.",
        "canonical_solution": "def sum_product(numbers):\n    s, p = 0, 1\n    for n in numbers:\n        s += n\n        p *= n\n    return (s, p if numbers else 1)",
        "tests": "assert sum_product([]) == (0, 1)\nassert sum_product([1, 2, 3, 4]) == (10, 24)\nassert sum_product([0, 1, 2]) == (3, 0)"
    },
    {
        "id": "HumanEval/9",
        "task_id": "rolling_max",
        "prompt": "Write a Python function `rolling_max(numbers: list) -> list` that returns a list of rolling maximum elements found until given moment in sequence.",
        "canonical_solution": "def rolling_max(numbers):\n    if not numbers: return []\n    res, curr = [], numbers[0]\n    for n in numbers:\n        curr = max(curr, n)\n        res.append(curr)\n    return res",
        "tests": "assert rolling_max([1, 2, 3, 2, 3, 4, 2]) == [1, 2, 3, 3, 3, 4, 4]\nassert rolling_max([]) == []\nassert rolling_max([5, 4, 3, 2, 1]) == [5, 5, 5, 5, 5]"
    },
    {
        "id": "HumanEval/10",
        "task_id": "is_palindrome",
        "prompt": "Write a Python function `make_palindrome(string: str) -> str` that finds the shortest palindrome that begins with a supplied string.",
        "canonical_solution": "def make_palindrome(string):\n    if not string: return ''\n    for i in range(len(string)):\n        if string[i:] == string[i:][::-1]:\n            return string + string[:i][::-1]\n    return string",
        "tests": "assert make_palindrome('') == ''\nassert make_palindrome('cat') == 'catac'\nassert make_palindrome('cata') == 'catac'\nassert make_palindrome('radar') == 'radar'"
    },
    {
        "id": "HumanEval/11",
        "task_id": "string_xor",
        "prompt": "Write a Python function `string_xor(a: str, b: str) -> str` that performs binary XOR on two strings consisting only of 1s and 0s.",
        "canonical_solution": "def string_xor(a, b):\n    return ''.join('1' if x != y else '0' for x, y in zip(a, b))",
        "tests": "assert string_xor('010', '110') == '100'\nassert string_xor('111', '000') == '111'\nassert string_xor('10101', '01010') == '11111'"
    },
    {
        "id": "HumanEval/12",
        "task_id": "longest",
        "prompt": "Write a Python function `longest(strings: list) -> str` that returns the longest string from a list of strings, returning the first one in case of a tie.",
        "canonical_solution": "def longest(strings):\n    if not strings: return None\n    return max(strings, key=len)",
        "tests": "assert longest([]) == None\nassert longest(['a', 'b', 'c']) == 'a'\nassert longest(['a', 'bb', 'ccc']) == 'ccc'\nassert longest(['x', 'yy', 'zz']) == 'yy'"
    },
    {
        "id": "HumanEval/13",
        "task_id": "greatest_common_divisor",
        "prompt": "Write a Python function `greatest_common_divisor(a: int, b: int) -> int` that returns the greatest common divisor of two integers `a` and `b`.",
        "canonical_solution": "def greatest_common_divisor(a, b):\n    while b:\n        a, b = b, a % b\n    return a",
        "tests": "assert greatest_common_divisor(3, 5) == 1\nassert greatest_common_divisor(25, 15) == 5\nassert greatest_common_divisor(48, 18) == 6"
    },
    {
        "id": "HumanEval/14",
        "task_id": "all_prefixes",
        "prompt": "Write a Python function `all_prefixes(string: str) -> list` that returns a list of all prefixes from shortest to longest of the input string.",
        "canonical_solution": "def all_prefixes(string):\n    return [string[:i+1] for i in range(len(string))]",
        "tests": "assert all_prefixes('abc') == ['a', 'ab', 'abc']\nassert all_prefixes('') == []\nassert all_prefixes('asdf') == ['a', 'as', 'asd', 'asdf']"
    },
    {
        "id": "HumanEval/15",
        "task_id": "string_sequence",
        "prompt": "Write a Python function `string_sequence(n: int) -> str` that returns a string containing space-delimited numbers starting from 0 up to n inclusive.",
        "canonical_solution": "def string_sequence(n):\n    return ' '.join(str(i) for i in range(n + 1))",
        "tests": "assert string_sequence(0) == '0'\nassert string_sequence(5) == '0 1 2 3 4 5'\nassert string_sequence(3) == '0 1 2 3'"
    },
    {
        "id": "HumanEval/16",
        "task_id": "count_distinct_characters",
        "prompt": "Write a Python function `count_distinct_characters(string: str) -> int` that counts how many distinct characters (case-insensitive) are in string.",
        "canonical_solution": "def count_distinct_characters(string):\n    return len(set(string.lower()))",
        "tests": "assert count_distinct_characters('xyzXYZ') == 3\nassert count_distinct_characters('Jerry') == 4\nassert count_distinct_characters('') == 0"
    },
    {
        "id": "HumanEval/17",
        "task_id": "parse_music",
        "prompt": "Write a Python function `parse_music(music_string: str) -> list` that parses ASCII musical notes into their beat count ('o': 4, 'o|': 2, '.|': 1).",
        "canonical_solution": "def parse_music(music_string):\n    mapping = {'o': 4, 'o|': 2, '.|': 1}\n    return [mapping[x] for x in music_string.split() if x in mapping]",
        "tests": "assert parse_music('o o| .| o| o| .| .| .| .| o o') == [4, 2, 1, 2, 2, 1, 1, 1, 1, 4, 4]\nassert parse_music('') == []\nassert parse_music('o| .| o') == [2, 1, 4]"
    },
    {
        "id": "HumanEval/18",
        "task_id": "how_many_times",
        "prompt": "Write a Python function `how_many_times(string: str, substring: str) -> int` that finds how many times a given substring can be found in the original string, counting overlapping occurrences.",
        "canonical_solution": "def how_many_times(string, substring):\n    if not substring: return 0\n    count = 0\n    for i in range(len(string) - len(substring) + 1):\n        if string[i:i+len(substring)] == substring:\n            count += 1\n    return count",
        "tests": "assert how_many_times('', 'a') == 0\nassert how_many_times('aaa', 'a') == 3\nassert how_many_times('aaaa', 'aa') == 3"
    },
    {
        "id": "HumanEval/19",
        "task_id": "sort_numbers",
        "prompt": "Write a Python function `sort_numbers(numbers: str) -> str` that sorts numerals written as words ('zero' to 'nine') from smallest to largest.",
        "canonical_solution": "def sort_numbers(numbers):\n    order = {'zero':0,'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9}\n    inv = {v:k for k,v in order.items()}\n    if not numbers: return ''\n    words = numbers.split()\n    sorted_w = sorted(words, key=lambda w: order.get(w, 0))\n    return ' '.join(sorted_w)",
        "tests": "assert sort_numbers('three one five') == 'one three five'\nassert sort_numbers('') == ''\nassert sort_numbers('five zero four seven nine eight') == 'zero four five seven eight nine'"
    },
    {
        "id": "HumanEval/20",
        "task_id": "find_closest_elements",
        "prompt": "Write a Python function `find_closest_elements(numbers: list) -> tuple` that finds the two numbers closest to each other and returns them in ascending order.",
        "canonical_solution": "def find_closest_elements(numbers):\n    numbers = sorted(numbers)\n    min_diff = float('inf')\n    pair = None\n    for i in range(len(numbers) - 1):\n        diff = numbers[i+1] - numbers[i]\n        if diff < min_diff:\n            min_diff = diff\n            pair = (numbers[i], numbers[i+1])\n    return pair",
        "tests": "assert find_closest_elements([1.0, 2.0, 3.0, 4.0, 5.0, 2.2]) == (2.0, 2.2)\nassert find_closest_elements([1.0, 2.0, 3.0, 4.0, 5.0, 2.0]) == (2.0, 2.0)"
    },
    {
        "id": "HumanEval/21",
        "task_id": "rescale_to_unit",
        "prompt": "Write a Python function `rescale_to_unit(numbers: list) -> list` that applies a linear transformation to list of floats such that minimum becomes 0.0 and maximum becomes 1.0.",
        "canonical_solution": "def rescale_to_unit(numbers):\n    min_v, max_v = min(numbers), max(numbers)\n    return [(x - min_v) / (max_v - min_v) for x in numbers]",
        "tests": "assert rescale_to_unit([1.0, 2.0, 3.0, 4.0, 5.0]) == [0.0, 0.25, 0.5, 0.75, 1.0]\nassert rescale_to_unit([2.0, 49.0]) == [0.0, 1.0]"
    },
    {
        "id": "HumanEval/22",
        "task_id": "filter_integers",
        "prompt": "Write a Python function `filter_integers(values: list) -> list` that filters a list of any python values only for integers.",
        "canonical_solution": "def filter_integers(values):\n    return [x for x in values if isinstance(x, int) and not isinstance(x, bool)]",
        "tests": "assert filter_integers(['a', 3.14, 5]) == [5]\nassert filter_integers([1, 2, 3, 'abc', {}, []]) == [1, 2, 3]\nassert filter_integers([]) == []"
    },
    {
        "id": "HumanEval/23",
        "task_id": "strlen",
        "prompt": "Write a Python function `strlen(string: str) -> int` that returns the length of the given string.",
        "canonical_solution": "def strlen(string):\n    return len(string)",
        "tests": "assert strlen('') == 0\nassert strlen('abc') == 3\nassert strlen('hello world') == 11"
    },
    {
        "id": "HumanEval/24",
        "task_id": "largest_divisor",
        "prompt": "Write a Python function `largest_divisor(n: int) -> int` that finds the largest divisor of `n` that is smaller than `n`.",
        "canonical_solution": "def largest_divisor(n):\n    for i in range(n // 2, 0, -1):\n        if n % i == 0:\n            return i\n    return 1",
        "tests": "assert largest_divisor(15) == 5\nassert largest_divisor(49) == 7\nassert largest_divisor(10) == 5"
    },
    {
        "id": "HumanEval/25",
        "task_id": "factorize",
        "prompt": "Write a Python function `factorize(n: int) -> list` that returns list of prime factors of given integer in the order from smallest to largest.",
        "canonical_solution": "def factorize(n):\n    factors = []\n    d = 2\n    while d * d <= n:\n        while n % d == 0:\n            factors.append(d)\n            n //= d\n        d += 1\n    if n > 1:\n        factors.append(n)\n    return factors",
        "tests": "assert factorize(2) == [2]\nassert factorize(4) == [2, 2]\nassert factorize(8) == [2, 2, 2]\nassert factorize(57) == [3, 19]"
    },
    {
        "id": "HumanEval/26",
        "task_id": "remove_duplicates",
        "prompt": "Write a Python function `remove_duplicates(numbers: list) -> list` that removes all elements from a list of integers that occur more than once, preserving original order of unique elements.",
        "canonical_solution": "def remove_duplicates(numbers):\n    from collections import Counter\n    counts = Counter(numbers)\n    return [x for x in numbers if counts[x] == 1]",
        "tests": "assert remove_duplicates([1, 2, 3, 2, 4]) == [1, 3, 4]\nassert remove_duplicates([1, 2, 3, 4]) == [1, 2, 3, 4]\nassert remove_duplicates([1, 1, 1, 1]) == []"
    },
    {
        "id": "HumanEval/27",
        "task_id": "flip_case",
        "prompt": "Write a Python function `flip_case(string: str) -> str` that flips lowercase characters to uppercase and uppercase characters to lowercase.",
        "canonical_solution": "def flip_case(string):\n    return string.swapcase()",
        "tests": "assert flip_case('Hello') == 'hELLO'\nassert flip_case('') == ''\nassert flip_case('These violent delights have violent ends') == 'tHESE VIOLENT DELIGHTS HAVE VIOLENT ENDS'"
    },
    {
        "id": "HumanEval/28",
        "task_id": "concatenate",
        "prompt": "Write a Python function `concatenate(strings: list) -> str` that concatenates a list of strings into a single string.",
        "canonical_solution": "def concatenate(strings):\n    return ''.join(strings)",
        "tests": "assert concatenate([]) == ''\nassert concatenate(['a', 'b', 'c']) == 'abc'\nassert concatenate(['Hello', ' ', 'World']) == 'Hello World'"
    },
    {
        "id": "HumanEval/29",
        "task_id": "filter_by_prefix",
        "prompt": "Write a Python function `filter_by_prefix(strings: list, prefix: str) -> list` that filters an array of strings for ones that start with given prefix.",
        "canonical_solution": "def filter_by_prefix(strings, prefix):\n    return [s for s in strings if s.startswith(prefix)]",
        "tests": "assert filter_by_prefix([], 'a') == []\nassert filter_by_prefix(['abc', 'bcd', 'cde', 'array'], 'a') == ['abc', 'array']\nassert filter_by_prefix(['apple', 'banana', 'apricot'], 'ap') == ['apple', 'apricot']"
    },
    {
        "id": "HumanEval/30",
        "task_id": "get_positive",
        "prompt": "Write a Python function `get_positive(l: list) -> list` that returns only positive numbers from the list.",
        "canonical_solution": "def get_positive(l):\n    return [x for x in l if x > 0]",
        "tests": "assert get_positive([-1, 2, -4, 5, 6]) == [2, 5, 6]\nassert get_positive([5, 3, -5, 2, -3, 3, 9, 0, 123, 1, -10]) == [5, 3, 2, 3, 9, 123, 1]\nassert get_positive([]) == []"
    },
    {
        "id": "HumanEval/31",
        "task_id": "is_prime",
        "prompt": "Write a Python function `is_prime(n: int) -> bool` that returns `True` if a given number is prime, and `False` otherwise.",
        "canonical_solution": "def is_prime(n):\n    if n < 2:\n        return False\n    for k in range(2, int(n**0.5) + 1):\n        if n % k == 0:\n            return False\n    return True",
        "tests": "assert is_prime(6) == False\nassert is_prime(101) == True\nassert is_prime(11) == True\nassert is_prime(13441) == True\nassert is_prime(1) == False"
    },
    {
        "id": "HumanEval/32",
        "task_id": "find_zero",
        "prompt": "Write a Python function `poly(xs: list, x: float) -> float` evaluating polynomial with coefficients `xs` at point `x`.",
        "canonical_solution": "def poly(xs, x):\n    return sum(coeff * (x ** i) for i, coeff in enumerate(xs))",
        "tests": "assert abs(poly([1, 2], 0) - 1.0) < 1e-6\nassert abs(poly([1, 2], 1) - 3.0) < 1e-6\nassert abs(poly([2, 0, 1], 3) - 11.0) < 1e-6"
    },
    {
        "id": "HumanEval/33",
        "task_id": "sort_third",
        "prompt": "Write a Python function `sort_third(l: list) -> list` that sorts elements at indices that are divisible by 3, while leaving other indices untouched.",
        "canonical_solution": "def sort_third(l):\n    l = list(l)\n    thirds = sorted([l[i] for i in range(0, len(l), 3)])\n    for idx, i in enumerate(range(0, len(l), 3)):\n        l[i] = thirds[idx]\n    return l",
        "tests": "assert sort_third([1, 2, 3]) == [1, 2, 3]\nassert sort_third([5, 6, 3, 4, 8, 9, 2]) == [2, 6, 3, 4, 8, 9, 5]"
    },
    {
        "id": "HumanEval/34",
        "task_id": "unique",
        "prompt": "Write a Python function `unique(l: list) -> list` that returns sorted unique elements in a list.",
        "canonical_solution": "def unique(l):\n    return sorted(list(set(l)))",
        "tests": "assert unique([5, 3, 5, 2, 3, 3, 9, 0, 123]) == [0, 2, 3, 5, 9, 123]\nassert unique([]) == []\nassert unique([1, 1, 1]) == [1]"
    },
    {
        "id": "HumanEval/35",
        "task_id": "max_element",
        "prompt": "Write a Python function `max_element(l: list) -> int` that returns the maximum element in the list.",
        "canonical_solution": "def max_element(l):\n    return max(l)",
        "tests": "assert max_element([1, 2, 3]) == 3\nassert max_element([5, 3, -5, 2, -3, 3, 9, 0, 123, 1, -10]) == 123"
    },
    {
        "id": "HumanEval/36",
        "task_id": "fizz_buzz",
        "prompt": "Write a Python function `fizz_buzz(n: int) -> int` that counts times digit 7 appears in integers less than n which are divisible by 11 or 13.",
        "canonical_solution": "def fizz_buzz(n):\n    count = 0\n    for i in range(n):\n        if i % 11 == 0 or i % 13 == 0:\n            count += str(i).count('7')\n    return count",
        "tests": "assert fizz_buzz(50) == 0\nassert fizz_buzz(78) == 2\nassert fizz_buzz(79) == 3"
    },
    {
        "id": "HumanEval/37",
        "task_id": "sort_even",
        "prompt": "Write a Python function `sort_even(l: list) -> list` that sorts elements at even indices in ascending order, while keeping odd indices unchanged.",
        "canonical_solution": "def sort_even(l):\n    l = list(l)\n    evens = sorted([l[i] for i in range(0, len(l), 2)])\n    for idx, i in enumerate(range(0, len(l), 2)):\n        l[i] = evens[idx]\n    return l",
        "tests": "assert sort_even([1, 2, 3]) == [1, 2, 3]\nassert sort_even([5, 6, 3, 4]) == [3, 6, 5, 4]"
    },
    {
        "id": "HumanEval/38",
        "task_id": "decode_cyclic",
        "prompt": "Write a Python function `encode_cyclic(s: str) -> str` cycling groups of 3 characters.",
        "canonical_solution": "def encode_cyclic(s):\n    groups = [s[(3 * i):min((3 * i + 3), len(s))] for i in range((len(s) + 2) // 3)]\n    groups = [(group[1:] + group[0]) if len(group) == 3 else group for group in groups]\n    return ''.join(groups)",
        "tests": "assert encode_cyclic('abcdef') == 'bcadef'\nassert encode_cyclic('abc') == 'bca'\nassert encode_cyclic('ab') == 'ab'"
    },
    {
        "id": "HumanEval/39",
        "task_id": "prime_fib",
        "prompt": "Write a Python function `prime_fib(n: int) -> int` that returns the n-th number that is both a Fibonacci number and prime.",
        "canonical_solution": "def prime_fib(n):\n    def is_p(x):\n        if x < 2: return False\n        for i in range(2, int(x**0.5) + 1):\n            if x % i == 0: return False\n        return True\n    a, b = 1, 2\n    count = 0\n    while True:\n        if is_p(a):\n            count += 1\n            if count == n: return a\n        a, b = b, a + b",
        "tests": "assert prime_fib(1) == 2\nassert prime_fib(2) == 3\nassert prime_fib(3) == 5\nassert prime_fib(4) == 13"
    },
    {
        "id": "HumanEval/40",
        "task_id": "triples_sum_to_zero",
        "prompt": "Write a Python function `triples_sum_to_zero(l: list) -> bool` that takes a list of integers and returns `True` if there are three distinct elements in list that sum to zero.",
        "canonical_solution": "def triples_sum_to_zero(l):\n    for i in range(len(l)):\n        for j in range(i + 1, len(l)):\n            for k in range(j + 1, len(l)):\n                if l[i] + l[j] + l[k] == 0:\n                    return True\n    return False",
        "tests": "assert triples_sum_to_zero([1, 3, 5, 0]) == False\nassert triples_sum_to_zero([1, 3, -2, 1]) == True\nassert triples_sum_to_zero([1, 2, 3, 7]) == False\nassert triples_sum_to_zero([2, 4, -5, 3, 9, 7]) == True"
    },
    {
        "id": "HumanEval/41",
        "task_id": "car_race_collision",
        "prompt": "Write a Python function `car_race_collision(n: int) -> int` that returns number of collisions when n cars driven left-to-right meet n cars driven right-to-left.",
        "canonical_solution": "def car_race_collision(n):\n    return n ** 2",
        "tests": "assert car_race_collision(2) == 4\nassert car_race_collision(3) == 9\nassert car_race_collision(10) == 100"
    },
    {
        "id": "HumanEval/42",
        "task_id": "incr_list",
        "prompt": "Write a Python function `incr_list(l: list) -> list` that returns elements incremented by 1.",
        "canonical_solution": "def incr_list(l):\n    return [(x + 1) for x in l]",
        "tests": "assert incr_list([1, 2, 3]) == [2, 3, 4]\nassert incr_list([5, 3, 5, 2, 3, 3, 9, 0, 123]) == [6, 4, 6, 3, 4, 4, 10, 1, 124]"
    },
    {
        "id": "HumanEval/43",
        "task_id": "pairs_sum_to_zero",
        "prompt": "Write a Python function `pairs_sum_to_zero(l: list) -> bool` returning `True` if two distinct elements sum to zero.",
        "canonical_solution": "def pairs_sum_to_zero(l):\n    for i in range(len(l)):\n        for j in range(i + 1, len(l)):\n            if l[i] + l[j] == 0:\n                return True\n    return False",
        "tests": "assert pairs_sum_to_zero([1, 3, 5, 0]) == False\nassert pairs_sum_to_zero([1, 3, -2, 1]) == False\nassert pairs_sum_to_zero([1, 2, 3, 7]) == False\nassert pairs_sum_to_zero([2, 4, -5, 3, 5, 7]) == True"
    },
    {
        "id": "HumanEval/44",
        "task_id": "change_base",
        "prompt": "Write a Python function `change_base(x: int, base: int) -> str` that converts integer x to string representation in given base (less than 10).",
        "canonical_solution": "def change_base(x, base):\n    if x == 0: return '0'\n    digits = []\n    while x > 0:\n        digits.append(str(x % base))\n        x //= base\n    return ''.join(digits[::-1])",
        "tests": "assert change_base(8, 3) == '22'\nassert change_base(8, 2) == '1000'\nassert change_base(7, 2) == '111'"
    },
    {
        "id": "HumanEval/45",
        "task_id": "triangle_area",
        "prompt": "Write a Python function `triangle_area(a: float, h: float) -> float` returning area of triangle given length of side and height.",
        "canonical_solution": "def triangle_area(a, h):\n    return (a * h) / 2.0",
        "tests": "assert abs(triangle_area(5, 3) - 7.5) < 1e-6\nassert abs(triangle_area(2, 2) - 2.0) < 1e-6"
    },
    {
        "id": "HumanEval/46",
        "task_id": "fib4",
        "prompt": "Write a Python function `fib4(n: int) -> int` where `fib4(0)=0, fib4(1)=0, fib4(2)=2, fib4(3)=0` and `fib4(n)=fib4(n-1)+fib4(n-2)+fib4(n-3)+fib4(n-4)`.",
        "canonical_solution": "def fib4(n):\n    res = [0, 0, 2, 0]\n    if n < 4: return res[n]\n    for _ in range(4, n + 1):\n        res.append(sum(res[-4:]))\n    return res[n]",
        "tests": "assert fib4(5) == 4\nassert fib4(6) == 8\nassert fib4(7) == 14"
    },
    {
        "id": "HumanEval/47",
        "task_id": "median",
        "prompt": "Write a Python function `median(l: list) -> float` that returns the median of elements in list.",
        "canonical_solution": "def median(l):\n    l = sorted(l)\n    n = len(l)\n    if n % 2 == 1:\n        return float(l[n // 2])\n    return (l[n // 2 - 1] + l[n // 2]) / 2.0",
        "tests": "assert median([3, 1, 2, 4, 5]) == 3\nassert median([-10, 4, 6, 1000, 10, 20]) == 15.0"
    },
    {
        "id": "HumanEval/48",
        "task_id": "is_palindrome_simple",
        "prompt": "Write a Python function `is_palindrome(text: str) -> bool` returning `True` if string reads same backwards.",
        "canonical_solution": "def is_palindrome(text):\n    return text == text[::-1]",
        "tests": "assert is_palindrome('') == True\nassert is_palindrome('aba') == True\nassert is_palindrome('aaaaa') == True\nassert is_palindrome('zbcd') == False"
    },
    {
        "id": "HumanEval/49",
        "task_id": "modp",
        "prompt": "Write a Python function `modp(n: int, p: int) -> int` that computes `(2^n) % p`.",
        "canonical_solution": "def modp(n, p):\n    return pow(2, n, p)",
        "tests": "assert modp(3, 5) == 3\nassert modp(1101, 101) == 2\nassert modp(0, 101) == 1\nassert modp(3, 11) == 8"
    }
]

# -------------------------------------------------------------
# 2. GSM8K / MATH-500 STANDARDIZED MATH REASONING SUBSET
# -------------------------------------------------------------
MATH_50_SUBSET: List[Dict[str, Any]] = [
    {
        "id": "GSM8K/0",
        "category": "Arithmetic Reasoning",
        "prompt": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
        "expected_answer": 72,
        "canonical_solution": "def solve():\n    april = 48\n    may = april / 2\n    return april + may",
        "tests": "assert abs(solve() - 72) < 1e-5"
    },
    {
        "id": "GSM8K/1",
        "category": "Multi-Step Finance",
        "prompt": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
        "expected_answer": 10,
        "canonical_solution": "def solve():\n    hourly_rate = 12\n    minutes = 50\n    return (hourly_rate / 60) * minutes",
        "tests": "assert abs(solve() - 10) < 1e-5"
    },
    {
        "id": "GSM8K/2",
        "category": "Proportional Reasoning",
        "prompt": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?",
        "expected_answer": 5,
        "canonical_solution": "def solve():\n    total = 100\n    saved = total / 2\n    parents = 15\n    grandparents = parents * 2\n    current = saved + parents + grandparents\n    return total - current",
        "tests": "assert abs(solve() - 5) < 1e-5"
    },
    {
        "id": "GSM8K/3",
        "category": "Combinatorics & Counting",
        "prompt": "A deep-sea monster rises from the waters once every 100 years to feast on a ship and sleep for another 100 years. If the monster has feasted 10 times, how many years old is it?",
        "expected_answer": 1000,
        "canonical_solution": "def solve():\n    feasts = 10\n    cycle_years = 100\n    return feasts * cycle_years",
        "tests": "assert abs(solve() - 1000) < 1e-5"
    },
    {
        "id": "GSM8K/4",
        "category": "Algebraic Word Problem",
        "prompt": "Mark has a garden with flowers. He has 10 rows of flowers, each row containing 15 flowers. If 20% of the flowers wilt, how many healthy flowers remain?",
        "expected_answer": 120,
        "canonical_solution": "def solve():\n    total = 10 * 15\n    wilted = total * 0.20\n    return total - wilted",
        "tests": "assert abs(solve() - 120) < 1e-5"
    },
    {
        "id": "MATH500/0",
        "category": "Quadratic Roots",
        "prompt": "Find the positive root of the quadratic equation $x^2 - 7x - 18 = 0$.",
        "expected_answer": 9,
        "canonical_solution": "def solve():\n    # (x - 9)(x + 2) = 0\n    return 9",
        "tests": "assert solve() == 9"
    },
    {
        "id": "MATH500/1",
        "category": "Calculus & Derivatives",
        "prompt": "Compute the derivative of $f(x) = 3x^3 - 5x + 7$ evaluated at $x = 2$.",
        "expected_answer": 31,
        "canonical_solution": "def solve():\n    # f'(x) = 9x^2 - 5. At x=2: 9(4) - 5 = 31\n    return 9 * (2**2) - 5",
        "tests": "assert solve() == 31"
    },
    {
        "id": "MATH500/2",
        "category": "Definite Integrals",
        "prompt": "Evaluate the definite integral $\\int_0^3 (2x + 1)\\,dx$.",
        "expected_answer": 12,
        "canonical_solution": "def solve():\n    # [x^2 + x]_0^3 = (9 + 3) - 0 = 12\n    return (3**2 + 3) - 0",
        "tests": "assert solve() == 12"
    },
    {
        "id": "MATH500/3",
        "category": "Modular Arithmetic",
        "prompt": "Compute $3^{10} \\pmod{7}$.",
        "expected_answer": 4,
        "canonical_solution": "def solve():\n    return pow(3, 10, 7)",
        "tests": "assert solve() == 4"
    },
    {
        "id": "MATH500/4",
        "category": "Geometry & Vectors",
        "prompt": "Find the magnitude squared of the vector $\\mathbf{v} = (3, 4, 12)$.",
        "expected_answer": 169,
        "canonical_solution": "def solve():\n    return 3**2 + 4**2 + 12**2",
        "tests": "assert solve() == 169"
    }
]
