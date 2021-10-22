import math

def floor_a_by_b(a, b):
    return int(float(a) / b)

def ceil_a_by_b(a, b):
    return int(math.ceil(float(a) / b))


def log2(a):
    return math.log(a) / math.log(2)

def lookup_pandas_dataframe(data, lookup_dict):
    '''
    Lookup a pandas dataframe using a key-value dict
    '''
    data = data.drop_duplicates()
    for key in lookup_dict:
        data = data.loc[data[key] == lookup_dict[key]]

    # assert len(data) == 1, ("Found {} entries for dict {}".format(len(data), lookup_dict))
    return data



def is_power_two(n):
    if n == 2:
        return True
    elif n%2 != 0:
        return False
    else:
        return is_power(n/2.0)


def largest_power_of_two_divisor(num):
    if num % 2 != 0: return 1
    factor = 0
    while num % 2 == 0:
        num /= 2
        factor += 1
    return 2 ** factor


def get_two_largest_divisor(n):
    div_list = []
    for i in range(1, int(n/2)+1):
        if n % i == 0:
            if len(div_list) == 0:
                div_list = [i, n/i]
            else:
                if abs(i-n/i) < abs(div_list[0]-div_list[1]):
                    div_list = [i, n/i]
    if n == 1:
        return [1,1]
    return div_list
