def max_min_normalization(value):
    """Max-Min Normalization
    Formula: (raw - Min)/(Max - Min)
    :return range of values[0, 1]
    """
    new_value = 2*(value - value.min())/(value.max() - value.min()) - 1
    return new_value