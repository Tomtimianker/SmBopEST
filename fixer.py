import re

def fix_sentance(s):
    a = re.search('JOIN (.*) ON (.*) = (.*)\t', s)
    if (a is None):
        return s
    if not (s[a.regs[1][0]:a.regs[1][1]] == s[a.regs[3][0] : a.regs[3][1]]):
        return fix_double_join(s)
    start = s[:a.regs[0][0]]
    end = '\t' + s[a.regs[0][1] : ]
    return start + " WHERE " + s[a.regs[2][0] : a.regs[2][1]] + ' = "' + s[a.regs[1][0]:a.regs[1][1]] + '"' + end

def fix_double_join(s):
    a = re.search('JOIN.*(JOIN (.*) ON (.*) = (.*)\t)', s)
    if (a is None):
        return s
    if not (s[a.regs[2][0]:a.regs[2][1]] == s[a.regs[4][0] : a.regs[4][1]]):
        return s
    start = s[:a.regs[1][0]]
    end = '\t' + s[a.regs[1][1] : ]
    return start + " WHERE " + s[a.regs[3][0] : a.regs[3][1]] + ' = "' + s[a.regs[2][0]:a.regs[2][1]] + '"' + end

def write_new_file():
    with open('predictions_with_vals.txt') as input:
        with open('predictions_with_vals_fixed.txt', 'w') as output:
            for line in input:
                fixed = fix_sentance(line)
                output.write(fixed)
                print(fixed, end = '')


write_new_file()