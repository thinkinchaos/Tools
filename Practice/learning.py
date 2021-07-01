def lengthOfLongestSubstring(s):
    max_length = 0
    tmp = []
    for i in range(len(s)):
        if s[i] not in tmp:
            tmp.append(s[i])
        else:
            tmp.clear()
            tmp.append(s[i])
        length = len(tmp)
        if max_length < length:
            max_length = length
    return max_length


if __name__ == '__main__':
    ret = lengthOfLongestSubstring('bbbbb')
    print(ret)
