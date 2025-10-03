def check(s, filename):
    words = s.split()
    
    count_dict = {}
    
    for word in words:
        lower_word = word.lower()
        count_dict[lower_word] = count_dict.get(lower_word, 0) + 1
    
    sorted_words = sorted(count_dict.items(), key=lambda x: x[0])
    
    with open(filename, 'w', encoding='utf-8') as f:
        for word, count in sorted_words:
            f.write(f"{word} {count}\n")