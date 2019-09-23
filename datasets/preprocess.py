# -*- coding: utf-8 -*-

import csv


# 让文本只保留汉字
def is_chinese(uchar):
    if uchar >= u'\u4e00' and uchar <= u'\u9fa5':
        return True
    else:
        return False

def format_str(content):
    content_str = ''
    for i in content:
        if is_chinese(i):
            content_str = content_str + ｉ
    return content_str

def process(filename):
    header = []
    rows = []
    with open(filename, 'r', encoding='utf-8-sig') as f:
        f_csv = csv.reader(f)
        for i, row in enumerate(f_csv):
            if i == 0:
                header = [row[0], row[2]] + row[3:]
                print(header)
            else:
                row = [row[0], row[2]] + row[3:]
                rows.append(row)
    
    with open('Clean_'+filename, 'w', encoding='utf-8-sig', newline='') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(header)
        for row in rows:
            f_csv.writerow(row)

if __name__ == '__main__':
    process('Train_Data.csv')
    process('Test_Data.csv')
    