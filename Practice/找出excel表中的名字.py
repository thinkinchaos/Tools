import xlrd

book = xlrd.open_workbook('file-tools/1.xlsx')

sheet1 = book.sheets()[0]

A = sheet1.col_values(0)  # A
B = sheet1.col_values(2)  # B
C = sheet1.col_values(4)  # C

names = set()
for i in A:
    if i in B or i in C:
        names.add(i)

for i in names:
    print(i)
