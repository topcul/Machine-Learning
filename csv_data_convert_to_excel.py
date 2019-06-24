import xlwt
book = xlwt.Workbook(encoding="utf-8")
sheet1 = book.add_sheet("Sheet 1")

f = open('./wdbc.csv')
row=1

for i,line in enumerate(f.readlines()):
    kayitYap=True
    currentline = line.split(",") #Veriler virgule ayrildigi icin virgule gore split edilir.
    print (currentline)
    
    for m in range(0,len(currentline)):
        if currentline[m]=='?' or currentline[m]==None or currentline[m]==' ' or currentline[m]=="" or currentline[m]==" ":
            kayitYap=False #.data ya da csv uzantýlý indirlen veri setinde herhangi bir yanlýslýk olursa o satýrý excel dosyasýna kaydetmiyor.
  
    if kayitYap==True:
        for column_no in range(0,len(currentline)):
            sheet1.write(row,column_no, currentline[column_no])     
        row+=1
with open('./wdbc.csv') as f:
    print ("Toplam:",sum(1 for _ in f),row)
f.close()
book.save("./wdbc.xls")