import pandas as pd
import json
import csv

with open('file2.csv', 'w') as csvfile:
    file2writer = csv.writer(csvfile, delimiter=',')
    
    with open('iclr-results-full.csv') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter='\t') 
    
        for line in csv_reader:
            dic = json.loads(line[1])
            try:
                for exact in dic.get('ies'):
                    temp = []
                    Uuid = json.loads(line[0]).get('uuid')
                    Text = exact.get('exact')
                    Id1 = exact.get('label').get('children')[0].get('value')
                    Id2 = exact.get('label').get('children')[0].get('children')[0].get('value')
                    Type = exact.get('label').get('value')

                    temp.append(Uuid)
                    temp.append(Text)
                    temp.append(Id1)
                    temp.append(Id2)
                    temp.append(Type)

                    file2writer.writerow(temp)
            except:
                None

with open('file1.csv', 'w') as csvfile:
    file1writer = csv.writer(csvfile, delimiter=',')
    
    with open('iclr-results-full.csv') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter='\t') 
        
        for line in csv_reader:
            try:
                Uuid = json.loads(line[0]).get('uuid')
                SrcText = json.loads(line[0]).get('srcText')
                Cat = json.loads(line[1]).get('classifications')[0].get('value')
                
                temp = []
                temp.append(Uuid)
                temp.append(SrcText)
                temp.append(Cat)

                file1writer.writerow(temp)
            except:
                None