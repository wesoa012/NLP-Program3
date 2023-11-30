from os import listdir
def dataAdder():
    for file in listdir('./Data/adds'):
        if("Conviction" in file):
            AddToFile = open("./Data/txts/Conviction.txt", "a")
        elif("Thorn" in file):
            AddToFile = open("./Data/txts/Thorn.txt", "a")
        elif("Rubbish" in file):
            AddToFile = open("./Data/txts/Rubbish.txt", "a")
        else:
            print("not for this program")
        
        ReadFromFile = open("./Data/adds/"+file, "r")
        if("1.txt" in file):
            sense = "1"
        elif("2.txt" in file):
            sense = "2"
        else:
            print("Extra File Detected...skipping it")
            continue
        AddToFile.write('\n')
        for line in ReadFromFile.readlines():
            AddToFile.write(sense + " " + line)
        
    
    
def dataToCSVs():
    for file in listdir('./Data/txts'):
        csv = open("./Data/csvs/" + file[0:-4] + ".csv", "w")
        txt = open("./Data/txts/" + file, "r")
        
        for line in txt.readlines():
            line = line.replace('"',"'")
            csv.write(str(int(line[0]) - 1) + ',"' + line[2:-1] + '"' + '\n')
    
dataAdder()
dataToCSVs()