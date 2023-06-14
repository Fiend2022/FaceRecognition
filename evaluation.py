def getDataOneWithMatch(filename:str):
    with open(filename, 'r') as file:
        text = file.readlines()
        count = len(text)
        lines = [l.rstrip() for l in text]
        data = []
        for i in lines:
            splitString = i.split(' ')
            template = splitString[1]
            condidate = splitString[0]
            label = int(splitString[2])
            data.append((template, condidate, label))
    return data, count

def checkLabelCorrect(pair:tuple[str, str, int]):
    class1 = int((pair[0].split('/', 1)[0]))
    class2 = int((pair[1].split('/', 1)[0]))
    if class1 == class2 and pair[2] == 1:
        return True
    elif class1 == class2 and pair[2] == 0:
        return False
    elif class1 != class2 and pair[2] == 1:
        return False
    elif class1 != class2 and pair[2] == 0:
        return True

def getDataBestComp(filename:str):
    with open(filename, 'r') as file:
        text = file.readlines()
        count = len(text)
        lines = [l.rstrip() for l in text]
        data = []
        for i in lines:
            splitString = i.split(' ')
            template = splitString[1]
            condidate = splitString[0]
            data.append((template, condidate))
    return data, count

def checkPairCorrect(pair:tuple[str, str]):
    class1 = int((pair[0].split('/', 1)[0]))
    class2 = int((pair[1].split('/', 1)[0]))
    if class1 == class2:
        return True
    else:
        return False




class Eval:
    def run(self, file1:str, file2:str):
        outputs, numOfTest = getDataOneWithMatch(file1)
        errors = 0
        for i in outputs:
            res = checkLabelCorrect(i)
            if res == False:
                errors += 1

        print("Probability of errors: ", errors / numOfTest, errors, " ", numOfTest)

        outputs, numOfTest = getDataBestComp(file2)
        errors = 0
        for i in outputs:
            res = checkPairCorrect(i)
            if res == False:
                errors += 1

        print("Probability of errors: ", errors / numOfTest, errors, " ", numOfTest)
        return None