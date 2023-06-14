from evaluation import Eval
from predict import Predict
from preprocessing import Preprocessing

if __name__ == "__main__":
    prep = Preprocessing()
    #prep.run('animal', 'preprocessedAnimals')
    pred = Predict()
    pred.run('preprocessedAnimals/', 'template.txt', 'example.txt')
    eval = Eval()
    eval.run('VGG16OneWithAll.txt', 'VGG16BestComplience.txt')
