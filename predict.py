# face verification with the VGGFace2 model
from matplotlib import pyplot
from PIL import Image
from numpy import asarray
from scipy.spatial.distance import *
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity
dataDir = 'preprocessedAnimals/'
# extract a single face from a given photograph
def extract_face(filename, required_size=(224, 224)):
    # load image from file
    pixels = Image.open(dataDir + filename)
    face = pixels.resize(required_size)
    face_array = asarray(face)
    return face_array



# extract faces and calculate face embeddings for a list of photo files
def get_embeddings(filenames):
    # extract faces
    faces = [extract_face(f) for f in filenames]
    # convert into an array of samples
    samples = asarray(faces, 'float32')
    # prepare the face for the model, e.g. center pixels
    samples = preprocess_input(samples, version=1)
    # create a vggface model
    model = VGGFace(model='vgg16', include_top=False, input_shape=(224, 224, 3), pooling='avg')
    # perform prediction
    yhat = model.predict(samples)
    return yhat


# determine if a candidate face is a match for a known face
def is_match(known_embedding, candidate_embedding, thresh=0.10):
    # calculate distance between embeddings
    score = cosine(known_embedding, candidate_embedding)
    if score > thresh:
        return 0, score
    else:
        return 1, score


# define filenames
templateFilenames = []
exampleFilenames = []

with open('template.txt', 'r') as file1:
    lines = file1.readlines()
    lines = [l.rstrip() for l in lines]
    for line in lines:
        #line = 'animal/' + line
        templateFilenames.append(line)
with open('example.txt', 'r') as file2:
    lines = file2.readlines()
    lines = [l.rstrip() for l in lines]
    for line in lines:
        #line = 'animal/' + line
        exampleFilenames.append(line)

# get embeddings file filenames
templateEmbeddings = get_embeddings(templateFilenames)
exampleEmbeddings = get_embeddings(exampleFilenames)

templatesData = []

for temp, file in zip(templateEmbeddings, templateFilenames):
    templatesData.append((temp, file))

examplesData = []
for example, file in zip(exampleEmbeddings, exampleFilenames):
        examplesData.append((example, file))

results = []

with open("VGG16OneWithAll.txt", "w") as resultFile:
    for face in examplesData:
        metaData = []
        for template in templatesData:
            label, score = is_match(template[0], face[0])
            resultFile.write(template[1] + " " + face[1] + " " + str(label) + "\n")
            md = (template[1], score, label)
            metaData.append(md)
        res = (face[1], metaData)
        results.append(res)

with open("VGG16BestComplience.txt", "w") as resultFile:
    for example in results:
        data = example[1]
        complience = 1
        bestTemplate = ""
        for info in data:
            currentComp = info[1]
            currentTemplate = info[0]
            if (currentComp < complience):
                complience = currentComp
                bestTemplate = currentTemplate
        resultFile.write(example[0] + " " + bestTemplate + "\n")