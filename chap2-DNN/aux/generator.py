import numpy as np

def generate_data():
    Age = np.arange(30, 100, 1)
    Gender = [0, 1]
    Years = np.arange(0, 40, 1)

    age = np.random.choice(Age)
    gender = np.random.choice(Gender)
    years = min(np.random.choice(Years), age - 10)

    # self-defined policy
    if gender == 1:
        if years > 10:
            if age > 50:
                label = np.random.random() > 0.5 - years * 0.01
            elif age > 40:
                label = np.random.random() > 0.65 - years * 0.01
            else:
                label = np.random.random() > 0.8 - years * 0.01
        elif years > 5:
            if age > 50:
                label = np.random.random() > 0.7 - years * 0.005
            elif age > 40:
                label = np.random.random() > 0.8 - years * 0.005
            else:
                label = np.random.random() > 0.9 - years * 0.005
        else:
            label = np.random.random() > 0.97
    
    else:
        if years > 10:
            if age > 50:
                label = np.random.random() > 0.5 - years * 0.01 + 0.5
            elif age > 40:
                label = np.random.random() > 0.65 - years * 0.01 + 0.5
            else:
                label = np.random.random() > 0.8 - years * 0.01 + 0.5
        elif years > 5:
            if age > 50:
                label = np.random.random() > 0.7 - years * 0.005 + 0.5
            elif age > 40:
                label = np.random.random() > 0.8 - years * 0.005 + 0.5
            else:
                label = np.random.random() > 0.9 - years * 0.005 + 0.5
        else:
            label = np.random.random() > 0.98

    return age, gender, years, label


# for training
training = []
with open("training.txt", "w") as f:
    for _ in range(5000):
        age, gender, years, label = generate_data()
        print ("{},{},{},{}".format(age, gender, years, label), file=f)
        training.append([age, gender, years, label])
training = np.array(training)

# for val
val = []
with open("validation.txt", "w") as f:
    for _ in range(800):
        age, gender, years, label = generate_data()
        print ("{},{},{},{}".format(age, gender, years, label), file=f)
        val.append([age, gender, years, label])
val = np.array(val)

# for training
test = []
with open("test.txt", "w") as f:
    for _ in range(500):
        age, gender, years, label = generate_data()
        print ("{},{},{},{}".format(age, gender, years, label), file=f)
        test.append([age, gender, years, label])
test = np.array(test)


# use sklearn to test
from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(random_state=1, max_iter=300).fit(training[:, :3], training[:, -1])
print (clf.score(test[:, :3], test[:, -1]))
print (clf.score(val[:, :3], val[:, -1]))
 

