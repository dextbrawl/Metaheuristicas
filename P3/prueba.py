import joblib
import numpy as np
import matplotlib.pyplot as plt
class BlackBoxModel:
    #Considerar blackbox_modelA y blackbox_modelB
    def __init__(self, path="blackbox_model.pkl"):
        self.model = joblib.load(path)
    def predict(self, x):
        x = np.array(x).reshape(1, -1)
        return self.model.predict(x)[0]

    
    
if __name__ == "__main__":
    model = BlackBoxModel("blackbox_modelB.pkl")
    x_red, x_blue = [], []
    y_red, y_blue = [], []

    for x in np.arange(-5.0, 5.0, 0.05):
        for y in np.arange(-5.0, 5.0, 0.05):
            category = model.predict([x, y])

            if category == 1:
                x_red.append(x)
                y_red.append(y)
            else:
                x_blue.append(x)
                y_blue.append(y)

    plt.figure(figsize=(8, 6))
    plt.scatter(x_red, y_red, color='red', s=20, label='Clase 1')
    plt.scatter(x_blue, y_blue, color='blue', s=20, label='Clase 0')

    plt.xlabel('Eje X')
    plt.ylabel('Eje Y')
    plt.title('Frontera de Decisión - Modelo A')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    plt.savefig('frontera_decisionB.png')
    plt.show()
    print(model.predict(x))