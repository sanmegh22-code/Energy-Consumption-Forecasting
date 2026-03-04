import matplotlib.pyplot as plt

def create_prediction_graph(value):

    plt.figure()

    plt.bar(["Predicted Energy"], [value])

    plt.ylabel("Energy (Wh)")

    plt.title("Predicted Appliance Energy Consumption")

    plt.savefig("prediction_graph.png")

    plt.close()


def create_training_graph(actual, predicted):

    plt.figure()

    plt.plot(actual, label="Actual")

    plt.plot(predicted, label="Predicted")

    plt.title("Actual vs Predicted Energy Consumption")

    plt.xlabel("Samples")

    plt.ylabel("Energy (Wh)")

    plt.legend()

    plt.savefig("training_graph.png")

    plt.close()