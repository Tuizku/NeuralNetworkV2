import tkinter as tk
import math
import numpy as np
import neuralnetwork
from funcs import *


model = None # neuralnetwork.Model instance
root = None # Tkinter root window

# Tkinter widgets
canvas = None
epochs_entry = None
learning_rate_entry = None

# Settings
draw_enabled = True
max_neurons_normally = 8
min_neuron_radius_for_text = 16
neuron_hor_gap = 128

start_window_margin = 16
start_neuron_radius = 32
start_neuron_ver_gap = 16

window_margin = 0
neuron_radius = 0
neuron_ver_gap = 0


def open(_model):
    """
    Opens the debug window.

    Parameters:
    _model: Neuralnetwork Model instance
    """

    # Set the _model to the global model variable
    global model
    model = _model

    # Root window settings
    global root
    root = tk.Tk()
    root.title("Debug Neural Network V2 (by Tuisku Kahra)")
    root.geometry("1280x720")

    # --- TOP UI WIDGETS --- #
    top_widgets = []

    # Basic buttons
    top_widgets.append(TopWidget(tk.Button(root, text="draw toggle", command=draw_toggle), 100))
    top_widgets.append(TopWidget(tk.Button(root, text="random predict", command=predict_random), 100))
    top_widgets.append(TopWidget(tk.Button(root, text="randomize", command=randomize), 70))
    top_widgets.append(TopWidget(tk.Button(root, text="print activations", command=print_activations), 100))
    top_widgets.append(TopWidget(tk.Button(root, text="print weights", command=print_weights), 80))
    top_widgets.append(TopWidget(tk.Button(root, text="print biases", command=print_biases), 80))
    
    # Epochs Entry with Label
    top_widgets.append(TopWidget(tk.Label(root, text="epochs:"), 50))
    vcmd = (root.register(on_int_entry_validate), '%P')
    global epochs_entry
    epochs_entry = tk.Entry(root, validate="key", validatecommand=vcmd)
    top_widgets.append(TopWidget(epochs_entry, 40))
    
    # Learning rate Entry with Label
    top_widgets.append(TopWidget(tk.Label(root, text="learning rate:"), 80))
    vcmd2 = (root.register(on_float_entry_validate), '%P')
    global learning_rate_entry
    learning_rate_entry = tk.Entry(root, validate="key", validatecommand=vcmd2)
    top_widgets.append(TopWidget(learning_rate_entry, 40))

    # Training Buttons
    top_widgets.append(TopWidget(tk.Button(root, text="train", command=train), 50))
    top_widgets.append(TopWidget(tk.Button(root, text="predict single test", command=predict_single_test), 120))

    # Save/Load Model Button
    top_widgets.append(TopWidget(tk.Button(root, text="save model", command=save_model), 100))
    top_widgets.append(TopWidget(tk.Button(root, text="load model", command=load_model), 100))

    # Place widgets
    next_x = 0
    for i in range(len(top_widgets)):
        top_widget = top_widgets[i]
        top_widget.widget.place(x=next_x, y=0, width=top_widget.width, height=30)
        next_x += top_widget.width

    # Canvas
    global canvas
    canvas = tk.Canvas(root, bg="black", width=1280, height=690)
    canvas.place(x=0, y=30, width=1280, height=690)

    # Find out the max neuron count in a layer
    max_neuron_count = 0
    for i in range(len(model.layers)):
        neuron_count = len(model.layers[i].neurons)
        if neuron_count > max_neuron_count:
            max_neuron_count = neuron_count

    update_settings(max_neuron_count)
    update_canvas()
    root.mainloop()

def update_settings(max_neuron_count):
    """
    Updates the settings that manage the drawing of the neural network.

    Parameters:
    max_neuron_count (int): This is used to scale the neuralnetwork on the canvas.
    """
    
    global root
    global draw_enabled
    global window_margin
    global neuron_radius
    global neuron_ver_gap

    # If the neural network is too big, the drawing will be disabled when the window opens. It can still be enabled from the button.
    if max_neuron_count > 32 and draw_enabled:
        draw_toggle()

    # Change the drawing settings based on the divider
    divider = max(0.5, max_neuron_count / max_neurons_normally)
    window_margin = start_window_margin / divider
    neuron_radius = start_neuron_radius / divider
    neuron_ver_gap = start_neuron_ver_gap / divider

def update_canvas():
    """
    Updates the canvas that draws the neural network.
    """

    if not draw_enabled or np.any(model.layers[0].activations == None):
        return

    # Draw calls
    for x in range(len(model.layers)):
        layer = model.layers[x]
        for y in range(len(layer.neurons)):
            # Draw the neuron
            xPos, yPos = calculate_neuron_pos(x, y)
            draw_neuron(canvas, xPos, yPos, layer.activations[y])

            # Draw the connections between the neurons if current layer isn't the input layer
            if x > 0:
                for y2 in range(len(model.layers[x - 1].neurons)):
                    last_layer_neuron_xPos, last_layer_neuron_yPos = calculate_neuron_pos(x - 1, y2)
                    impact = max(min((layer.weights[y][y2] + 1) * 0.5, 1.0), 0.0)
                    draw_connection(canvas, last_layer_neuron_xPos + neuron_radius, last_layer_neuron_yPos, xPos - neuron_radius, yPos, impact)


#region Drawing Functions
def draw_neuron(canvas, x, y, activation):
    """
    Draws the single neuron to (x, y) position. The activation is used to choose the color of the neuron's fill.
    """

    # Choose the fill color from a "red-green gradient"
    red = int(min(1, 2 - (activation * 2)) * 225)
    green = int(min(1, activation * 2) * 225)

    # Create the neuron circle and the text if there is space for it
    canvas.create_oval(x - neuron_radius, y - neuron_radius, x + neuron_radius, y + neuron_radius, fill=f"#{red:02x}{green:02x}00", outline="white", width=2)
    if neuron_radius >= min_neuron_radius_for_text:
        canvas.create_text(x, y, text=f"{round(activation, 3)}")

def draw_connection(canvas, x1, y1, x2, y2, impact):
    """
    Draws the connection line between 2 neurons.

    Parameters:
    x1, y1 (int): The most right point on the neuron in an earlier layer
    x2, y2 (int): The most left point on the neuron in this layer
    """

    # Choose the line fill color between a "red-green gradient"
    red = int(min(1, 2 - (impact * 2)) * 225)
    green = int(min(1, impact * 2) * 225)

    # Create the line
    canvas.create_line(x1, y1, x2, y2, fill=f"#{red:02x}{green:02x}00", width=2)

def calculate_neuron_pos(xIndex, yIndex):
    """
    Calculates the neuron position based on the layer index (xIndex) and the neuron index (yIndex)
    """

    xPos = (xIndex * (neuron_radius * 2)) + (xIndex * neuron_hor_gap) + neuron_radius + window_margin
    yPos = (yIndex * (neuron_radius * 2)) + (yIndex * neuron_ver_gap) + neuron_radius + window_margin
    return xPos, yPos
#endregion


#region Widget Functions
class TopWidget:
    def __init__(self, widget, width):
        self.widget = widget
        self.width = width

def on_int_entry_validate(P):
    if P.isdigit() or P == "":  # Allow only digits and empty string
        return True
    else:
        return False

def on_float_entry_validate(P):
    if P == "" or P == ".":
        return True
    try:
        float(P)
        return True
    except ValueError:
        return False

def draw_toggle():
    global draw_enabled
    draw_enabled = not draw_enabled

    if draw_enabled:
        root.geometry("1280x720")
        update_canvas()
    else:
        root.geometry("1280x30")

def predict_random():
    model.predict_random()
    update_canvas()

def randomize():
    model.randomize()
    model.predict_random()
    update_canvas()

def print_weights():
    print("\n--- weights ---")
    for i in range(len(model.layers)):
        print(f"layer {i}:\n{model.layers[i].weights}")

def print_biases():
    print("\n--- biases ---")
    for i in range(len(model.layers)):
        print(f"layer {i}:\n{model.layers[i].biases}")

def print_activations():
    print("\n--- activations ---")
    for i in range(len(model.layers)):
        print(f"layer {i}: {model.layers[i].activations}")

def predict_single_test():
    if not np.any(model.x_test) or not np.any(model.y_test):
            warning("no test data. couldn't test")
            return

    tests_len = len(model.x_test)
    test_i = np.random.randint(0, tests_len)
    model.predict(model.x_test[test_i])
    update_canvas()

def train():
    # Tries to get the training entry values from widgets
    epochs = 0
    learning_rate = 0
    try:
        epochs = int(epochs_entry.get())
        learning_rate = float(learning_rate_entry.get())
    except:
        warning("couldn't start training, epochs or learning_rate were invalid")
        return
    
    # Train and update canvas
    model.train(epochs, learning_rate)
    update_canvas()

def save_model():
    neuralnetwork.save_model(model)

def load_model():
    global model
    model = neuralnetwork.load_model(model.name)
#endregion