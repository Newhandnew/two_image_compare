import tkinter as tk
from tkinter import filedialog


def click_ok():
    # global count
    # count = count + 1
    # label.configure(text="Click OK " + str(count) + " times")
    file = filedialog.askopenfile(parent=root, mode='rb', title='Choose a file')
    if file:
        data = file.read()
        # print(file.name)
        file.close()
        # print("I got %d bytes from this file." % len(data))
        return file.name
    else:
        print("No file!")


root = tk.Tk()
root.title("Tk GUI")
label = tk.Label(root, text="Hello World!")
widget = tk.Label(root, compound='top')
# widget.image = tk.PhotoImage(file=file)

count = 0

button = tk.Button(root, text="OK", command=click_ok)

label.pack()
button.pack()

root.mainloop()