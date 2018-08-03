import tkinter as tk
from tkinter import filedialog


class Window(object):
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Tk GUI")
        self.label = tk.Label(self.root, text="Hello World!", font=('Helvetica', '12'))
        widget = tk.Label(self.root, compound='top')
        self.btn_load = tk.Button(self.root, text="Load", command=self.click_load, font=('Helvetica', '18'))
        self.root.geometry("500x100")
        self.root.resizable(width=False, height=False)

    def click_load(self):
        file = filedialog.askopenfile(parent=self.root, mode='rb', title='Choose a file')
        if file:
            self.label.config(text=file.name)
            data = file.read()

            file.close()
            return file.name
        else:
            self.label.config(text="No file!")

    def run(self):
        self.label.pack()
        self.btn_load.pack()
        self.root.mainloop()


def main():
    window = Window()
    window.run()


if __name__ == '__main__':
    main()