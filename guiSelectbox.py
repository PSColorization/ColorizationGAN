import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()

x = filedialog.askopenfile(mode="r", master=root)
print(x.name)
