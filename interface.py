import tkinter as tk
from tkinter import ttk
from tkinter import filedialog as fd
from tkinter.messagebox import showinfo
from tkinter import *

# create the root window
root = tk.Tk()
root.title('Reconnaissances des feuilles d\'arbres')
root.resizable(False, False)
root.geometry('600x750')


def select_file():
    filetypes = (
        ('picture', '*.jpg'),
        ('All files', '*.*')
    )

    filename = fd.askopenfilename(
        title='Open a file',
        initialdir='/',
        filetypes=filetypes)

    showinfo(
        title='Selected File',
        message=filename
    )


# open button
open_button = ttk.Button(
    root,
    text='Open a File',
    command=select_file
)

open_button.pack(expand=True)

# run the application
root.mainloop()
