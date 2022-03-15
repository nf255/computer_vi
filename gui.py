from tkinter import *

root = Tk()


def myClick():
    myLabel3 = Label(root, text="I was clicked!")
    myLabel3.grid(row=3, column=1)


# Creating a Label Widget and a button
myLabel1 = Label(root, text="Hello World!")
myLabel2 = Label(root, text="This is a test!")
myButton1 = Button(root, text="Click Me!", padx=50, pady=50, command=myClick)
myButton2 = Button(root, text="Click Me!", state=DISABLED)


# Shoving it onto the screen
myLabel1.grid(row=0, column=0)
myLabel2.grid(row=1, column=1)
myButton1.grid(row=2, column=0)
myButton2.grid(row=2, column=1)

root.mainloop()
