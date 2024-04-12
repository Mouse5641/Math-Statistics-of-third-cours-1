import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from гістограма import to_np_array, build_plot
from первинний_аналіз_для_Х import create_new_window_for_x
from первинний_аналіз_для_У import create_new_window_for_y
from стохастичний_звязок import correlation_coefficient
from лінійна_регресі import linear_regression
from параболічна_регресія import parabolic_regression
from квазілінійна_регресія import quasi_linear_regression

content = ''


def open_file():
    global content
    file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
    if file_path:
        with open(file_path, 'r') as file:
            content = file.read()
            print(content)
            text_widget.delete("1.0", tk.END)
            text_widget.insert(tk.END, content)

            text_widget1.insert(tk.END, correlation_coefficient(eval(content)))
        data = to_np_array(content)
        if data is not None:
            image = Image.open(build_plot(data)[0])
            photo = ImageTk.PhotoImage(image)
            label.configure(image=photo)
            label.photo = photo

            image2 = Image.open(build_plot(data)[1])
            photo = ImageTk.PhotoImage(image2)
            label2.configure(image=photo)
            label2.photo = photo


# Створення вікна
window = tk.Tk()
window.geometry("1480x1300")  # Розміри вікна
window.title("Математична статистика")

label = tk.Label(window)
label.place(x=0, y=0)

label2 = tk.Label(window)
label2.place(x=850, y=0)

# Створення текстового віджету
text_widget = tk.Text(window, height=10, width=80)  # Встановлення розмірів
text_widget.place(x=0, y=650)

text_widget1 = tk.Text(window, height=10, width=100)
text_widget1.place(x=660, y=650)

# Створення стрічки меню
menubar = tk.Menu(window)

# Створення меню "Файл"
file_menu1 = tk.Menu(menubar, tearoff=0)
file_menu1.add_command(label="Обрати файл", command=open_file)
file_menu1.add_separator()
file_menu1.add_command(label="Вийти", command=window.quit)

file_menu2 = tk.Menu(menubar, tearoff=0)
file_menu2.add_command(label="для х", command=lambda: create_new_window_for_x(content))
file_menu2.add_separator()
file_menu2.add_command(label="для у", command=lambda: create_new_window_for_y(content))

# Приєднання меню "Файл" до стрічки меню
menubar.add_cascade(label="Файл", menu=file_menu1)
menubar.add_cascade(label="Первинний аналіз", menu=file_menu2)

# Встановлення стрічки меню для вікна
window.config(menu=menubar)


def on_button_click():
    selected_option = var.get()
    if selected_option in options:
        perform_action(selected_option)


def perform_action(selected_option):
    if selected_option in option_functions:
        option_functions[selected_option]()


# Створення змінної для варіанту вибору
var = tk.StringVar()
radiobutton1 = tk.Radiobutton(window, text="Лінійний", variable=var, value="Option 1")
radiobutton1.place(x=660, y=0)
radiobutton2 = tk.Radiobutton(window, text="Параболічний", variable=var, value="Option 2")
radiobutton2.place(x=660, y=30)
radiobutton3 = tk.Radiobutton(window, text="Квазілінійний", variable=var, value="Option 3")
radiobutton3.place(x=660, y=60)

button = tk.Button(window, text="Виконати дії", command=on_button_click)
button.place(x=660, y=90)


def action_for_option1():
    first, second = linear_regression(eval(content))[0], linear_regression(eval(content))[1]

    text_widget1.insert(tk.END, first)
    image = Image.open(second)
    photo = ImageTk.PhotoImage(image)
    label.configure(image=photo)
    label.photo = photo


def action_for_option2():
    first, second = parabolic_regression(eval(content))[0], parabolic_regression(eval(content))[1]

    text_widget1.insert(tk.END, first)
    image = Image.open(second)
    photo = ImageTk.PhotoImage(image)
    label.configure(image=photo)
    label.photo = photo


def action_for_option3():
    first, second = quasi_linear_regression(eval(content))[0], quasi_linear_regression(eval(content))[1]

    text_widget1.insert(tk.END, first)
    image = Image.open(second)
    photo = ImageTk.PhotoImage(image)
    label.configure(image=photo)
    label.photo = photo


options = ["Option 1", "Option 2", "Option 3"]
option_functions = {
    "Option 1": action_for_option1,
    "Option 2": action_for_option2,
    "Option 3": action_for_option3
}

# Запуск головного циклу
window.mainloop()

print(content)
