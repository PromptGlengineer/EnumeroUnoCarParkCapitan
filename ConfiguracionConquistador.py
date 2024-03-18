import tkinter as tk
from PIL import ImageTk, Image

class ConfiguracionConquistador:
    def __init__(self, master):
        self.master = master
        self.canvas = tk.Canvas(self.master, width=500, height=500)
        self.canvas.pack()

        # Load the image
        self.image = Image.open("a.png")
        self.photo = ImageTk.PhotoImage(self.image)
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        # Bind the draw function to the left mouse button click
        self.canvas.bind("<Button-1>", self.draw)

        # Initialize the line coordinates
        self.line_coords = []
        self.last_coords = None

    def draw(self, event):
        # If a point has been clicked before, draw the line on the canvas
        if self.last_coords:
            self.canvas.create_line(self.last_coords, (event.x, event.y), fill="black")
            self.line_coords.append({"name": f"line{len(self.line_coords) + 1}", "x1": self.last_coords[0], "y1": self.last_coords[1], "x2": event.x, "y2": event.y})

        # Save the coordinates of the current click
        self.last_coords = (event.x, event.y)

        # Print the line coordinates
        print(self.line_coords)

root = tk.Tk()
app = ConfiguracionConquistador(root)
root.mainloop()