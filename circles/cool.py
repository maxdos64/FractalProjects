import numpy as np
import tkinter
import random as rnd
import time

PIXEL_WIDTH  = 2000
PIXEL_HEIGHT = 1500
SHAPE_OUTLINE_WIDTH=3
MAX_SHRINK_FACTOR = 0.04
MIN_SHRINK_FACTOR = 0.4
MIN_FILL_RADIUS = 20
ZOOM_FACTOR = 0.1

def ogl_point_to_pixel(point):
    """Converts a (2,) np array of OpenGL standard CoSy to left upper edge pixel coordinates"""

    
    x = short_edge/2 + point[0] * short_edge/2
    y = short_edge/2 + point[1] * short_edge/2

    if short_edge == PIXEL_HEIGHT:
        x += (PIXEL_WIDTH - short_edge) * 0.5
    else:
        y += (PIXEL_HEIGHT - short_edge) * 0.5

    return x, y

def ogl_mag_to_length(magnitude):
    """ Convert OGL vector magnitude to pixel length """
    # 1.0 should reach to border of shortest edge
    return short_edge/2 * magnitude

def length_to_ogl_mag(length):
    """ Convert OGL vector magnitude to pixel length """
    return length/(short_edge/2)

short_edge = min(PIXEL_WIDTH, PIXEL_HEIGHT)
shape_outline_width_ogl = length_to_ogl_mag(SHAPE_OUTLINE_WIDTH)
min_fill_radius_ogl = length_to_ogl_mag(MIN_FILL_RADIUS)


def circle_to_bb(center_x, center_y, radius):
    return (center_x - radius, center_y - radius, center_x + radius, center_y + radius)

class Shape:
    """ The shapes our fractal is made of """
               
    all_shapes = []

    def __init__(self, center):
        if type(center) is not np.ndarray or np.shape(center) != (2,):
            raise ValueError
        self.center = center
        Shape.all_shapes.append(self)

    def render(self, canvas):
        raise NotImplementedError

    def is_inside(self, bounds):
        raise NotImplementedError

    def stuff(self):
        """ Stuff our shape with more shapes """
        raise NotImplementedError

    def zoom(self, factor):
        raise NotImplementedError


class Circle(Shape):
    """ A circle shape """

    def __init__(self, center, radius):
        super().__init__(center)
        self.radius = radius
        self.children = []

    def render(self, canvas):
        canvas.create_oval(*circle_to_bb(*ogl_point_to_pixel(self.center), ogl_mag_to_length(self.radius)), fill = '', outline="red", width = SHAPE_OUTLINE_WIDTH)

    def stuff(self):
        rnd.seed(123511231321231)

        stuffing = []

        while len(self.children) <= 50:
            magnitude = rnd.random() * (self.radius - MAX_SHRINK_FACTOR * self.radius)
            angle = rnd.random() * 2 * np.pi
            point = self.center + np.array([np.sin(angle) * magnitude, np.cos(angle) * magnitude])

            # check if this is inside any item already
            # First limitation is the outer rim of self(parent), so get dist to border
            smallest = self.radius - np.linalg.norm(self.center - point) - shape_outline_width_ogl
            # It should not too big, so look at shrink factor
            smallest = min(smallest, MIN_SHRINK_FACTOR * self.radius)
            for c in self.children:
                r = c.get_max_radius(point)
                if(r < smallest):
                    smallest = r
                if r <= MAX_SHRINK_FACTOR * self.radius:
                    break
            # Is this point already inside/too close to a child?
            if(smallest <= MAX_SHRINK_FACTOR * self.radius):
                continue
            # If not, put it in it!
            self.children.append(Circle(point, smallest))
            # self.children.append(Circle(point, rnd.uniform(min_size_ogl, smallest)))

    def zoom(self, factor):
        self.center *= (1.0+factor)

    def pan(self, vector):
        print(self.center)
        self.center -= vector

    # Specfic functions
    def get_max_radius(self, point):
        """Returns the maximum radius a circle around point may take without intersecting me"""
        return np.linalg.norm(self.center - point) - self.radius




class FuzzyFractal:

    def __init__(self):
        # Generate the first shape
        first = Circle(np.array([0.0, 0.0]), radius=1.0)
        # Make recursive fill until deepest visible level
        self.target = self.stuffing(first)

    def stuffing(self, current_deepest):
        current = current_deepest
        while(True):
            current.stuff()
            # What was the biggest circle drawn?
            biggest_drawn = max(current.children, key=lambda item : item.radius).radius
            if(biggest_drawn) < min_fill_radius_ogl:
                break
            current = rnd.choice(current.children)
        return current

    def step(self):
        # Zoom in on current target
        # How many zoom steps are we away of target being size 1.0?
        steps_away = 1.0 / ZOOM_FACTOR / self.target.radius 
        pan_step = self.target.center / steps_away 
        print(pan_step)
        for s in Shape.all_shapes:
            # First pan
            s.pan(pan_step)
            # Then zoom
            s.zoom(ZOOM_FACTOR)
            
        
    def render_frame(self, canvas):
        for s in Shape.all_shapes:
            s.render(canvas)

if __name__ == '__main__':
    root_window = tkinter.Tk()
    canvas = tkinter.Canvas(master = root_window, width = PIXEL_WIDTH, height = PIXEL_HEIGHT, background = '#FFE7A1')
    canvas.pack()
    
    fractal = FuzzyFractal()

    while(True):
        fractal.render_frame(canvas)
        root_window.update()
        time.sleep(0.1)
        fractal.step()

 


    

