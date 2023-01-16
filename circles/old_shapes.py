import numpy as np
import tkinter
import random as rnd
import time
import math
import threading
import colorsys

PIXEL_WIDTH  = 7680
PIXEL_HEIGHT = 4320
PIXEL_WIDTH  = 1920
PIXEL_HEIGHT = 1080
SHAPE_OUTLINE_WIDTH=1
MAX_SHRINK_FACTOR = 0.02
MIN_SHRINK_FACTOR = 0.4
MIN_RENDER_RADIUS = 1
ZOOM_FACTOR = 1.005
START_RADIUS = 1.0

short_edge = min(PIXEL_WIDTH, PIXEL_HEIGHT)

import time
def timed(func):
    def inner(*args, **kwargs):
        start_time = time.time()
        func(*args, **kwargs)
        print(func.__name__, time.time() - start_time)
    return inner

def ogl_point_to_pixel(point):
    """Converts a (2,) np array of OpenGL standard CoSy to left upper edge pixel coordinates"""

    
    x = short_edge/2 + point[0] * short_edge/2
    y = short_edge/2 + point[1] * short_edge/2

    if short_edge == PIXEL_HEIGHT:
        x += (PIXEL_WIDTH - short_edge) * 0.5
    else:
        y += (PIXEL_HEIGHT - short_edge) * 0.5

    return x, y

def pixel_to_ogl(x, y):
    px = (x - PIXEL_WIDTH/2) / short_edge * 2.0
    py = (y - PIXEL_HEIGHT/2) / short_edge * 2.0
    return np.array([px, py])

def ogl_mag_to_length(magnitude):
    """ Convert OGL vector magnitude to pixel length """
    # 1.0 should reach to border of shortest edge
    return short_edge/2 * magnitude

def length_to_ogl_mag(length):
    """ Convert OGL vector magnitude to pixel length """
    return length/(short_edge/2)

shape_outline_width_ogl = length_to_ogl_mag(SHAPE_OUTLINE_WIDTH)
min_render_radius_ogl = length_to_ogl_mag(MIN_RENDER_RADIUS)


def circle_to_bb(center_x, center_y, radius):
    return (center_x - radius, center_y - radius, center_x + radius, center_y + radius)

class Colorizer:
    def __init__(self):
        self.hue_palete = [i for i in range(0, 360, 12)]
        self.current_hue = 0

    def get_color(self, factor):
        """ expects a factor between 0.0 and 1.0 returns rgb hex"""
        rgb = colorsys.hsv_to_rgb(self.hue_palete[self.current_hue]/360, factor, 1.0)
        rgbhex = "#" + "".join("%02X" % round(i * 255) for i in rgb)
        return rgbhex

    def step(self):
        self.current_hue = (self.current_hue + 1) % len(self.hue_palete)


class Shape:
    """ The shapes our fractal is made of """
               
    all_shapes = []

    def __init__(self, parent, center, fill_color):
        if type(center) is not np.ndarray or np.shape(center) != (2,):
            raise ValueError
        self.center = center
        self.parent = parent
        self.fill_color = fill_color
        Shape.all_shapes.append(self)

    def render(self, canvas):
        raise NotImplementedError

    def is_inside(self, bounds):
        raise NotImplementedError

    def stuff(self):
        """ Stuff our shape with more shapes """
        raise NotImplementedError

    def zoom(self, factor, dest):
        raise NotImplementedError


class Circle(Shape):
    """ A circle shape """

    def __init__(self, parent, center, colorizer, radius):
        self.radius = radius
        self.colorizer = colorizer
        color_factor = 1.0 if parent is None else 1.0 -  self.radius / (parent.radius * MIN_SHRINK_FACTOR * 1.7)
        super().__init__(parent, center, fill_color = colorizer.get_color(color_factor))
        self.children = []

    def render(self, canvas, mark=False):

        # If the circle radius would be too small, skip drawing it
        if self.radius < min_render_radius_ogl:
            return
        if mark:
            canvas.create_oval(*circle_to_bb(*ogl_point_to_pixel(self.center), ogl_mag_to_length(self.radius)), fill = 'red', outline="red", width = SHAPE_OUTLINE_WIDTH)
        else:
            canvas.create_oval(*circle_to_bb(*ogl_point_to_pixel(self.center), ogl_mag_to_length(self.radius)), fill = self.fill_color, outline="red", width = SHAPE_OUTLINE_WIDTH)

    def is_inside(self, view_lu, view_ru, view_rl, view_ll):
        """ Expects bounds to be OGL coordinates of outer corners of view and returns false if object is 'out of view' """

        # Basically, determine if any border of the rectangle intersects with a circle line 
        corners_ordered = [view_lu, view_ru, view_rl, view_ll, view_lu]
        for a, b in zip(corners_ordered, corners_ordered[1:]):
            # Get Vectors M to A and a to b
            ma = self.center - a
            ab = b - a

            # Determine projection of ma on ab and then make this a global point by adding a
            d = a + (np.dot(ma, ab) / np.dot(ab, ab)) * ab
            ad = d - a

            # Determine wether d is on ab
            k = ad[0] / ab[0]
            if k <= 0.0:
                # Closest point of line to m is a
                dist_to_m = np.linalg.norm(a - self.center)
            elif k >= 1.0:
                # Closest point of line to m is b
                dist_to_m = np.linalg.norm(b - self.center)
            else:
                # Closest point to m is d
                dist_to_m = np.linalg.norm(d - self.center)
            
            if dist_to_m < self.radius:
                # A border intersects with the circle
                return True

        # Every circles point starts in the view, it should not get outside the view before radius is at least length of shortest dimension of box
        shortest_side = min(np.linalg.norm(view_lu - view_ru), np.linalg.norm(view_lu - view_ll))
        if self.radius <= shortest_side:
            return True
        
        return False


    def stuff(self):
        # rnd.seed(123231321231)

        stuffing = []
        # Progress color scheme
        self.colorizer.step()

        # print("AAAAAAAAAAAAAAAAAAAAAa")
        reps_stuck = 0
        while len(self.children) <= 70 and reps_stuck < 20:
            reps_stuck += 1
            magnitude = rnd.random() * (self.radius - MAX_SHRINK_FACTOR * self.radius)
            angle = rnd.random() * 2 * np.pi
            point = self.center + np.array([np.sin(angle) * magnitude, np.cos(angle) * magnitude])

            # For this point we find the biggest possible radius for a circle to be placed here
            # First limitation is the outer rim of self(parent), so smallest possible circle that does not intersect border of parent(self)
            smallest = self.radius - np.linalg.norm(self.center - point)
            # The new child should not be too big, so look at shrink factor
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
            self.children.append(Circle(self, point, self.colorizer, smallest))
            reps_stuck = 0
            # self.children.append(Circle(point, rnd.uniform(min_size_ogl, smallest)))

    def zoom(self, factor, dest):
        self.radius *= factor
        self.center = (self.center - dest) * factor + dest

    def pan(self, vector):
        self.center -= vector

    # Specfic functions
    def get_max_radius(self, point):
        """Returns the maximum radius a circle around point may take without intersecting me"""
        return np.linalg.norm(self.center - point) - self.radius



class FuzzyFractal:

    def __init__(self):
        # Init the Colorizer
        self.colorizer = Colorizer()
        # Generate the first shape
        first = Circle(None, np.array([0.0, 0.0]), radius=START_RADIUS, colorizer=self.colorizer)
        self.colorizer.step()
        # Make recursive fill until deepest visible level
        self.target = self.rec_stuff(first)
        # print("Center {}", self.target.center)

    def worth_stuffing(self, shape):
        """ Answers, wether shape is worth stuffing, aka if it's stuffing will even be rendered """
        # If the biggest circle that could be drawn in this is too small to be rendered, it's not worth generating stuffing
        return (shape.radius * MIN_SHRINK_FACTOR > min_render_radius_ogl)

    def rec_stuff(self, current_deepest):
        """ Recursively stuffs the current_deepest until invariant is broken, returns the next shape that was to be stuffed but is not yet """
        current = current_deepest
        while(True):
            current.stuff()
            current = rnd.choice(current.children)
            if self.worth_stuffing(current) == False:
                break
        return current

    def step(self):
        # Zoom in on current target
        # How many zoom steps are we away of target being size 1.0?
        # steps_away = 1.0 / ZOOM_FACTOR / self.target.radius 
        steps_away = math.log(START_RADIUS / self.target.radius, ZOOM_FACTOR)
        pan_step = self.target.center / steps_away 
        for s in Shape.all_shapes:
            # First pan
            # Then zoom
            s.zoom(ZOOM_FACTOR, self.target.center)
            # s.pan(pan_step)
        # Check wether more stuffing can be done, if yes, do so and select new target
        if self.worth_stuffing(self.target):
            self.target = self.rec_stuff(self.target)
            print("NEW TARGET")
            
    # @timed    
    def render_frame(self, canvas):
        viewport_corners = [pixel_to_ogl(0, 0), pixel_to_ogl(PIXEL_WIDTH, 0), pixel_to_ogl(PIXEL_WIDTH, PIXEL_HEIGHT), pixel_to_ogl(0, PIXEL_HEIGHT)]

        # Remove all shapes that are out of view
        Shape.all_shapes = list(filter(lambda s: s.is_inside(*viewport_corners), Shape.all_shapes))
        # print("Currently there are {} shapes in memory".format(len(Shape.all_shapes)))

        for s in Shape.all_shapes:
            s.render(canvas)

def loop(canvas, fractal, root_window):
    canvas.delete("all")
    step_thread = threading.Thread(target = fractal.step(), daemon=True)
    step_thread.start()
    fractal.render_frame(canvas)
    root_window.update()
    #time.sleep(0.005)
    step_thread.join()

if __name__ == '__main__':
    root_window = tkinter.Tk()
    canvas = tkinter.Canvas(master = root_window, width = PIXEL_WIDTH, height = PIXEL_HEIGHT, background = '#FFE7A1')
    canvas.pack()
    
    fractal = FuzzyFractal()
    while True:
        loop(canvas, fractal, root_window)

        

 


    

