import pygame
import numpy as np
import cv2
import kociemba
import threading
import time
import socket
import sys
import math

pygame.init()
W, H = 1600, 900
screen = pygame.display.set_mode((W, H))
myfont = pygame.font.SysFont('Arial', 30)
textsurface = myfont.render('Top face', False, (255, 255, 255))
textsurface2 = myfont.render('Full cube', False, (255, 255, 255))
done = False

turn_r_clock_msg = myfont.render('Turn the right side of the cube towards yourself.', False, (255, 255, 255))
turn_r_cclock_msg = myfont.render('Turn the right side of the cube away from yourself.', False, (255, 255, 255))

turn_l_clock_msg = myfont.render('Turn the left side of the cube away from yourself.', False, (255, 255, 255))
turn_l_cclock_msg = myfont.render('Turn the left side of the cube towards yourself.', False, (255, 255, 255))

turn_u_clock_msg = myfont.render('Turn the top face of the cube clockwise.', False, (255, 255, 255))
turn_u_cclock_msg = myfont.render('Turn the top face of the cube counterclockwise.', False, (255, 255, 255))

turn_x_clock_msg = myfont.render('Turn the entire cube away from yourself.', False, (255, 255, 255))
turn_x_cclock_msg = myfont.render('Turn the entire cube towards yourself.', False, (255, 255, 255))

turn_z_clock_msg = myfont.render('Turn the entire cube to your right.', False, (255, 255, 255))
turn_z_cclock_msg = myfont.render('Turn the entire cube to your left.', False, (255, 255, 255))

top_msg = myfont.render('Top', False, (255, 255, 255))
bottom_msg = myfont.render('Bottom', False, (255, 255, 255))
left_msg = myfont.render('Left', False, (255, 255, 255))
right_msg = myfont.render('Right', False, (255, 255, 255))
back_msg = myfont.render('Back', False, (255, 255, 255))
front_msg = myfont.render('Front', False, (255, 255, 255))

DISPLAY_COLOR_MAP = {
    "X": (0,0,0),
    " ": (255,0,255),
    "R": (255,0,0),
    "W": (255,255,255),
    "B": (0,0,255),
    "Y": (255,255,0),
    "G": (0,255,0),
    "O": (255,165,0)
}

class Cube:
    """
    A rubik's cube
    """

    def __init__(self):
        self.clear()

    def clear(self):
        self.up = np.full((3,3), " ")
        self.down = np.full((3,3), " ")
        self.left = np.full((3,3), " ")
        self.right = np.full((3,3), " ")
        self.front = np.full((3,3), " ")
        self.back = np.full((3,3), " ")

    def set_faces(self, up, down, left, right, front, back):
        self.up = up
        self.down = down
        self.left = left
        self.right = right
        self.front = front
        self.back = back

    def set_top_face_matrix(self, face):
        self.up = face

    def set_top_face_list(self, points):
        self.up[1,1] = points[0]
        self.up[0,2] = points[1]
        self.up[0,1] = points[2]
        self.up[0,0] = points[3]
        self.up[1,0] = points[4]
        self.up[2,0] = points[5]
        self.up[2,1] = points[6]
        self.up[2,2] = points[7]
        self.up[1,2] = points[8]


    def get_top(self):
        return self.up

    def check_top_face_list(self, points):
        return (self.up[1,1] == points[0]) and (self.up[0,2] == points[1]) and (self.up[0,1] == points[2]) and (self.up[0,0] == points[3]) and (self.up[1,0] == points[4]) and (self.up[2,0] == points[5]) and (self.up[2,1] == points[6]) and (self.up[2,2] == points[7]) and (self.up[1,2] == points[8])

    def check_solved(self):
        for face in [self.up, self.down, self.left, self.right, self.front, self.back]:
            if not np.all(face == face[0,0]):
                return False
        return True

    def get_face(self, direction):
        if direction == "u":
            return self.up
        if direction == "l":
            return self.left
        if direction == "r":
            return self.right
        if direction == "f":
            return self.front
        if direction == "b":
            return self.back
        if direction == "d":
            return self.down
        return np.full((3,3), -1)

    def rotate_x_clock(self):
        new_up = np.rot90(self.front, 0)
        new_down = np.rot90(self.back, 2)
        new_left = np.rot90(self.left, 1)
        new_right = np.rot90(self.right, 3)
        new_front = np.rot90(self.down, 0)
        new_back = np.rot90(self.up, 2)

        self.set_faces(new_up, new_down, new_left, new_right, new_front, new_back)

    def rotate_x_counter(self):
        new_up = np.rot90(self.back, 2)
        new_down = np.rot90(self.front, 0)
        new_left = np.rot90(self.left, 3)
        new_right = np.rot90(self.right, 1)
        new_front = np.rot90(self.up, 0)
        new_back = np.rot90(self.down, 2)
        self.set_faces(new_up, new_down, new_left, new_right, new_front, new_back)

    def rotate_y_clock(self):
        new_up = np.rot90(self.up, 3)
        new_down = np.rot90(self.down, 1)
        new_left = np.rot90(self.front, 0)
        new_right = np.rot90(self.back, 0)
        new_front = np.rot90(self.right, 0)
        new_back = np.rot90(self.left, 0)
        self.set_faces(new_up, new_down, new_left, new_right, new_front, new_back)

    def rotate_y_counter(self):
        new_up = np.rot90(self.up, 1)
        new_down = np.rot90(self.down, 3)
        new_left = np.rot90(self.back, 0)
        new_right = np.rot90(self.front, 0)
        new_front = np.rot90(self.left, 0)
        new_back = np.rot90(self.right, 0)
        self.set_faces(new_up, new_down, new_left, new_right, new_front, new_back)

    def rotate_z_clock(self):
        new_up = np.rot90(self.left, 3)
        new_down = np.rot90(self.right, 3)
        new_left = np.rot90(self.down, 3)
        new_right = np.rot90(self.up, 3)
        new_front = np.rot90(self.front, 3)
        new_back = np.rot90(self.back, 1)
        self.set_faces(new_up, new_down, new_left, new_right, new_front, new_back)

    def rotate_z_counter(self):
        new_up = np.rot90(self.right, 1)
        new_down = np.rot90(self.left, 1)
        new_left = np.rot90(self.up, 1)
        new_right = np.rot90(self.down, 1)
        new_front = np.rot90(self.front, 1)
        new_back = np.rot90(self.back, 3)
        self.set_faces(new_up, new_down, new_left, new_right, new_front, new_back)

    def turn_u_clock(self):
        new_up = np.rot90(self.up, 3)
        new_down = self.down
        new_left = np.vstack((self.front[:1,:], self.left[1:,:]))
        new_right = np.vstack((self.back[:1,:], self.right[1:,:]))
        new_front = np.vstack((self.right[:1,:], self.front[1:,:]))
        new_back = np.vstack((self.left[:1,:], self.back[1:,:]))
        self.set_faces(new_up, new_down, new_left, new_right, new_front, new_back)

    def turn_u_counter(self):
        new_up = np.rot90(self.up, 1)
        new_down = self.down
        new_left = np.vstack((self.back[:1,:], self.left[1:,:]))
        new_right = np.vstack((self.front[:1,:], self.right[1:,:]))
        new_front = np.vstack((self.left[:1,:], self.front[1:,:]))
        new_back = np.vstack((self.right[:1,:], self.back[1:,:]))
        self.set_faces(new_up, new_down, new_left, new_right, new_front, new_back)

    def turn_d_clock(self):
        new_up = self.up
        new_down = np.rot90(self.down, 3)
        new_left = np.vstack((self.left[:2,:], self.back[2,:]))
        new_right = np.vstack((self.right[:2,:], self.front[2,:]))
        new_front = np.vstack((self.front[:2,:], self.left[2,:]))
        new_back = np.vstack((self.back[:2,:], self.right[2,:]))
        self.set_faces(new_up, new_down, new_left, new_right, new_front, new_back)

    def turn_d_counter(self):
        new_up = self.up
        new_down = np.rot90(self.down, 1)
        new_left = np.vstack((self.left[:2,:], self.front[2,:]))
        new_right = np.vstack((self.right[:2,:], self.back[2,:]))
        new_front = np.vstack((self.front[:2,:], self.right[2,:]))
        new_back = np.vstack((self.back[:2,:], self.left[2,:]))
        self.set_faces(new_up, new_down, new_left, new_right, new_front, new_back)

    def turn_l_clock(self):
        new_up = np.column_stack((np.rot90(self.back[:,2:],2),self.up[:,1:]))
        new_down = np.column_stack((self.front[:,:1], self.down[:,1:]))
        new_left = np.rot90(self.left, 3)
        new_right = self.right
        new_front = np.column_stack((self.up[:,:1], self.front[:,1:]))
        new_back = np.column_stack((self.back[:,:2],np.rot90(self.down[:,:1],2)))
        self.set_faces(new_up, new_down, new_left, new_right, new_front, new_back)

    def turn_l_counter(self):
        new_up = np.column_stack((self.front[:,:1],self.up[:,1:]))
        new_down = np.column_stack((np.rot90(self.back[:,2:],2), self.down[:,1:]))
        new_left = np.rot90(self.left, 1)
        new_right = self.right
        new_front = np.column_stack((self.down[:,:1], self.front[:,1:]))
        new_back = np.column_stack((self.back[:,:2],np.rot90(self.up[:,:1],2)))
        self.set_faces(new_up, new_down, new_left, new_right, new_front, new_back)

    def turn_r_clock(self):
        new_up = np.column_stack((self.up[:,:2],self.front[:,2:]))
        new_down = np.column_stack((self.down[:,:2],np.rot90(self.back[:,:1],2)))
        new_left = self.left
        new_right = np.rot90(self.right, 3)
        new_front = np.column_stack((self.front[:,:2], self.down[:,2:]))
        new_back = np.column_stack((np.rot90(self.up[:,2:],2), self.back[:,1:]))
        self.set_faces(new_up, new_down, new_left, new_right, new_front, new_back)

    def turn_r_counter(self):
        new_up = np.column_stack((self.up[:,:2],np.rot90(self.back[:,:1],2)))
        new_down = np.column_stack((self.down[:,:2],self.front[:,2:]))
        new_left = self.left
        new_right = np.rot90(self.right, 1)
        new_front = np.column_stack((self.front[:,:2], self.up[:,2:]))
        new_back = np.column_stack((np.rot90(self.down[:,2:],2), self.back[:,1:]))
        self.set_faces(new_up, new_down, new_left, new_right, new_front, new_back)

    def turn_f_clock(self):
        new_up = np.vstack((self.up[:2,:],np.rot90(self.left[:,2:],3)))
        new_down = np.vstack((np.rot90(self.right[:,:1],3),self.down[1:,:]))
        new_left = np.column_stack((self.left[:,:2],np.rot90(self.down[:1,:],3)))
        new_right = np.column_stack((np.rot90(self.up[2:,:],3),self.right[:,1:]))
        new_front = np.rot90(self.front,3)
        new_back = self.back
        self.set_faces(new_up, new_down, new_left, new_right, new_front, new_back)

    def turn_f_counter(self):
        new_up = np.vstack((self.up[:2,:],np.rot90(self.right[:,:1],1)))
        new_down = np.vstack((np.rot90(self.left[:,2:],1),self.down[1:,:]))
        new_left = np.column_stack((self.left[:,:2],np.rot90(self.up[2:,:],1)))
        new_right = np.column_stack((np.rot90(self.down[:1,:],1),self.right[:,1:]))
        new_front = np.rot90(self.front,1)
        new_back = self.back
        self.set_faces(new_up, new_down, new_left, new_right, new_front, new_back)

    def turn_b_clock(self):
        new_up = np.vstack((np.rot90(self.right[:,2:],1),self.up[1:,:]))
        new_down = np.vstack((self.down[:2,:],np.rot90(self.left[:,:1],1)))
        new_left = np.column_stack((np.rot90(self.up[:1,:],1),self.left[:,1:]))
        new_right = np.column_stack((self.right[:,:2],np.rot90(self.down[2:,:],1)))
        new_front = self.front
        new_back = np.rot90(self.back,3)
        self.set_faces(new_up, new_down, new_left, new_right, new_front, new_back)

    def turn_b_counter(self):
        new_up = np.vstack((np.rot90(self.left[:,:1],3),self.up[1:,:]))
        new_down = np.vstack((self.down[:2,:],np.rot90(self.right[:,2:],3)))
        new_left = np.column_stack((np.rot90(self.down[2:,:],3),self.left[:,1:]))
        new_right = np.column_stack((self.right[:,:2],np.rot90(self.up[:1,:],3)))
        new_front = self.front
        new_back = np.rot90(self.back,1)
        self.set_faces(new_up, new_down, new_left, new_right, new_front, new_back)


    def execute_move(self, move):
        moves = {
            "X": self.rotate_x_clock,
            "x": self.rotate_x_counter,
            "Y": self.rotate_y_clock,
            "y": self.rotate_y_counter,
            "Z": self.rotate_z_clock,
            "z": self.rotate_z_counter,
            "U": self.turn_u_clock,
            "u": self.turn_u_counter,
            "D": self.turn_d_clock,
            "d": self.turn_d_counter,
            "L": self.turn_l_clock,
            "l": self.turn_l_counter,
            "R": self.turn_r_clock,
            "r": self.turn_r_counter,
            "F": self.turn_f_clock,
            "f": self.turn_f_counter,
            "B": self.turn_b_clock,
            "b": self.turn_b_counter
        }
        moves[move]()

    def execute_string(self, move_string):
        for move in move_string:
            self.execute_move(move)

    def export_kociemba_string(self):
        koc_dict = {
            self.up[1,1]: "U",
            self.front[1,1]: "F",
            self.right[1,1]: "R",
            self.back[1,1]: "B",
            self.left[1,1]: "L",
            self.down[1,1]: "D"
        }

        out = ""
        for face in [self.up, self.right, self.front, self.down, self.left, self.back]:
            for row in face:
                for square in row:
                    out += koc_dict[square]
        return out

    def generate_solution(self):
        state_string = self.export_kociemba_string()
        sol_string = kociemba.solve(state_string)
        return sol_string


    def print_urf(self):
        print("      ", self.up[0,0], self.up[0,1], self.up[0,2], "  /")
        print("    ", self.up[1,0], self.up[1,1], self.up[1,2], "  /")
        print("  ", self.up[2,0], self.up[2,1], self.up[2,2], "  /  ", self.right[0,2])
        print("   - - - -  ", self.right[0,1], self.right[1,2])
        print("  ", self.front[0,0], self.front[0,1], self.front[0,2], "|", self.right[0,0], self.right[1,1], self.right[2,2])
        print("  ", self.front[1,0], self.front[1,1], self.front[1,2], "|", self.right[1,0], self.right[2,1])
        print("  ", self.front[2,0], self.front[2,1], self.front[2,2], "|", self.right[2,0])
        print()
        print("      ", self.down[0,2], self.down[1,2], self.down[2,2], "  /")
        print("    ", self.down[0,1], self.down[1,1], self.down[2,1], "  /")
        print("  ", self.down[0,0], self.down[1,0], self.down[2,0], "  /  ", self.back[2,0])
        print("   - - - -  ", self.back[2,1], self.back[1,0])
        print("  ", self.left[2,2], self.left[2,1], self.left[2,0], "|", self.back[2,2], self.back[1,1], self.back[0,0])
        print("  ", self.left[1,2], self.left[1,1], self.left[1,0], "|", self.back[1,2], self.back[0,1])
        print("  ", self.left[0,2], self.left[0,1], self.left[0,0], "|", self.back[0,2])

def draw_centered_square(screen, center, color, size):
    x, y = center
    h = size/2
    pygame.draw.rect(screen, color, pygame.Rect(x-h, y-h, size,size))

def draw_cube_face(screen, face, center, square_size=30, square_sep=38):
    draw_centered_square(screen, center, DISPLAY_COLOR_MAP[face[1,1]], square_size)
    draw_centered_square(screen, center - [square_sep, 0], DISPLAY_COLOR_MAP[face[1,0]], square_size)
    draw_centered_square(screen, center + [square_sep, 0], DISPLAY_COLOR_MAP[face[1,2]], square_size)
    draw_centered_square(screen, center - [0, square_sep], DISPLAY_COLOR_MAP[face[0,1]], square_size)
    draw_centered_square(screen, center + [0, square_sep], DISPLAY_COLOR_MAP[face[2,1]], square_size)

    draw_centered_square(screen, center + [square_sep, square_sep], DISPLAY_COLOR_MAP[face[2,2]], square_size)
    draw_centered_square(screen, center + [square_sep, -square_sep], DISPLAY_COLOR_MAP[face[0,2]], square_size)
    draw_centered_square(screen, center + [-square_sep, -square_sep], DISPLAY_COLOR_MAP[face[0,0]], square_size)
    draw_centered_square(screen, center + [-square_sep, square_sep], DISPLAY_COLOR_MAP[face[2,0]], square_size)

def draw_cube_net(screen, cube, center, square_size=30, square_sep=38, face_sep=140):
    draw_cube_face(screen, c.front, center, square_size, square_sep)
    draw_cube_face(screen, c.up, center-[0,face_sep], square_size, square_sep)
    draw_cube_face(screen, c.left, center-[face_sep,0], square_size, square_sep)
    draw_cube_face(screen, c.back, center-[2*face_sep,0], square_size, square_sep)
    draw_cube_face(screen, c.down, center+[0,face_sep], square_size, square_sep)
    draw_cube_face(screen, c.right, center+[face_sep,0], square_size, square_sep)

def draw_arrow(screen, color, start, end):
    pygame.draw.line(screen,color,start,end,6)
    rotation = math.degrees(math.atan2(start[1]-end[1], end[0]-start[0]))+90
    pygame.draw.polygon(screen, color, ((end[0]+20*math.sin(math.radians(rotation)), end[1]+20*math.cos(math.radians(rotation))), (end[0]+20*math.sin(math.radians(rotation-120)), end[1]+20*math.cos(math.radians(rotation-120))), (end[0]+20*math.sin(math.radians(rotation+120)), end[1]+20*math.cos(math.radians(rotation+120)))))

def turn_r_cclock(screen, center, square_size=30*4, square_sep=38*4, msg=True):
    r = square_sep + square_size/2
    draw_arrow(screen, (255,255,255), center + [r + 50, -r], center + [r + 50, r])
    if msg:
        screen.blit(turn_r_cclock_msg, tuple([50, H-100]))

def turn_r_clock(screen, center, square_size=30*4, square_sep=38*4, msg=True):
    r = square_sep + square_size/2
    draw_arrow(screen, (255,255,255), center + [r + 50, r], center + [r + 50, -r])
    if msg:
        screen.blit(turn_r_cclock_msg, tuple([50, H-100]))

def turn_l_cclock(screen, center, square_size=30*4, square_sep=38*4, msg=True):
    r = square_sep + square_size/2
    draw_arrow(screen, (255,255,255), center + [-r - 50, r], center + [-r - 50, -r])
    if msg:
        screen.blit(turn_l_cclock_msg, tuple([50, H-100]))


def turn_l_clock(screen, center, square_size=30*4, square_sep=38*4, msg=True):
    r = square_sep + square_size/2
    draw_arrow(screen, (255,255,255), center + [-r - 50, -r], center + [-r - 50, r])
    if msg:
        screen.blit(turn_l_clock_msg, tuple([50, H-100]))

def turn_f_cclock(screen, center, square_size=30*4, square_sep=38*4):
    r = square_sep + square_size/2
    draw_arrow(screen, (255,255,255), center + [r, r + 50], center + [-r, r + 50])

def turn_f_clock(screen, center, square_size=30*4, square_sep=38*4):
    r = square_sep + square_size/2
    draw_arrow(screen, (255,255,255), center + [-r, r + 50], center + [r, r + 50])

def turn_b_cclock(screen, center, square_size=30*4, square_sep=38*4):
    r = square_sep + square_size/2
    draw_arrow(screen, (255,255,255), center + [-r, -r - 50], center + [r, -r - 50])

def turn_b_clock(screen, center, square_size=30*4, square_sep=38*4):
    r = square_sep + square_size/2
    draw_arrow(screen, (255,255,255), center + [r, -r - 50], center + [-r, -r - 50])

def turn_u_cclock(screen, center, square_size=30*4, square_sep=38*4):
    turn_l_clock(screen, center, square_size=square_size, square_sep=square_sep, msg=False)
    turn_f_clock(screen, center, square_size=square_size, square_sep=square_sep)
    turn_r_clock(screen, center, square_size=square_size, square_sep=square_sep, msg=False)
    turn_b_clock(screen, center, square_size=square_size, square_sep=square_sep)
    screen.blit(turn_u_cclock_msg, tuple([50, H-100]))

def turn_u_clock(screen, center, square_size=30*4, square_sep=38*4):
    turn_l_cclock(screen, center, square_size=square_size, square_sep=square_sep, msg=False)
    turn_f_cclock(screen, center, square_size=square_size, square_sep=square_sep)
    turn_r_cclock(screen, center, square_size=square_size, square_sep=square_sep, msg=False)
    turn_b_cclock(screen, center, square_size=square_size, square_sep=square_sep)
    screen.blit(turn_u_clock_msg, tuple([50, H-100]))

def turn_x_clock(screen, center, square_size=30*4, square_sep=38*4):
    turn_l_cclock(screen, center, square_size=square_size, square_sep=square_sep, msg=False)
    turn_r_clock(screen, center, square_size=square_size, square_sep=square_sep, msg=False)
    screen.blit(turn_x_clock_msg, tuple([50, H-100]))

def turn_x_cclock(screen, center, square_size=30*4, square_sep=38*4):
    turn_l_clock(screen, center, square_size=square_size, square_sep=square_sep, msg=False)
    turn_r_cclock(screen, center, square_size=square_size, square_sep=square_sep, msg=False)
    screen.blit(turn_x_cclock_msg, tuple([50, H-100]))

def turn_y_clock(screen, center, square_size=30*4, square_sep=38*4):
    turn_f_clock(screen, center, square_size=square_size, square_sep=square_sep)
    turn_b_cclock(screen, center, square_size=square_size, square_sep=square_sep)
    screen.blit(turn_y_clock_msg, tuple([50, H-100]))

def turn_y_cclock(screen, center, square_size=30*4, square_sep=38*4):
    turn_f_cclock(screen, center, square_size=square_size, square_sep=square_sep)
    turn_b_clock(screen, center, square_size=square_size, square_sep=square_sep)
    screen.blit(turn_y_cclock_msg, tuple([50, H-100]))

while not done:
        for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True

        c = Cube()
        c.set_top_face_matrix(np.array([["R", "G", "B"], ["O", "Y", "W"], ["R", "G", "B"]]))
        draw_cube_net(screen, c, np.array([400,500]))
        draw_cube_face(screen, c.up, np.array([1200,500]), square_size=30*4, square_sep=38*4)

        turn_x_clock(screen, np.array([1200,500]))
        screen.blit(textsurface,(1140,100))
        screen.blit(textsurface2,(340,100))

        screen.blit(top_msg,(375,250))
        screen.blit(bottom_msg,(350,720))
        screen.blit(left_msg,(235,400))
        screen.blit(back_msg,(85,400))
        screen.blit(right_msg,(505,400))


        pygame.display.flip()
