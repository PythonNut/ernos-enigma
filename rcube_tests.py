import numpy as np
import cv2
import kociemba
import threading
import time
import socket
import sys

HOST = 'localhost'	# Symbolic name, meaning all available interfaces
PORT = 9999	# Arbitrary non-privileged port

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print('Socket created')

#Bind socket to local host and port
try:
	s.bind((HOST, PORT))
except socket.error as msg:
	print('Bind failed. Error Code : ' + str(msg[0]) + ' Message ' + msg[1])
	sys.exit()

print('Socket bind complete')

#Start listening on socket
s.listen(10)
print('Socket now listening')

#now keep talking with the client
try:
    while 1:
        #wait to accept a connection - blocking call
        conn, addr = s.accept()
        print('Connected with ' + addr[0] + ':' + str(addr[1]))
finally:
    s.close()

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

def translate_koc_string(string):
        moves = string.split(' ')
        out = ""
        for move in moves:
            if len(move) == 1:
                out += move
            else:
                if move[1] == "'":
                    out += move[0].lower()
                elif move[1] == "2":
                    out += move[0].lower() * 2
        return out

def apply_x_clock_to_moves(string):
    return string.translate(str.maketrans("dfubDFUB","fubdFUBD"))

def apply_x_counter_to_moves(string):
    return string.translate(str.maketrans("dfubDFUB","bdfuBDFU"))

def apply_z_clock_to_moves(string):
    return string.translate(str.maketrans("urdlURDL","rdluRDLU"))

def apply_z_counter_to_moves(string):
    return string.translate(str.maketrans("urdlURDL","lurdLURD"))

def convert_moves_to_nice_moves(string):
    out = ""
    while len(string) > 0:
        if string[0] == "d":
            out += "Z"
            string = apply_z_clock_to_moves(string)
        elif string[0] == "D":
            out += "z"
            string = apply_z_counter_to_moves(string)
        elif string[0] == "f" or string[0] == "F":
            out += "X"
            string = apply_x_clock_to_moves(string)
        elif string[0] == "b" or string[0] == "B":
            out += "x"
            string = apply_x_counter_to_moves(string)
        else:
            out += string[0]
            string = string[1:]
    return out

most_recent_data = ""

def master_looper():
    cube = Cube()
    cube_is_valid = False
    learning_cube = False
    data = [0]*9
    i = 0

    stuff = [[1]*9,[2]*9,[3]*9,[4]*9,[5]*9,[6]*9]

    moves_to_make = "ZXZXZ"

    while True:

       # while data == most_recent_data:
       #     time.sleep(.1)
       # data = most_recent_data
       # if i == 6:
           # break
        data = [int(x) for x in input().split()]

        if not cube_is_valid and not learning_cube:
            cube.clear()
            moves_to_make = "ZXZXZ"
            learning_cube = True

        if learning_cube:
            cube.set_top_face_list(data)
        elif not cube.check_top_face_list(data):
            cube_is_valid = False

        if len(moves_to_make) == 0:
            if cube.check_solved():
                break
            #handle solved
            if not cube_is_valid:
                try:
                    sol = cube.generate_solution()
                except:
                    cube_is_valid = False
                else:
                    moves_to_make = convert_moves_to_nice_moves(translate_koc_string(sol))
                    cube_is_valid = True
                    learning_cube = False
                    cube.execute_move(moves_to_make[0])
                    print(moves_to_make[0])
                    moves_to_make = moves_to_make[1:]
        else:
            cube.execute_move(moves_to_make[0])
            print(moves_to_make[0])
            moves_to_make = moves_to_make[1:]

#threading.Thread(target = master_looper).start()
