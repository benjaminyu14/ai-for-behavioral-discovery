# Benjamin Yu July 2024
# This code simulates and records bees travelling to their hive in the prescence of a distortion field.
# Reads .txt containing each trajectory's assigned speed; Reads .json containing trajectory's assigned field direction angle & framecount
# Caps each trajectory's framecount to nearest multiple of 7 (can change), writing only those frames to .mp4
# Can add multiple hives to travel to
# Goal: encourage ML model to learn sequential motion rather than other features
# Eliminated following features: speed, orientation, location (partially)

import pygame
import sys
import math
import ast
import cv2
import numpy as np
from collections import defaultdict
field_to_frames = defaultdict(list)
import json

WIDTH, HEIGHT = 960, 720
FPS = (30)  
WHITE = (255, 255, 255)

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Animal Motion Simulation")
clock = pygame.time.Clock()

magnitude = float(sys.argv[1])
hive1 = ast.literal_eval(sys.argv[2])
coordinates_list = []

# read the randomized speeds
speeds_list = np.loadtxt('varying-speeds/speeds_list_1_9.0.txt').tolist()

# reading angle_to_framecount dictionary:
with open('varying-speeds/field_to_frames_7-17_1_9.0.json', 'r') as f:
    f_to_frm = json.load(f)

class start_loc(pygame.sprite.Sprite):
    def __init__(self, position):
        width = 5
        height = 5
        super().__init__()
        self.image = pygame.Surface((width, height))
        self.image.fill((0, 0, 0))   
        self.rect = self.image.get_rect()
        self.rect.center = position
    
    def coords(self) -> tuple:
        return self.rect.center

class Animal(pygame.sprite.Sprite):
    
    def __init__(self, start_loc_position):
        super().__init__()
        self.original_image = pygame.Surface((20, 13), pygame.SRCALPHA)
        pygame.draw.ellipse(self.original_image, (0, 128, 255), (0, 0, 20, 13))
        pygame.draw.circle(self.original_image, (255, 255, 255), (20, 7), 5)
        self.image = self.original_image.copy()
        self.x = float(start_loc_position[0])
        self.y = float(start_loc_position[1])
        self.rect = self.image.get_rect()
        self.rect.center = start_loc_position
        self.speed = 0 # animal speed
        self.orientation = 0  
        self.target_hive_source = None 
        self.start_loc_position = start_loc_position 
        self.is_at_start_loc = True  
        self.mode_go_start_loc = False
        self.collisions = 0
    
    def update(self, hive_sources, start_loc, frame_writer, angle, frm_count, frames, velocity):
        def base_vector(magnitude, getting_hive):
            self.speed = magnitude
            if getting_hive:
                dx = self.target_hive_source.rect.centerx - self.rect.centerx
                dy = self.target_hive_source.rect.centery - self.rect.centery
            else:
                dx = self.start_loc_position[0] - self.rect.centerx
                dy = self.start_loc_position[1] - self.rect.centery
            target_orientation = math.degrees(math.atan2(dy, dx))
            self.orientation = target_orientation

        def distortion_vector(magnitude, angle, getting_hive):
            rad_angle = math.radians(angle)
            x = magnitude * math.cos(rad_angle)
            y = -1 * magnitude * math.sin(rad_angle)
            if getting_hive:
                distance_x = (self.target_hive_source.rect.centerx - self.rect.centerx)
                distance_y = self.target_hive_source.rect.centery - self.rect.centery 
            else:
                distance_x = self.start_loc_position[0] - self.rect.centerx
                distance_y = self.start_loc_position[1] - self.rect.centery
            scale = math.sqrt(distance_x**2 + distance_y**2) 
            return x * min(1, scale / 100), y * min(1, scale / 100) # distortion is leveled off by arbitrary distance value 100
  

        def moveToSource():
            base_vector(velocity, True)
            distort_x, distort_y = distortion_vector(magnitude, angle, True)
            rad_angle = math.radians(self.orientation)
            original_dx = self.speed * math.cos(rad_angle)
            original_dy = self.speed * math.sin(rad_angle)

            # calculate the resultant vector
            resultant_dx = original_dx + distort_x
            resultant_dy = original_dy + distort_y

            # normalize the resultant vector to the original speed (prevents speed from varying between a trajectory's frames)
            resultant_magnitude = math.sqrt(resultant_dx**2 + resultant_dy**2)
            if resultant_magnitude != 0:
                normalized_dx = self.speed * (resultant_dx / resultant_magnitude)
                normalized_dy = self.speed * (resultant_dy / resultant_magnitude)
            else:
                normalized_dx, normalized_dy = 0, 0

            self.x = float(self.x + normalized_dx)
            self.y = float(self.y + normalized_dy)

            self.rect.x = max(0, min(self.x, WIDTH - self.rect.width))  ####
            self.rect.y = max(0, min(self.y, HEIGHT - self.rect.height)) ####

            coordinates_list.append((self.rect.x, self.rect.y))

            # rotates animal to face correct orientation
            self.image = pygame.transform.rotate(self.original_image, -self.orientation)


        def moveTostart_loc():
            pygame.draw.ellipse(self.original_image, (0, 0, 0), (0, 0, 20, 13))
            pygame.draw.circle(self.original_image, (0, 0, 0), (20, 7), 5)
            self.x = start_loc.rect.centerx
            self.y = start_loc.rect.centery 
      
            # clamp to stay in window
            self.rect.x = max(0, min(self.x, WIDTH - self.rect.width))
            self.rect.y = max(0, min(self.y, HEIGHT - self.rect.height))

            # rotates animal to face correct orientation
            self.image = pygame.transform.rotate(self.original_image, -self.orientation)
            
            if pygame.sprite.collide_rect(self, start_loc):
                self.collisions += 1 
                pygame.draw.ellipse(self.original_image, (0, 128, 255), (0, 0, 30, 20))
                pygame.draw.circle(self.original_image, (255, 255, 255), (30, 10), 7)

                if hive_sources:
                    self.mode_go_start_loc = False
                    # set the next target_hive_source (if available)
                    next_source_index = (hive_sources.sprites().index(self.target_hive_source) + 1) % len(hive_sources.sprites())
                    self.target_hive_source = hive_sources.sprites()[next_source_index]
                else:
                    # all hive sources visited, return start_loc
                    self.target_hive_source = None   
        
        if self.is_at_start_loc:
            # if at start_loc, choose the next hive source as the target
            if self.target_hive_source == None:
                self.target_hive_source = hive_sources.sprites()[0]
                self.is_at_start_loc = False

            self.collisions += 1 

        # if the animal has a target_hive_source, move towards it
        if self.target_hive_source and not self.mode_go_start_loc:
            moveToSource()
        
            # account for if n frames are already a multiple of 7, remove the black screen
            if frames == ((frames // 7) * 7):
                frames -= 1

            ## write only the last n frames, where n is a multiple of 7
            if  frm_count >= frames - ((frames // 7 ) * 7) and frm_count < frames:
                frame_data = pygame.image.tostring(screen, 'RGB')
                frame_np = np.frombuffer(frame_data, dtype=np.uint8).reshape((HEIGHT, WIDTH, 3))
                frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
                frame_writer.write(frame_bgr)
            
            # check if the animal has reached the target_hive_source
            if pygame.sprite.collide_rect(self, self.target_hive_source):
                # handle interaction with the hive source (e.g., print a message)
                self.target_hive_source.handle_interaction()
                self.mode_go_start_loc = True
        else:
            moveTostart_loc()

class hiveSource(pygame.sprite.Sprite):
    def __init__(self, position):
        super().__init__()
        self.image = pygame.Surface((30, 30))
        self.image.fill((0, 255, 0))
        self.rect = self.image.get_rect()
        self.rect.center = position

    def handle_interaction(self):
        print("Animal interacts with hive source at", self.rect.center)

def main():

    #y_coords =  [10, 17, 24, 31, 38, 45, 52, 59, 66, 73, 80, 87, 94, 101, 108, 115, 122, 129, 136, 143, 150, 157, 164, 171, 178, 185, 192, 199, 206, 214, 221, 228, 235, 242, 249, 256, 263, 270, 277, 284, 291, 298, 305, 312, 319, 326, 333, 340, 347, 354, 361, 368, 375, 382, 389, 396, 403, 411, 418, 425, 432, 439, 446, 453, 460, 467, 474, 481, 488, 495, 502, 509, 516, 523, 530, 537, 544, 551, 558, 565, 572, 579, 586, 593, 600, 607, 615, 622, 629, 636, 643, 650, 657, 664, 671, 678, 685, 692, 699, 706]
    y_coords = [13, 20, 27, 34, 41, 48, 55, 62, 69, 76, 83, 90, 97, 104, 112, 119, 126, 133, 140, 147, 154, 161, 168, 175, 182, 189, 196, 203, 210, 217, 224, 231, 238, 245, 252, 259, 266, 273, 280, 287, 294, 301, 308, 316, 323, 330, 337, 344, 351, 358, 365, 372, 379, 386, 393, 400, 407, 414, 421, 428, 435, 442, 449, 456, 463, 470, 477, 484, 491, 498, 505, 513, 520, 527, 534, 541, 548, 555, 562, 569, 576, 583, 590, 597, 604, 611, 618, 625, 632, 639, 646, 653, 660, 667, 674, 681, 688, 695, 702, 710]
    #x_coords =  [10, 19, 28, 38, 47, 57, 66, 76, 85, 95, 104, 113, 123, 132, 142, 151, 161, 170, 180, 189, 198, 208, 217, 227, 236, 246, 255, 265, 274, 283, 293, 302, 312, 321, 331, 340, 350, 359, 368, 378, 387, 397, 406, 416, 425, 435, 444, 454, 463, 472, 482, 491, 501, 510, 520, 529, 539, 548, 557, 567, 576, 586, 595, 605, 614, 624, 633, 642, 652, 661, 671, 680, 690, 699, 709, 718, 727, 737, 746, 756, 765, 775, 784, 794, 803, 813, 822, 831, 841, 850, 860, 869, 879, 888, 898, 907, 916, 926, 935, 945]
    x_coords = [14, 24, 33, 43, 52, 61, 71, 80, 90, 99, 109, 118, 128, 137, 146, 156, 165, 175, 184, 194, 203, 213, 222, 232, 241, 250, 260, 269, 279, 288, 298, 307, 317, 326, 335, 345, 354, 364, 373, 383, 392, 402, 411, 420, 430, 439, 449, 458, 468, 477, 487, 496, 505, 515, 524, 534, 543, 553, 562, 572, 581, 591, 600, 609, 619, 628, 638, 647, 657, 666, 676, 685, 694, 704, 713, 723, 732, 742, 751, 761, 770, 779, 789, 798, 808, 817, 827, 836, 846, 855, 864, 874, 883, 893, 902, 912, 921, 931, 940, 950]
    frame_writer = cv2.VideoWriter(f'varying-speeds/2024-01-01 07:17:03.{str(magnitude)}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), FPS, (WIDTH, HEIGHT))
    for i in range(4):
        angle = 0
        coords = []
        if i == 0:
            x = 960
            coords = y_coords
        elif i == 1:
            coords = x_coords
            y = 720
        elif i == 2:
            x = 0
            coords = y_coords
        elif i == 3:
            coords = x_coords
            y = 0
        for j in range(len(coords)): 
            if i == 0 or i == 2:
                start = (x, coords[j])
                if j % 2 == 0:
                    angle = 90
                elif j % 2 == 1:
                    angle = 270
            elif i == 1 or i == 3:
                start = (coords[j], y)
                if j % 2 == 0:
                    angle = 0
                elif j % 2 == 1:
                    angle = 180
            frames = f_to_frm[str(angle)].pop(0)
            velocity = int(speeds_list.pop(0))
            h = start_loc(start)
            animal = Animal(h.coords())
            source_1 = hiveSource(hive1)
            hive_sources = pygame.sprite.Group(source_1)
            
            frm_count = 0

            while animal.collisions <= 1:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                # moves the animal between start_loc and hive sources
                
                animal.update(hive_sources, h, frame_writer, angle, frm_count, frames, velocity)
                frm_count += 1

                screen.fill((0, 0, 0)) 

                # draws everything
                for hive in hive_sources:
                    screen.blit(hive.image, hive.rect)

                screen.blit(animal.image, animal.rect)
                screen.blit(h.image, h.rect)

                pygame.display.flip()
                clock.tick(FPS)

            field_to_frames[angle].append(frm_count - 1)
            animal.collisions = 0
    
    frame_writer.release()

if __name__ == "__main__":
    main()
