import numpy as np


class DataAugmentor():

    compass = {'N': 2,
               'W': 3,
               'S': 4,
               'E': 5}

    def get_batch(self, batch):

        batch__90 = [(self.rot_90(x,y)) for x,y in batch]
        batch_180 = [(self.rot_90(x,y)) for x,y in batch__90]
        batch_270 = [(self.rot_90(x,y)) for x,y in batch_180]

        batch += batch__90 + batch_180 + batch_270
        batch += [(self.flip_v(x,y)) for x,y in batch]

        return batch

    @staticmethod
    def rot_90_degrees(x,y):
        """
        Rotate 90 degrees anticlockwise
        """
        old_directions = [self.compass['N'],
                          self.compass['W'],
                          self.compass['S'],
                          self.compass['E']]

        new_directions = [self.compass['W'],
                          self.compass['S'],
                          self.compass['E'],
                          self.compass['N']]

        x = np.rot90(x, 1)
        y[:,:,old_directions] = y[:,:,new_directions]

        return np.ascontiguousarray(x), np.ascontiguousarray(y)

    @staticmethod
    def flip_v(x, y):
        """
        Flip the map vertically
        """

        old_directions = [self.compass['N'],
                          self.compass['W'],
                          self.compass['S'],
                          self.compass['E']]

        new_directions = [self.compass['N'],
                          self.compass['E'],
                          self.compass['S'],
                          self.compass['W']]

        x = x[:,::-1,:]
        y[:,:,old_directions] = y[:,:,new_directions]

        return np.ascontiguousarray(x), np.ascontiguousarray(y)

    @staticmethod
    def flip_h(x,y):
        """
        Flip the map horizontally
        """
        old_directions = [self.compass['N'],
                          self.compass['W'],
                          self.compass['S'],
                          self.compass['E']]

        new_directions = [self.compass['S'],
                          self.compass['W'],
                          self.compass['N'],
                          self.compass['E']]


        x_h = x[::-1,:,:]
        y[:,:,old_directions] = y[:,:,new_directions]

        return np.ascontiguousarray(x_h), np.ascontiguousarray(y_h)
