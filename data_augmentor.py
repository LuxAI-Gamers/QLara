import numpy as np


class DataAugmentor():

    compass = {'N': 2,
               'W': 3,
               'S': 4,
               'E': 5}

    def get_batch(self, x_batch, y_batch):

        batch = list(zip(x_batch, y_batch))
        batch__90 = [(self.rot_90(x, y)) for x, y in batch]
        batch_180 = [(self.rot_90(x, y)) for x, y in batch__90]
        batch_270 = [(self.rot_90(x, y)) for x, y in batch_180]

        batch += batch__90 + batch_180 + batch_270
        batch += [(self.flip_v(x, y)) for x, y in batch]

        return zip(*batch)

    @classmethod
    def rot_90(cls, x, y):
        """
        Rotate 90 degrees anticlockwise
        """
        old_directions = [cls.compass['N'],
                          cls.compass['W'],
                          cls.compass['S'],
                          cls.compass['E']]

        new_directions = [cls.compass['W'],
                          cls.compass['S'],
                          cls.compass['E'],
                          cls.compass['N']]

        x = np.rot90(x, 1)
        y[:, :, old_directions] = y[:, :, new_directions]

        return np.ascontiguousarray(x), np.ascontiguousarray(y)

    @classmethod
    def flip_v(cls, x, y):
        """
        Flip the map vertically
        """

        old_directions = [cls.compass['N'],
                          cls.compass['W'],
                          cls.compass['S'],
                          cls.compass['E']]

        new_directions = [cls.compass['N'],
                          cls.compass['E'],
                          cls.compass['S'],
                          cls.compass['W']]

        x = x[:, ::-1, :]
        y[:, :, old_directions] = y[:, :, new_directions]

        return np.ascontiguousarray(x), np.ascontiguousarray(y)

    @classmethod
    def flip_h(cls, x, y):
        """
        Flip the map horizontally
        """
        old_directions = [cls.compass['N'],
                          cls.compass['W'],
                          cls.compass['S'],
                          cls.compass['E']]

        new_directions = [cls.compass['S'],
                          cls.compass['W'],
                          cls.compass['N'],
                          cls.compass['E']]

        x_h = x[::-1, :, :]
        y[:, :, old_directions] = y[:, :, new_directions]

        return np.ascontiguousarray(x_h), np.ascontiguousarray(y_h)
