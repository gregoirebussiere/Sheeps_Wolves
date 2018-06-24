'''
Generalized behavior for random walking, one grid cell at a time.
'''
import random

from mesa import Agent


class RandomWalker(Agent):
    '''
    Class implementing random walker methods in a generalized manner.

    Not indended to be used on its own, but to inherit its methods to multiple
    other agents.

    '''

    grid = None
    x = None
    y = None
    moore = True

    def __init__(self, unique_id, pos, model, moore=True):
        '''
        grid: The MultiGrid object in which the agent lives.
        x: The agent's current x coordinate
        y: The agent's current y coordinate
        moore: If True, may move in all 8 directions.
                Otherwise, only up, down, left, right.
        '''
        super().__init__(unique_id, model)
        self.pos = pos
        self.moore = moore

    def random_move(self,compGreg):
        '''
        Step one cell in any allowable direction.
        '''
        # Pick the next cell from the adjacent cells.


        if random.random()>compGreg:
            next_moves = self.model.grid.get_neighborhood(self.pos, self.moore, True)
            next_move = random.choice(next_moves)
            self.model.grid.move_agent(self, next_move)

        else:
            next_moves = self.model.grid.get_neighborhood(self.pos, self.moore, True)     
            for i in (next_moves):
                for k in self.model.grid[i[0]][i[1]]:
                    if isinstance(k,type(self)):
                        return
                next_move = random.choice(next_moves)
                self.model.grid.move_agent(self, next_move)
        # Now move:
        
