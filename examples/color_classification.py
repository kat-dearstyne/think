import math
import random

from think import (Agent, Chunk, Color, Data, Environment, Memory, Motor,
                   Result, Task, Vision, World)

from examples.classification_common import Condition

random.seed(0)


class ColorClassificationTask(Task):
    N_BLOCKS = 3

    COLORS = [

        # Extracted from the graph in the paper
        Color(s=14, l=78), Color(s=56, l=98), Color(s=32, l=68),
        Color(s=64, l=81), Color(s=18, l=52), Color(s=48, l=52),
        Color(s=79, l=49), Color(s=31, l=34), Color(s=64, l=34),
        Color(s=12, l=9), Color(s=52, l=2), Color(s=65, l=2),

        # Similar to above, but L/2 - maybe better not to use
        # Color(s=15, l=64), Color(s=56, l=74), Color(s=32, l=59),
        # Color(s=65, l=65), Color(s=18, l=51), Color(s=46, l=51),
        # Color(s=78, l=48), Color(s=32, l=42), Color(s=64, l=42),
        # Color(s=12, l=29), Color(s=52, l=26), Color(s=65, l=26),

        # Tried the Munsell colors, both HSV and HSL - but they're not
        # even close! So we use the colors extracted from the graph.
        # Color(h=353, s=36, l=74),  # Munsell 5R 7/4
        # Color(h=357, s=81, l=77),  # Munsell 5R 7/8
        # Color(h=357, s=40, l=65),  # Munsell 5R 6/6
        # Color(h=358, s=68, l=67),  # Munsell 5R 6/10
        # Color(h=355, s=20, l=53),  # Munsell 5R 5/4
        # Color(h=358, s=40, l=55),  # Munsell 5R 5/8
        # Color(h=356, s=61, l=56),  # Munsell 5R 5/12
        # Color(h=357, s=31, l=44),  # Munsell 5R 4/6
        # Color(h=355, s=48, l=45),  # Munsell 5R 4/10
        # Color(h=19,  s=39, l=29),  # Munsell 5R 3/4
        # Color(h=27,  s=98, l=23),  # Munsell 5R 3/8
        # Color(h=348, s=72, l=32),  # Munsell 5R 3/10

    ]

    CATEGORIES = [1, 2, 2, 2, 1, 2, 2, 1, 2, 1, 1, 1]

    def __init__(self, env, condition, corrects=None):
        super().__init__()
        self.display = env.display
        self.keyboard = env.keyboard
        self.corrects = corrects or Data(len(self.COLORS))
        self.done = False
        self.condition = condition

    def run(self, time):

        def handle_key(key):
            if str(key) == str(self.trial_category):
                self.log('correct response')
                self.corrects.add(self.color_num, 1)
            else:
                self.log('incorrect response')
                self.corrects.add(self.color_num, 0)

        self.keyboard.add_type_fn(handle_key)

        for block in range(self.N_BLOCKS):
            color_nums = self.condition.train_stim
            random.shuffle(color_nums)
            for num in color_nums:
                self.color_num = num
                self.trial_start = self.time()
                self.trial_category = self.CATEGORIES[num]
                # self.display.clear()
                color_visual = self.display.add_color(
                    50, 50, 15, self.COLORS[num], isa='color')
                self.display.set_attend(color_visual)
                self.wait(5)
                self.display.clear()
                category_visual = self.display.add_text(
                    50, 50, self.trial_category)
                self.display.set_attend(category_visual)
                self.wait(5)
                self.display.clear()


class ColorClassificationAgent(Agent):

    def __init__(self, env, condition, output=False):
        super().__init__(output=output)
        self.memory = Memory(self, Memory.ADVANCED_DECAY)
        self.vision = Vision(self, env.display)
        self.motor = Motor(self, self.vision, env)
        self.condition = condition

        self.memory.latency_factor = 1.0
        self.memory.decay_rate = .25  # .5
        self.memory.activation_noise = .5  # .25

        self.memory.retrieval_threshold = -.5  # -1.1

        # self.memory.match_scale = 100.0  # effectively perfect matching
        self.memory.match_scale = 1.0

        self.memory.use_blending = False

        def distance_deg(hue1, hue2):
            d = hue1 - hue2
            return min(abs(d), abs(d+360), abs(d-360)) / 360

        def distance_pct(val1, val2):
            dist = abs(val1 - val2) / 100
            return dist

        self.memory.add_distance_fn('h', distance_deg)
        self.memory.add_distance_fn('s', distance_pct)
        self.memory.add_distance_fn('l', distance_pct)

    def guess(self):
        return random.randint(1, 2)

    def run(self, time):
        while self.time() < time:
            visual = self.vision.wait_for(isa='color')
            color = self.vision.encode(visual)
            chunk = self.memory.recall(h=color.h, s=color.s, l=color.l)
            self.motor.type(chunk.category if chunk else self.guess())
            visual = self.vision.wait_for(isa='text')
            category = self.vision.encode(visual)
            self.memory.store(h=color.h, s=color.s,
                              l=color.l, category=int(category))


class ColorCondition(Condition):

    def __init__(self, name, base_stim, exe_num=-1, exe_freq=19, base_freq=4,
                 # sim_weights=(0, 1.17, .83)
                 sim_weights=(1, 1, 1)
                 ):
        super().__init__(name, base_stim, exe_num, exe_freq, base_freq, sim_weights)


class ColorClassificationSimulation:
    TRAIN_STIM = [i for i in range(len(ColorClassificationTask.COLORS))]
    CONDITIONS = [ColorCondition('B1', TRAIN_STIM),
                  ColorCondition('E2', TRAIN_STIM, exe_num=1),
                  ColorCondition('E7', TRAIN_STIM, exe_num=6),
                  ColorCondition('E6(3)', TRAIN_STIM, exe_num=5, exe_freq=12),
                  ColorCondition('E6(5)', TRAIN_STIM, exe_num=5)]
    HUMAN_CORRECT = {'B1': [.318, .123, .513, .113, .175, .337, .13, .162, .372, .097, .143, .272],
                     'E2': [.296, .026, .46, .067, .114, .328, .181, .116, .409, .05, .103, .223],
                     'E7': [.308, .147, .555, .103, .16, .384, .039, .131, .345, .066, .146, .267],
                     'E6(3)': [.332, .155, .420, .118, .195, .208, .167, .168, .328, .115, .169, .254],
                     'E6(5)': [.288, .141, .415, .119, .19, .123, .157, .167, .336, .087, .108, .182]}

    def run(self, n_runs=50, output=False, real_time=False, print_results=False, show_experiment=True):

        for condition in self.CONDITIONS:
            print(f'\nRunning condition {condition.name}...')
            corrects = Data(len(ColorClassificationTask.COLORS))

            for _ in range(n_runs):
                env = Environment(window=(500, 500)
                                  if show_experiment else None)
                task = ColorClassificationTask(
                    env, condition, corrects=corrects)
                agent = ColorClassificationAgent(env, condition)
                World(task, agent).run(720, output=output, real_time=real_time)

            learning_error = Result(corrects.proportion(0),
                                    self.HUMAN_CORRECT[condition.name])

            if print_results:
                learning_error.output(
                    "Proportion of Learning Errors for " + condition.name, 2)
            # break


if __name__ == '__main__':
    # ColorClassificationSimulation().run(output=True,
    #                                     real_time=True,
    #                                     n_runs=20,
    #                                     print_results=True, show_experiment=False)
    ColorClassificationSimulation().run(output=False,
                                        real_time=False,
                                        n_runs=30,
                                        print_results=True, show_experiment=False)
