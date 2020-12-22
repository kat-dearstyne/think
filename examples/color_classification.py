import colorsys
import random

from think import Agent, Data, Environment, Memory, Motor, Task, Vision, World, Clock, Result


class Color:

    def __init__(self, h=0, s=0, l=0):
        self.h = h  # hue
        self.s = s  # saturation
        self.l = l  # lightness
        self.rgb = self.__convert_to_rgb(h, s, l)

    def __convert_to_rgb(self, h, s, l):
        return tuple(round(i * 255) for i in colorsys.hls_to_rgb(h, s, l))

    def __repr__(self):
        return str((self.h, self.s, self.l))

    def __eq__(self, other):
        if isinstance(other, Color):
            return self.h == other.h and self.s == other.s and self.l == other.l
        return False


class ColorClassificationTask(Task):
    N_COLORS = 12
    N_BLOCKS = 3
    BASE_FREQ = 4
    EXP_FREQ = 19
    CONDITIONS = {'B': None, 'E2': 2, 'E7': 7}
    COLORS = {1: Color(s=.15, l=.64),
              2: Color(s=.56, l=.74),
              3: Color(s=.32, l=.59),
              4: Color(s=.65, l=.65),
              5: Color(s=.28, l=.52),
              6: Color(s=.46, l=.51),
              7: Color(s=.78, l=.48),
              8: Color(s=.32, l=.42),
              9: Color(s=.64, l=.42),
              10: Color(s=.12, l=.29),
              11: Color(s=.52, l=.26),
              12: Color(s=.65, l=.26)}
    PAIRS = [(1, 1), (2, 2), (3, 2), (4, 2), (5, 1), (6, 2), (7, 2),
             (8, 1), (9, 2), (10, 1), (11, 1), (12, 1)]  # (color, category) pairs

    def __init__(self, env, corrects=None):
        super().__init__()
        self.display = env.display
        self.keyboard = env.keyboard
        self.corrects = corrects or Data(self.N_COLORS)
        self.responded = False
        self.done = False
        self.condition_pairs = self.__create_conditions()
        self.curr_condition = 'B'

    def run(self, time):

        def handle_key(key):
            if str(key) == str(self.trial_category):
                self.log('correct response')
                self.corrects.add(self.color_num, 1)
                self.responded = True

        self.keyboard.add_type_fn(handle_key)

        for block in range(self.N_BLOCKS):
            pairs = self.condition_pairs[self.curr_condition]
            for num, category in pairs:
                self.color_num = num - 1
                color = self.COLORS[num]
                self.trial_start = self.time()
                self.trial_category = category
                self.responded = False
                self.display.clear()
                color_visual = self.display.add_color(50, 50, 15, color, isa='color')
                self.display.set_attend(color_visual)
                self.wait(5.0)
                if not self.responded:
                    self.log('incorrect response')
                    self.corrects.add(self.color_num, 0)
                self.display.clear()
                category_visual = self.display.add_text(50, 50, category)
                self.display.set_attend(category_visual)
                self.wait(5.0)

    def __create_conditions(self):
        pairs = []
        for i in range(self.BASE_FREQ):
            pairs.extend(self.PAIRS.copy())
        condition_pairs = {condition: pairs.copy() for condition in self.CONDITIONS.keys()}
        for condition, exp in self.CONDITIONS.items():
            if exp is not None:
                for i in range(self.EXP_FREQ-self.BASE_FREQ):
                    condition_pairs[condition].append(self.PAIRS[exp - 1])
            random.shuffle(condition_pairs[condition])
        return condition_pairs


class ColorClassificationAgent(Agent):

    def __init__(self, env, output=False):
        super().__init__(output=output)
        self.memory = Memory(self, Memory.OPTIMIZED_DECAY)
        self.vision = Vision(self, env.display)
        self.motor = Motor(self, self.vision, env)
        self.memory.decay_rate = .5
        self.memory.activation_noise = .5
        self.memory.retrieval_threshold = -1.8
        self.memory.latency_factor = .450

    def run(self, time):
        while self.time() < time:
            visual = self.vision.wait_for(isa='color')
            color = self.vision.encode(visual)
            chunk = self.memory.recall(color=color)
            if chunk:
                self.motor.type(chunk.get('category'))
            visual = self.vision.wait_for(isa='text')
            category = self.vision.encode(visual)
            self.memory.store(color=color, category=category)


class ColorClassificationSimulation():
    HUMAN_CORRECT = {'B': [.318, .123, .513, .113, .175, .337, .13, .162, .372, .097, .143, .272],
                     'E2': [.296, .026, .46, .067, .114, .328, .181, .116, .409, .05, .103, .223],
                     'E7': [.308, .147, .555, .103, .16, .384, .039, .131, .345, .066, .146, .267]}
    #TRIALS = {'B': 48, 'E2': 63, 'E7': 63}
    TRIALS = {'B': 1, 'E2': 1, 'E7': 1}

    def run(self, output=False, real_time=False, print_results=False):

        for condition, n_trials in self.TRIALS.items():
            corrects = Data(ColorClassificationTask.N_COLORS)

            for _ in range(n_trials):
                env = Environment(window=(500, 500))
                task = ColorClassificationTask(env, corrects=corrects)
                agent = ColorClassificationAgent(env)
                World(task, agent).run(1590, output=output, real_time=real_time)

            learning_error = Result(corrects.proportion(0), self.HUMAN_CORRECT[condition])

            if print_results:
                learning_error.output("Proportion of Learning Errors for " + condition, 2)


if __name__ == '__main__':
    ColorClassificationSimulation().run(output=False, real_time=False, print_results=True)
