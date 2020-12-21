import colorsys
import random

from think import Agent, Data, Environment, Memory, Motor, Task, Vision, World, Clock


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


class ColorClassificationTask(Task):
    N_BLOCKS = 1
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
        self.corrects = corrects or Data(self.N_BLOCKS)
        self.responded = False
        self.done = False

    def run(self, time):

        def handle_key(key):
            if str(key) == str(self.trial_category):
                self.log('correct response')
                self.corrects.add(self.block, 1)
                self.responded = True

        self.keyboard.add_type_fn(handle_key)

        for block in range(self.N_BLOCKS):
            self.block = block
            pairs = self.PAIRS.copy()
            random.shuffle(pairs)
            for num, category in pairs:
                color = self.COLORS[num]
                self.trial_start = self.time()
                self.trial_category = category
                self.responded = False
                self.display.clear()
                color_visual = self.display.add_color(50, 50, 50, 50, color, isa='color')
                self.display.set_attend(color_visual)
                self.wait(5.0)
                if not self.responded:
                    self.log('incorrect response')
                    self.corrects.add(self.block, 0)
                self.display.clear()
                category_visual = self.display.add_text(50, 50, category)
                self.display.set_attend(category_visual)
                self.wait(5.0)


class ColorClassificationAgent(Agent):

    def __init__(self, env, output=True, real_time=False):
        super().__init__(output=output, clock=Clock(real_time=real_time, output=output))
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
            visual = self.vision.wait_for(isa='category')
            category = self.vision.encode(visual)
            self.memory.store(color=color, category=category)


class ColorClassificationSimulation():
    HUMAN_CORRECT = [.000, .526, .667, .798, .887, .924, .958, .954]

    # HUMAN_CORRECT = [.318, .123, .513, .113, .175, .337, .13, .162, .372]

    def __init__(self, n_sims=10):
        self.n_sims = n_sims

    def run(self, output=False, real_time=False):
        corrects = Data(ColorClassificationTask.N_BLOCKS)

        for _ in range(self.n_sims):
            env = Environment(window=(500, 500))
            task = ColorClassificationTask(env, corrects=corrects)
            agent = ColorClassificationAgent(env, output=output, real_time=real_time)
            World(task, agent).run(1590)

        result_correct = corrects.analyze(self.HUMAN_CORRECT)

        if output:
            result_correct.output("Correctness", 2)

        return result_correct  # (result_correct, result_rt)


if __name__ == '__main__':
    ColorClassificationSimulation().run(output=True, real_time=True)
