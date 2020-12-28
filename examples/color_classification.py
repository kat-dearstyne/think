import colorsys
import random

from think import Agent, Data, Environment, Memory, Motor, Task, Vision, Chunk, World, Result


class Color:

    def __init__(self, h=0.0, s=0.0, l=0.0):
        self.h = h  # hue
        self.s = s  # saturation
        self.l = l  # lightness
        self.rgb = self.__convert_to_rgb(h, s, l)

    def __convert_to_rgb(self, h, s, l):
        return tuple(round(i * 255) for i in colorsys.hls_to_rgb(h, s / 100, l / 100))

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
    COLORS = {1: Color(s=15, l=64),
              2: Color(s=56, l=74),
              3: Color(s=32, l=59),
              4: Color(s=65, l=65),
              5: Color(s=18, l=51),
              6: Color(s=46, l=51),
              7: Color(s=78, l=48),
              8: Color(s=32, l=42),
              9: Color(s=64, l=42),
              10: Color(s=12, l=29),
              11: Color(s=52, l=26),
              12: Color(s=65, l=26)}
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
                for i in range(self.EXP_FREQ - self.BASE_FREQ):
                    condition_pairs[condition].append(self.PAIRS[exp - 1])
            random.shuffle(condition_pairs[condition])
        return condition_pairs


class ColorClassificationAgent(Agent):
    '''
    KNOWLEDGE = {
        1: [Color(0, 75, 65), Color(22, 30, 62), Color(9, 74, 54), Color(10, 54, 68), Color(31, 100, 48), Color(25, 86, 82),
            Color(19, 45, 36), Color(37, 26, 76), Color(28, 67, 42), Color(30, 69, 80), Color(33, 48, 51),
            Color(359, 25, 63),
            Color(26, 79, 50), Color(15, 90, 51)],
        2: [Color(350, 100, 88), Color(351, 100, 86), Color(330, 59, 100), Color(351, 48, 96), Color(1, 26, 87),
            Color(342, 46, 99), Color(330, 100, 80), Color(342, 30, 100), Color(350, 100, 84), Color(322, 22, 95),
            Color(25, 14, 95), Color(343, 13, 99), Color(0, 20, 96), Color(344, 26, 100)]}
    '''
    KNOWLEDGE = {1: [Color(11, 45, 35)], 2: [Color(350, 100, 88)]}
    KNOWLEDGE_STRENGTH = 1
    SIM_WEIGHTS = [0, 1.17, .83]

    def __init__(self, env, output=False):
        super().__init__(output=output)
        self.memory = Memory(self, Memory.OPTIMIZED_DECAY)
        self.vision = Vision(self, env.display)
        self.motor = Motor(self, self.vision, env)
        self.memory.decay_rate = .5
        self.memory.activation_noise = .5
        self.memory.retrieval_threshold = -1.8
        self.memory.latency_factor = .450
        self.memory.match_scale = 5
        self.__save_knowledge_to_memory()

    def __save_knowledge_to_memory(self):
        for category, colors in self.KNOWLEDGE.items():
            if not isinstance(colors, list):
                colors = [colors]
            for color in colors:
                chunk = Chunk(h=color.h, s=color.s, l=color.l, category=category)
                chunk.use_count = self.KNOWLEDGE_STRENGTH
                self.memory.store(chunk)

    def calculate_similarity_other(self, val1, val2, w=1.0):
        return (-abs(val1 - val2) / 100) * w

    def calculate_similarity_hue(self, hue1, hue2, w=1.0):
        return (-abs(abs(180 - hue1) - abs(180 - hue2)) / 360) * w

    def guess(self):
        return random.randint(1, 2)

    def run(self, time):
        while self.time() < time:
            visual = self.vision.wait_for(isa='color')
            color = self.vision.encode(visual)
            chunk = self.memory.recall(h=color.h, s=color.s, l=color.l,
                                       similarities={
                                           'h': lambda a, b: self.calculate_similarity_hue(a, b, self.SIM_WEIGHTS[0]),
                                           's': lambda a, b: self.calculate_similarity_other(a, b, self.SIM_WEIGHTS[1]),
                                           'l': lambda a, b: self.calculate_similarity_other(a, b, self.SIM_WEIGHTS[2])
                                       })
            if chunk:
                self.motor.type(chunk.get('category'))
            else:
                self.motor.type(self.guess())
            visual = self.vision.wait_for(isa='text')
            category = self.vision.encode(visual)
            self.memory.store(h=color.h, s=color.s, l=color.l, category=category)


class ColorClassificationSimulation():
    HUMAN_CORRECT = {'B': [.318, .123, .513, .113, .175, .337, .13, .162, .372, .097, .143, .272],
                     'E2': [.296, .026, .46, .067, .114, .328, .181, .116, .409, .05, .103, .223],
                     'E7': [.308, .147, .555, .103, .16, .384, .039, .131, .345, .066, .146, .267]}
    TRIALS = {'B': 48, 'E2': 63, 'E7': 63}

    #TRIALS = {'B': 1, 'E2': 1, 'E7': 1}

    def run(self, output=False, real_time=False, print_results=False, show_experiment=True):

        for condition, n_trials in self.TRIALS.items():
            corrects = Data(ColorClassificationTask.N_COLORS)

            for _ in range(n_trials):
                env = Environment(window=(500, 500) if show_experiment else None)
                task = ColorClassificationTask(env, corrects=corrects)
                agent = ColorClassificationAgent(env)
                World(task, agent).run(1590, output=output, real_time=real_time)

            learning_error = Result(corrects.proportion(0), self.HUMAN_CORRECT[condition])

            if print_results:
                learning_error.output("Proportion of Learning Errors for " + condition, 2)


if __name__ == '__main__':
    ColorClassificationSimulation().run(output=False, real_time=False, print_results=True, show_experiment=False)
