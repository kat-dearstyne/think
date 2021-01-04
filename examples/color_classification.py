import random

from think import Agent, Data, Environment, Memory, Motor, Task, Vision, Chunk, World, Result, Color

random.seed(20)


class Condition:  # TODO where should this go

    def __init__(self, name, base_train_stim, exe_num=-1, exe_freq=19, base_freq=4, sim_weights=(0, 1.17, .83)):
        self.name = name
        self.sim_weights = sim_weights
        self.train_stim = self._create_training_stimuli_nums(base_train_stim, exe_num, exe_freq, base_freq)

    def _create_training_stimuli_nums(self, base_stim, exe_num, exe_freq, base_freq):
        train_stim = []
        for i in range(base_freq):
            train_stim.extend(base_stim.copy())
        if exe_num >= 0:
            train_stim.extend([exe_num for i in range(exe_freq - base_freq)])
        return train_stim


class ColorClassificationTask(Task):
    N_BLOCKS = 3
    COLORS = [Color(s=15, l=64), Color(s=56, l=74), Color(s=32, l=59), Color(s=65, l=65), Color(s=18, l=51),
              Color(s=46, l=51), Color(s=78, l=48), Color(s=32, l=42), Color(s=64, l=42), Color(s=12, l=29),
              Color(s=52, l=26), Color(s=65, l=26)]
    CAT = [1, 2, 2, 2, 1, 2, 2, 1, 2, 1, 1, 1]

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
                self.trial_category = self.CAT[num]
                self.display.clear()
                color_visual = self.display.add_color(50, 50, 15, self.COLORS[num], isa='color')
                self.display.set_attend(color_visual)
                self.wait(5)
                self.display.clear()
                category_visual = self.display.add_text(50, 50, self.trial_category)
                self.display.set_attend(category_visual)


class ColorClassificationAgent(Agent):
    KNOWLEDGE = {1: [Color(11, 45, 35)], 2: [Color(350, 100, 88)]}
    KNOWLEDGE_STRENGTH = 1

    def __init__(self, env, condition, output=False):
        super().__init__(output=output)
        self.memory = Memory(self, Memory.OPTIMIZED_DECAY)
        self.vision = Vision(self, env.display)
        self.motor = Motor(self, self.vision, env)
        self.memory.decay_rate = .5
        self.memory.activation_noise = .5
        self.memory.retrieval_threshold = -1.8
        self.memory.latency_factor = .450
        self.memory.match_scale = 10
        self.condition = condition
        self._save_knowledge_to_memory()

    def _save_knowledge_to_memory(self):
        for category, colors in self.KNOWLEDGE.items():
            if not isinstance(colors, list):
                colors = [colors]
            for color in colors:
                chunk = Chunk(h=color.h, s=color.s, l=color.l, category=category)
                chunk.use_count = self.KNOWLEDGE_STRENGTH
                self.memory.store(chunk)

    def _get_distances(self):
        return {
            'h': lambda a, b: self.calculate_distances_hue(a, b, self.condition.sim_weights[0]),
            's': lambda a, b: self.calculate_distances_other(a, b, self.condition.sim_weights[1]),
            'l': lambda a, b: self.calculate_distances_other(a, b, self.condition.sim_weights[2])
        }

    def calculate_distances_other(self, val1, val2, w=1.0):
        return (abs(val1 - val2) / 100) * w

    def calculate_distances_hue(self, hue1, hue2, w=1.0):
        return (-abs((180 - hue1) - (180 - hue2)) / 360) * w

    def guess_bias(self):
        p = random.random()
        return 1 if p <= .67 else 2

    def guess(self):
        return random.randint(1, 2)

    def run(self, time):
        while self.time() < time:
            visual = self.vision.wait_for(isa='color')
            color = self.vision.encode(visual)
            chunk = self.memory.recall(distances=self._get_distances(), h=color.h,
                                       s=color.s, l=color.l)
            if chunk:
                selected_category = chunk.get('category')
            else:
                selected_category = self.guess_bias()
            self.motor.type(selected_category)
            visual = self.vision.wait_for(isa='text')
            category = self.vision.encode(visual)
            self.memory.store(h=color.h, s=color.s, l=color.l, category=category)


class ColorClassificationSimulation:
    TRAIN_STIM = [i for i in range(len(ColorClassificationTask.COLORS))]
    CONDITIONS = [Condition('B1', TRAIN_STIM), Condition('E2', TRAIN_STIM, exe_num=1), Condition('E7', TRAIN_STIM, exe_num=6),
                  Condition('E6(3)', TRAIN_STIM, exe_num=5, exe_freq=12), Condition('E6(5)', TRAIN_STIM, exe_num=5)]
    HUMAN_CORRECT = {'B1': [.318, .123, .513, .113, .175, .337, .13, .162, .372, .097, .143, .272],
                     'E2': [.296, .026, .46, .067, .114, .328, .181, .116, .409, .05, .103, .223],
                     'E7': [.308, .147, .555, .103, .16, .384, .039, .131, .345, .066, .146, .267],
                     'E6(3)': [.332, .155, .420, .118, .195, .208, .167, .168, .328, .115, .169, .254],
                     'E6(5)': [.288, .141, .415, .119, .19, .123, .157, .167, .336, .087, .108, .182]}

    def run(self, n_trials=50, output=False, real_time=False, print_results=False, show_experiment=True):

        for condition in self.CONDITIONS:
            corrects = Data(len(ColorClassificationTask.COLORS))

            for _ in range(n_trials):
                env = Environment(window=(500, 500) if show_experiment else None)
                task = ColorClassificationTask(env, condition, corrects=corrects)
                agent = ColorClassificationAgent(env, condition)
                World(task, agent).run(1590, output=output, real_time=real_time)  # TODO figure out time

            learning_error = Result(corrects.proportion(0), self.HUMAN_CORRECT[condition.name])

            if print_results:
                learning_error.output("Proportion of Learning Errors for " + condition.name, 2)


if __name__ == '__main__':
    ColorClassificationSimulation().run(output=False, real_time=False, print_results=True, show_experiment=False)
