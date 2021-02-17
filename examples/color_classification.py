import random

from examples.classification_common import Condition
from think import Agent, Data, Environment, Memory, Motor, Task, Vision, Chunk, World, Result, Color

random.seed(0)


class ColorClassificationTask(Task):
    N_BLOCKS = 3
    COLORS = [Color(s=15, l=64), Color(s=56, l=74), Color(s=32, l=59), Color(s=65, l=65), Color(s=18, l=51),
              Color(s=46, l=51), Color(s=78, l=48), Color(s=32, l=42), Color(s=64, l=42), Color(s=12, l=29),
              Color(s=52, l=26), Color(s=65, l=26)]
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
                color_visual = self.display.add_color(50, 50, 15, self.COLORS[num], isa='color')
                self.display.set_attend(color_visual)
                self.wait(5)
                self.display.clear()
                category_visual = self.display.add_text(50, 50, self.trial_category)
                self.display.set_attend(category_visual)
                self.wait(5)
                self.display.clear()


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
        self.memory.match_scale = 5
        self.memory.num_chunks = 3
        self.memory.use_blending = False
        self.condition = condition

        self._save_knowledge_to_memory()

        def distance_deg(hue1, hue2, w=0.0):
            d = hue1 - hue2
            return (min(abs(d), abs(d + 360), abs(d - 360)) / 360) * w

        def distance_pct(val1, val2, w=1.0):
            return (abs(val1 - val2) / 100) * w

        self.memory.add_distance_fn('h', lambda a, b: distance_deg(a, b, condition.sim_weights[0]))
        self.memory.add_distance_fn('s', lambda a, b: distance_pct(a, b, condition.sim_weights[1]))
        self.memory.add_distance_fn('l', lambda a, b: distance_pct(a, b, condition.sim_weights[2]))

    def _save_knowledge_to_memory(self):
        for category, colors in self.KNOWLEDGE.items():
            if not isinstance(colors, list):
                colors = [colors]
            for color in colors:
                chunk = Chunk(h=color.h, s=color.s, l=color.l, category=category)
                chunk.use_count = self.KNOWLEDGE_STRENGTH
                self.memory.store(chunk)

    def guess_bias(self):
        p = random.random()
        return 1 if p <= .67 else 2

    def guess(self):
        return random.randint(1, 2)

    def run(self, time):
        while self.time() < time:
            visual = self.vision.wait_for(isa='color')
            color = self.vision.encode(visual)
            chunks = self.memory.recall(h=color.h, s=color.s, l=color.l)
            chunks = [chunks] if not isinstance(chunks, list) and chunks else chunks
            if chunks:
                categories = [chunk.get('category') for chunk in chunks]
                selected_category = max(set(categories), key=categories.count)
            else:
                selected_category = self.guess_bias()
            self.motor.type(selected_category)
            visual = self.vision.wait_for(isa='text')
            category = self.vision.encode(visual)
            self.memory.store(h=color.h, s=color.s, l=color.l, category=int(category))


class ColorCondition(Condition):

    def __init__(self, name, base_stim, exe_num=-1, exe_freq=19, base_freq=4, sim_weights=(0, .89, 1.11)):
        super().__init__(name, base_stim, exe_num, exe_freq, base_freq, sim_weights)


class ColorClassificationSimulation:
    TRAIN_STIM = [i for i in range(len(ColorClassificationTask.COLORS))]
    CONDITIONS = [ColorCondition('B1', TRAIN_STIM), ColorCondition('E2', TRAIN_STIM, exe_num=1),
                  ColorCondition('E7', TRAIN_STIM, exe_num=6)]
    HUMAN_CORRECT = {'B1': [.318, .123, .513, .113, .175, .337, .13, .162, .372, .097, .143, .272],
                     'E2': [.296, .026, .46, .067, .114, .328, .181, .116, .409, .05, .103, .223],
                     'E7': [.308, .147, .555, .103, .16, .384, .039, .131, .345, .066, .146, .267]}

    def run(self, n_trials=50, output=False, real_time=False, print_results=False, show_experiment=True):

        for condition in self.CONDITIONS:
            corrects = Data(len(ColorClassificationTask.COLORS))

            for _ in range(n_trials):
                env = Environment(window=(500, 500) if show_experiment else None)
                task = ColorClassificationTask(env, condition, corrects=corrects)
                agent = ColorClassificationAgent(env, condition)
                World(task, agent).run(720, output=output, real_time=real_time)

            learning_error = Result(corrects.proportion(0), self.HUMAN_CORRECT[condition.name])

            if print_results:
                learning_error.output("Proportion of Learning Errors for " + condition.name, 2)


if __name__ == '__main__':
    ColorClassificationSimulation().run(output=False, real_time=False, print_results=True, show_experiment=False)
