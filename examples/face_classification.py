import random

from examples.classification_common import Condition
from think import Agent, Data, Environment, Memory, Motor, Task, Vision, World, Face, Result
from copy import deepcopy
random.seed(1)


class FaceClassificationTask(Task):
    PHASES = {'Training': 12, 'Transfer': 2}
    FACES = [Face(23.5, 21.5, 13.5, 16.5), Face(19.5, 11.5, 18, 12), Face(23.5, 16.5, 13.5, 12), Face(23.5, 16.5, 18, 12),  # 4
             Face(23.5, 16.5, 18, 7.5), Face(15, 11.5, 9, 16.5), Face(19.5, 16.5, 9, 7.5), Face(15, 11.5, 18, 16.5),  # 8
             Face(15, 11.5, 18, 7.5), Face(15, 11.5, 9, 7.5), Face(19.5, 21.5, 9, 16.5), Face(23.5, 11.5, 9, 7.5),  # 12
             Face(15, 21.5, 13.5, 12), Face(19.5, 16.5, 9, 16.5), Face(19.5, 16.5, 9, 12), Face(15, 16.5, 13.5, 16.5),  # 16
             Face(23.5, 11.5, 9, 16.5), Face(15, 11.5, 13.5, 12), Face(15, 21.5, 18, 16.5), Face(19.5, 11.5, 18, 16.5),  # 20
             Face(19.5, 21.5, 9, 7.5), Face(15, 21.5, 18, 12), Face(19.5, 11.5, 13.5, 16.5), Face(15, 16.5, 13.5, 12),  # 24
             Face(15, 16.5, 18, 16.5), Face(19.5, 21.5, 9, 16.5), Face(15, 16.5, 18, 12), Face(19.5, 16.5, 13.5, 7.5),  # 28
             Face(15, 21.5, 13.5, 16.5), Face(15, 11.5, 13.5, 12), Face(23.5, 11.5, 18, 12), Face(19.5, 11.5, 13.5, 12),  # 32
             Face(22.7, 16.5, 16.2, 12), Face(15.9, 12.5, 12.6, 11.1)]  # 34
    CAT_E1 = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 2]
    CAT_E2 = [2, 2, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 1, 2, 2, 2, 1, 2, 1, 1, 2, 1, 2, 1, 1, 1, 1, 2, 2, 2, 1, 1]

    def __init__(self, env, condition, corrects=None):
        super().__init__()
        self.display = env.display
        self.keyboard = env.keyboard
        self.selected_categories = corrects or Data(len(self.FACES))
        self.done = False
        self.condition = condition

    def run(self, time):

        def is_training():
            return self.phase == 'Training'

        def handle_key(key):
            if not is_training():
                self.selected_categories.add(self.face_num, int(key))

        self.keyboard.add_type_fn(handle_key)

        for phase in self.PHASES.keys():
            self.phase = phase
            for block in range(self.PHASES[phase]):
                face_nums = [i for i in range(0, 34)] if not is_training() else self.condition.train_stim
                random.shuffle(face_nums)
                for num in face_nums:
                    self.face_num = num
                    self.trial_start = self.time()
                    self.display.clear()
                    face_visual = self.display.add_face(50, 50, self.FACES[num], isa='face')
                    self.display.set_attend(face_visual)
                    self.wait(5)
                    self.display.clear()
                    category = self.CAT_E1[num] if self.condition.name[0] == '1' else self.CAT_E2[num]
                    category_visual = self.display.add_text(50, 50, category)
                    self.display.set_attend(category_visual)
                    self.wait(5)
                    self.display.clear()


class FaceClassificationAgent(Agent):
    MAX_MIN = [(23.5, 15), (21.5, 11.5), (18, 9), (16.5, 7.5)]

    def __init__(self, env, condition, output=False):
        super().__init__(output=output)
        self.memory = Memory(self, Memory.OPTIMIZED_DECAY)
        self.vision = Vision(self, env.display)
        self.motor = Motor(self, self.vision, env)
        self.memory.decay_rate = .5
        self.memory.activation_noise = .5
        self.memory.retrieval_threshold = -1.7
        self.memory.latency_factor = .450
        self.memory.match_scale = 5
        self.memory.num_chunks = 5
        self.memory.use_blending = False

        self.memory.use_blending = False
        self.condition = condition

        def _distances(slot1, slot2, max_min, w=1.0):
            return (abs(slot1 - slot2) / (max_min[0] - max_min[1])) * w

        self.memory.add_distance_fn(Face.features[Face.EH],
                                    lambda a, b: _distances(a, b, self.MAX_MIN[Face.EH], self.condition.sim_weights[Face.EH]))
        self.memory.add_distance_fn(Face.features[Face.ES],
                                    lambda a, b: _distances(a, b, self.MAX_MIN[Face.ES], self.condition.sim_weights[Face.ES]))
        self.memory.add_distance_fn(Face.features[Face.MH],
                                    lambda a, b: _distances(a, b, self.MAX_MIN[Face.NL], self.condition.sim_weights[Face.NL]))
        self.memory.add_distance_fn(Face.features[Face.NL],
                                    lambda a, b: _distances(a, b, self.MAX_MIN[Face.MH], self.condition.sim_weights[Face.MH]))

    def guess(self):
        return random.randint(1, 2)

    def run(self, time):
        while self.time() < time:
            visual = self.vision.wait_for(isa='face')
            face = self.vision.encode(visual)
            chunks = self.memory.recall(eh=face.eh, es=face.es, nl=face.nl, mh=face.mh)
            chunks = [chunks] if not isinstance(chunks, list) and chunks else chunks
            if chunks:
                categories = [chunk.get('category') for chunk in chunks]
                selected_category = max(set(categories), key=categories.count)
            else:
                selected_category = self.guess()
            self.motor.type(selected_category)
            visual = self.vision.wait_for(isa='text')
            category = self.vision.encode(visual)
            self.memory.store(eh=face.eh, es=face.es, nl=face.nl, mh=face.mh, category=category)


class FaceCondition(Condition):

    def __init__(self, name, base_stim, exe_num=-1, exe_freq=5, base_freq=1, sim_weights=None, time=1380):
        super().__init__(name, base_stim, exe_num, exe_freq, base_freq, sim_weights)
        self.time = time


class FaceClassificationSimulation:
    TRAINING_STIMULI = {1: [i for i in range(10)], 2: [8, 12, 17, 19, 22, 23, 25, 27, 28, 32]}
    CONDITIONS = [FaceCondition('1EF', TRAINING_STIMULI[1], sim_weights=(.76, .48, 1, 1.76)),
                  FaceCondition('1HF7', TRAINING_STIMULI[1], exe_num=7, sim_weights=(.6, .6, 1.16, 1.64), time=1800),]
                  #FaceCondition('2EF', TRAINING_STIMULI[2], sim_weights=(.432, 1.852, 0, 1.716)),
                  #FaceCondition('2HF19', TRAINING_STIMULI[2], exe_num=19, sim_weights=(.504, 2.344, 0, 1.152), time=1800)]
    HUMAN_CORRECT = {
        '1EF': [.968, .665, .937, .975, .875, .134, .261, .171, .089, .045, .529, .538, .258, .541, .430, .283, .778,
                .145, .281, .780, .291, .277, .601, .289, .272, .490, .261, .386, .269, .114, .796, .538, .981, .126],
        '1HF7': [.956, .594, .931, .944, .775, .1, .1, .15, .069, .006, .4, .488, .275, .425, .238, .269, .675,
                 .094, .325, .681, .125, .269, .45, .219, .275, .413, .275, .144, .294, .094, .806, .300, .931, .05],
        '2EF': [.660, .253, .456, .375, .843, .156, .855, .113, .717, .681, .711, .763, .937, .338, .525, .394, .081,
                .381, .774, .095, .944, .925, .094, .633, .394, .744, .675, .844, .838, .318, .27, .283, .369, .488],
        '2HF19': [.748, .151, .43, .415, .698, .101, .881, .126, .516, .541, .881, .656, .956, .516, .66, .679, .031,
                  .189, .963, .069, .938, .969, .044, .811, .675, .855, .731, .844, .950, .222, .126, .169, .409, .264]}

    def run(self, n_trials=80, output=False, real_time=False, print_results=False, show_experiment=True):

        for condition in self.CONDITIONS:
            selected_categories = Data(len(FaceClassificationTask.FACES))

            for _ in range(n_trials):
                env = Environment(window=(500, 500) if show_experiment else None)
                task = FaceClassificationTask(env, condition, corrects=selected_categories)
                agent = FaceClassificationAgent(env, condition)
                World(task, agent).run(condition.time, output=output, real_time=real_time)

            prob_cat1 = Result(selected_categories.proportion(1), self.HUMAN_CORRECT[condition.name])

            if print_results:
                prob_cat1.output("Probability of Category 1 Selection " + condition.name, 2)


if __name__ == '__main__':
    FaceClassificationSimulation().run(output=False, real_time=False, print_results=True, show_experiment=False)
