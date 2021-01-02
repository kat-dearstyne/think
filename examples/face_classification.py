import random

from think import Agent, Data, Environment, Memory, Motor, Task, Vision, World, Face, Result

random.seed(20)


class Condition:  # TODO where should this go

    def __init__(self, name, train_stim, exe_num=-1, exe_freq=5):
        self.name = name
        self.train_stim = self._create_training_stimuli_nums(train_stim, exe_num, exe_freq)

    def _create_training_stimuli_nums(self, train_stim, exe_num, exe_freq):
        train_stim = train_stim.copy()
        if exe_num >= 0:
            train_stim.extend([exe_num for i in range(exe_freq - 1)])
        return train_stim


class FaceClassificationTask(Task):
    N_BLOCKS = {'Training': 12, 'Transfer': 2}
    FACES = [Face(23.5, 21.5, 13.5, 16.5), Face(19.5, 11.5, 18, 12), Face(23.5, 16.5, 13.5, 12), Face(23.5, 16.5, 18, 12),
             # 4
             Face(23.5, 16.5, 18, 7.5), Face(15, 11.5, 9, 16.5), Face(19.5, 16.5, 9, 7.5), Face(15, 11.5, 18, 16.5),  # 8
             Face(15, 11.5, 18, 7.5), Face(15, 11.5, 9, 7.5), Face(19.5, 21.5, 9, 7.5), Face(23.5, 11.5, 9, 7.5),  # 12
             Face(15, 21.5, 13.5, 12), Face(19.5, 16.5, 9, 16.5), Face(19.5, 16.5, 9, 12), Face(15, 16.5, 13.5, 16.5),  # 16
             Face(23.5, 11.5, 9, 16.5), Face(15, 11.5, 13.5, 12), Face(15, 21.5, 18, 16.5), Face(19.5, 11.5, 18, 16.5),  # 20
             Face(19.5, 21.5, 9, 7.5), Face(15, 21.5, 18, 12), Face(19.5, 11.5, 13.5, 16.5), Face(15, 16.5, 13.5, 12),  # 24
             Face(15, 16.5, 18, 16.5), Face(19.5, 21.5, 9, 16.5), Face(15, 16.5, 18, 12), Face(19.5, 16.5, 13.5, 7.5),  # 28
             Face(15, 21.5, 13.5, 16.5), Face(15, 11.5, 13.5, 12), Face(23.5, 11.5, 18, 12), Face(19.5, 11.5, 13.5, 12),
             # 32
             Face(22.7, 16.5, 16.2, 12), Face(15.9, 12.5, 12.6, 11.1)]  # 34
    CAT = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 2]

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

        for phase in self.N_BLOCKS.keys():
            self.phase = phase
            for block in range(self.N_BLOCKS[phase]):
                face_nums = [i for i in range(len(self.FACES))] if not is_training() else self.condition.train_stim
                random.shuffle(face_nums)
                for num in face_nums:
                    self.face_num = num
                    self.trial_start = self.time()
                    self.display.clear()
                    face_visual = self.display.add_face(50, 50, self.FACES[num], isa='face')
                    self.display.set_attend(face_visual)
                    self.wait(5)
                    self.display.clear()
                    if is_training():
                        category_visual = self.display.add_text(50, 50, self.CAT[num])
                        self.display.set_attend(category_visual)
                    else:
                        self.display.add_text(50, 50, '')


class FaceClassificationAgent(Agent):
    SIM_WEIGHTS = [.19, .12, .25, .45]
    MAX_MIN = [(23.5, 15), (21.5, 11.5), (18, 9), (16.5, 7.5)]  # TODO move somewhere? calculate instead of hard code?

    def __init__(self, env, output=False):
        super().__init__(output=output)
        self.memory = Memory(self, Memory.OPTIMIZED_DECAY)
        self.vision = Vision(self, env.display)
        self.motor = Motor(self, self.vision, env)
        self.memory.decay_rate = .5
        self.memory.activation_noise = .5
        self.memory.retrieval_threshold = -1.8
        self.memory.latency_factor = .450
        self.memory.match_scale = 10

    def _get_similarities(self):
        return {Face.features[i]: lambda a, b: self.calculate_similarity(a, b, self.MAX_MIN[i], self.SIM_WEIGHTS[i])
                for i in range(len(Face.features))}

    def calculate_similarity(self, slot1, slot2, max_min, w=1.0):
        return (abs(slot1 - slot2) / (max_min[0] - max_min[1])) * w

    def guess(self):
        return random.randint(1, 2)

    def run(self, time):
        while self.time() < time:
            visual = self.vision.wait_for(isa='face')
            face = self.vision.encode(visual)
            chunk = self.memory.recall(distances=self._get_similarities(), eh=face.eh, es=face.es, nl=face.nl, mh=face.mh)
            if chunk:
                selected_category = chunk.get('category')
            else:
                selected_category = self.guess()
            self.motor.type(selected_category)
            visual = self.vision.wait_for(isa='text')
            category = self.vision.encode(visual)
            if not category:
                category = selected_category
            self.memory.store(eh=face.eh, es=face.es, nl=face.nl, mh=face.mh, category=category)


class FaceClassificationSimulation:
    TRAINING_STIMULI = {1: [i for i in range(10)], 2: [8, 12, 17, 19, 22, 23, 25, 27, 28, 32]}
    CONDITIONS = [Condition('1EF', TRAINING_STIMULI[1]), Condition('1HF7', TRAINING_STIMULI[1], exe_num=7),
                  Condition('2EF', TRAINING_STIMULI[2]), Condition('2HF19', TRAINING_STIMULI[2], exe_num=19)]
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
                agent = FaceClassificationAgent(env)
                World(task, agent).run(2400, output=output, real_time=real_time)

            prob_cat1 = Result(selected_categories.proportion(1), self.HUMAN_CORRECT[condition.name])
            # TODO is proportion ^^ what you would do to find the prob of selecting category 1?? unsure what it expects...

            if print_results:
                prob_cat1.output("Probability of Category 1 Selection " + condition.name, 2)


if __name__ == '__main__':
    FaceClassificationSimulation().run(output=False, real_time=False, print_results=True, show_experiment=False)
