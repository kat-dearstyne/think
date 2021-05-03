class Condition:

    def __init__(self, name, base_stim, exe_num, exe_freq, base_freq, sim_weights):
        self.name = name
        self.sim_weights = sim_weights
        self.train_stim = self._create_training_stimuli_nums(base_stim, exe_num, exe_freq, base_freq)

    def _create_training_stimuli_nums(self, base_stim, exe_num, exe_freq, base_freq):
        train_stim = []
        for i in range(base_freq):
            train_stim.extend(base_stim.copy())
        if exe_num >= 0:
            train_stim.extend([exe_num for i in range(exe_freq - base_freq)])
        return train_stim


class Util:

    @staticmethod
    def get_top_category(chunks):
        chunks = [chunks] if not isinstance(chunks, list) and chunks else chunks
        if chunks:
            categories = [chunk.get('category') for chunk in chunks]
            selected_category = max(set(categories), key=categories.count)
            return selected_category
        return None
