import math
import random

from think import Buffer, Item, Module, Query, Fraction


class Chunk(Item):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.id = self.get('id') or self.get(
            'name') or self.get('isa') or 'chunk'
        self.creation_time = None
        self.activation = -math.log(.100)
        self.transient_activation = None
        self.use_count = 0
        self.uses = []

    def increment_use(self):
        self.use_count += 1

    def add_use(self, time):
        self.uses.append(time)

    def __str__(self):
        return '<{}>{}'.format(self.id, self.slots)


class Memory(Module):
    NO_DECAY = 1
    OPTIMIZED_DECAY = 2
    ADVANCED_DECAY = 3

    def __init__(self, agent, decay=None):
        super().__init__('memory', agent)
        self.decay = decay or Memory.NO_DECAY
        self.chunks = {}
        self.buffer = Buffer('memory', self)
        self.decay_rate = 0.5
        self.retrieval_threshold = 0.0
        self.latency_factor = 1.0
        self.activation_noise = None
        self.match_scale = None
        self.use_blending = False
        self.num_chunks = 1
        self.distance_fns = {}
        self._unique = 1

    def _uniquify(self, id):
        if id in self.chunks:
            self._unique += 1
            return self._uniquify('{}~{}'.format(id, self._unique))
        else:
            return id

    def add_distance_fn(self, slot, fn):
        self.distance_fns[slot] = fn
        return self

    def add(self, chunk=None, **kwargs):
        if not chunk:
            chunk = Chunk(**kwargs)
        chunk.id = self._uniquify(chunk.id)
        self.chunks[chunk.id] = chunk
        return chunk

    def get(self, id):
        return self.chunks[id]

    def _add_use(self, chunks):
        if not isinstance(chunks, list):
            chunks = [chunks]
        for chunk in chunks:
            if self.decay == Memory.OPTIMIZED_DECAY:
                chunk.increment_use()
            elif self.decay == Memory.ADVANCED_DECAY:
                chunk.add_use(self.time())

    def _get_match(self, chunk):
        for existing in self.chunks.values():
            if existing.equals(chunk):
                return existing
        return None

    def store(self, chunk=None, boost=None, **kwargs):
        if not chunk:
            chunk = Chunk(**kwargs)
        self.think('store {}'.format(chunk))
        match = self._get_match(chunk)
        if match is not None:
            self.log('stored and merged into {}'.format(match))
            self._add_use(match)
            chunk = match
        else:
            self.log('stored {}'.format(chunk))
            chunk.id = self._uniquify(chunk.id)
            chunk.creation_time = self.time()
            self._add_use(chunk)
            self.add(chunk)
        if boost is not None:
            for _ in range(boost):
                self._add_use(chunk)
            self.log('boosted {} times'.format(boost))
        self._compute_activation(chunk)
        return chunk

    def _compute_activation(self, chunk):
        if self.decay == Memory.NO_DECAY:
            return chunk.activation
        else:
            time = self.time()
            if time <= chunk.creation_time:
                time = chunk.creation_time + .001
            base_level = 0
            if self.decay == Memory.OPTIMIZED_DECAY:
                base_level = (math.log(chunk.use_count / (1 - self.decay_rate))
                              - self.decay_rate * math.log(time - chunk.creation_time))
            elif self.decay == Memory.ADVANCED_DECAY:
                uses = 0
                for use in chunk.uses:
                    uses += math.pow(time - use, -self.decay_rate) if time - use != 0 else 0
                base_level = math.log(uses)
            chunk.activation = base_level
            return chunk.activation

    def _compute_transient_act(self, chunk):
        act = self._compute_activation(chunk)
        if self.activation_noise is not None:
            act += random.gauss(0, self.activation_noise)
        chunk.transient_activation = act
        return chunk.transient_activation

    def _calculate_sim_val(self, act, sim):
        match_scale = math.exp(act) * self.match_scale
        return match_scale * sim

    def compute_prob_recall(self, chunk):
        if self.activation_noise is not None:
            act = self._compute_activation(chunk)
            exp = -(act - self.retrieval_threshold) / self.activation_noise
            return 1 / (1 + math.exp(exp))
        else:
            return None

    def compute_recall_time(self, chunk):
        return self.latency_factor * math.exp(min(-chunk.activation, self.retrieval_threshold))

    def _get_best_chunks(self, query=None):
        matches = (self._get_query_matches(query)
                   if not self.match_scale
                   else self.chunks.values())
        best_chunks = []
        best_act = []
        for chunk in matches:
            act = self._compute_transient_act(chunk)
            if self.match_scale:
                act += self._calculate_sim_val(act, query.similarity(chunk, self.distance_fns))
            if act > self.retrieval_threshold:
                if len(best_chunks) < self.num_chunks:
                    best_chunks.append(chunk)
                    best_act.append(act)
                elif best_act[-1] < act:
                    best_chunks[-1] = chunk
                    best_act[-1] = act
                best_act, best_chunks = (list(t) for t in zip(*sorted(zip(best_act, best_chunks))))
        return best_chunks if best_chunks else None

    def _get_chunks(self, query):
        if self.use_blending:
            chunks = self._get_blended_chunk(query)
        else:
            chunks = self._get_best_chunks(query)
        return chunks

    def _get_blended_chunk(self, query):
        matches = (self._get_query_matches(query)
                   if not self.match_scale
                   else self.chunks.values())
        if not matches:
            return None
        blended_slots = {}
        is_cont = {slot: self._is_continuous(val) for slot, val in blended_slots.items()}
        blended_chunk = None
        best_chunk = None
        for chunk in matches:
            act = self._compute_transient_act(chunk)
            if self.match_scale:
                act += self._calculate_sim_val(act, query.similarity(chunk, self.distance_fns))
            if act > self.retrieval_threshold:
                best_chunk = best_chunk if best_chunk and (best_chunk.transient_activation > act) else chunk
                for slot in chunk.get_slots():
                    if slot not in blended_slots:
                        blended_slots[slot] = Fraction()
                        is_cont[slot] = False
                    if self._is_continuous(chunk.get(slot)):
                        is_cont[slot] = True
                    act += self.retrieval_threshold
                    blended_slots[slot].numerator += act * chunk.get(slot)
                    blended_slots[slot].denominator += act
        if best_chunk:
            for slot, val in blended_slots.items():
                if isinstance(val, Fraction):
                    blended_slots[slot] = val.to_float() if is_cont[slot] else val.to_int()
            blended_chunk = Chunk(**blended_slots)
            blended_chunk.transient_activation = best_chunk.transient_activation
            blended_chunk.use_count = best_chunk.use_count
            blended_chunk.uses = best_chunk.uses
        return blended_chunk

    def _is_continuous(self, val):
        return isinstance(val, float)

    def _get_query_matches(self, query):
        matches = []
        for chunk in self.chunks.values():
            if query.matches(chunk):
                matches.append(chunk)
        return matches

    def start_recall(self, query=None, **kwargs):
        if not query or not isinstance(query, Query):
            query = Query(**kwargs)
        self.buffer.acquire()
        self.think('recall {}'.format(query))
        self.log('recalling {}'.format(query))
        chunks = self._get_chunks(query)
        self._start_recall(chunks)

    def start_recall_by_id(self, id):
        self.buffer.acquire()
        self.think('recall <{}>'.format(id))
        self.log('recalling <{}>'.format(id))
        chunk = self.get(id)
        act = self._compute_transient_act(chunk)
        self._start_recall(chunk if act >= self.retrieval_threshold else None)

    def _start_recall(self, chunks):
        if chunks is not None:
            chunks = chunks if isinstance(chunks, list) else [chunks]
            duration = 0
            for chunk in chunks:
                duration += self.latency_factor * \
                            math.exp(-chunk.transient_activation)
            duration = duration / len(chunks)
            chunks = chunks if len(chunks) > 1 else chunks.pop()
            self.buffer.set(chunks, duration, 'recalled {}'.format(
                chunks), lambda: self._add_use(chunks))
        else:
            duration = self.latency_factor * \
                       math.exp(-self.retrieval_threshold)
            self.buffer.clear(duration, 'recall failed')

    def get_recalled(self):
        return self.buffer.get_and_release()

    def recall(self, query=None, **kwargs):
        if not query or not isinstance(query, Query):
            query = Query(**kwargs)
        self.start_recall(query)
        return self.get_recalled()

    def recall_by_id(self, id):
        self.start_recall_by_id(id)
        return self.get_recalled()

    def rehearse(self, chunk):
        self.recall_by_id(chunk.id)

    def clear(self):
        self.chunks = {}
