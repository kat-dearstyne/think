import colorsys
import math
from sys import float_info


class Item:

    def __init__(self, **slotvals):
        self.slots = slotvals

    def clone(self):
        item = Item()
        for slot, val in self.slots.items():
            item.set(slot, val)
        return item

    def merge(self, item):
        for slot, val in item.slots.items():
            self.set(slot, val)
        return self

    def get_slots(self):
        return self.slots.keys()

    def get(self, slot):
        return self.slots[slot] if slot in self.slots else None

    def __getattr__(self, slot):
        if slot in super().__getattribute__('slots'):
            return self.get(slot)
        else:
            return super().__getattribute__(slot)

    def has(self, slot):
        return slot in self.slots

    def set(self, slot, val):
        self.slots[slot] = val
        return self

    def __setattr__(self, slot, val):
        if 'slots' in self.__dict__ and slot in self.slots:
            self.set(slot, val)
        else:
            super().__setattr__(slot, val)

    def unset(self, slot):
        self.slots.pop(slot, None)
        return self

    def __eq__(self, item):
        return len(self.slots) == len(item.slots) and self.matches(item)

    def equals(self, item):
        return self == item

    def to_dict(self):
        return self.slots

    def matches(self, item):
        for slot, val in item.slots.items():
            if slot not in self.slots or self.slots[slot] != val:
                return False
        return True

    def __str__(self):
        return '{}'.format(self.slots)


class SlotQuery:

    def __init__(self, slot, op, val):
        self.slot = slot
        self.op = op
        self.val = val

    def matches(self, item):
        val = item.get(self.slot)
        if self.op == '=':
            return val == self.val
        elif self.op == '!=':
            return val != self.val
        elif self.op == '>':
            return val > self.val
        elif self.op == '>=':
            return val >= self.val
        elif self.op == '<':
            return val < self.val
        elif self.op == '<=':
            return val <= self.val
        else:
            return False

    def __str__(self):
        return '{}{}{}'.format(self.slot, self.op, self.val)


class Query:

    def __init__(self, **slotvals):
        self.slotqs = []
        for slot, val in slotvals.items():
            self.eq(slot, val)

    def eq(self, slot, val):
        self.slotqs.append(SlotQuery(slot, '=', val))
        return self

    def ne(self, slot, val):
        self.slotqs.append(SlotQuery(slot, '!=', val))
        return self

    def gt(self, slot, val):
        self.slotqs.append(SlotQuery(slot, '>', val))
        return self

    def ge(self, slot, val):
        self.slotqs.append(SlotQuery(slot, '>=', val))
        return self

    def lt(self, slot, val):
        self.slotqs.append(SlotQuery(slot, '<', val))
        return self

    def le(self, slot, val):
        self.slotqs.append(SlotQuery(slot, '<=', val))
        return self

    def get(self, slot, op=None, val=None):
        for slotq in self.slotqs:
            if (slot == slotq.slot and (op is None or op == slotq.op)
                    and (val is None or val == slotq.val)):
                return slotq
        return None

    def has(self, slot, op=None, val=None):
        return self.get(slot, op, val) is not None

    def matches(self, item):
        for slotq in self.slotqs:
            if not slotq.matches(item):
                return False
        return True

    def similarity(self, item, distance_fns):
        dist = 0
        for slotq in self.slotqs:
            slot_dist = 0
            if slotq.slot in distance_fns:
                slot_dist = distance_fns[slotq.slot](
                    slotq.val, item.get(slotq.slot))
            else:
                if not slotq.matches(item):
                    slot_dist = 1
            dist += slot_dist**2
        return self._sim_exp(dist)

    def _sim_exp(self, total_dist, c=1):
        return math.e ** (-c * math.sqrt(total_dist)) - 1

    def __str__(self):
        return '[' + ', '.join(map(str, self.slotqs)) + ']'


class Location(Item):

    def __init__(self, x, y, isa='location'):
        super().__init__(isa=isa, x=x, y=y)

    def move(self, x, y):
        self.x = x
        self.y = y

    def distance_to(self, loc):
        dx = self.x - loc.x
        dy = self.y - loc.y
        return math.sqrt(dx * dx + dy * dy)

    def distance_to_area(self, area):
        dx = max(area.x1 - self.x, self.x - area.x2, 0)
        dy = max(area.y1 - self.y, self.y - area.y2, 0)
        return math.sqrt(dx * dx + dy * dy)

    def angle_to(self, loc):
        if self.distance_to(loc) == 0:
            return 0
        else:
            return math.atan2(loc.y - self.y, loc.x - self.x)


class Area(Location):

    def __init__(self, x, y, w, h, isa='area'):
        super().__init__(x, y, isa)
        self.set('w', w).set('h', h)
        self._reset_corners()

    def _reset_corners(self):
        self.x1 = self.x - self.w / 2
        self.x2 = self.x + self.w / 2
        self.y1 = self.y - self.h / 2
        self.y2 = self.y + self.h / 2

    def resize(self, w, h):
        self.w = w
        self.h = h
        self._reset_corners()

    def contains(self, loc):
        return (self.x1 <= loc.x) and (loc.x <= self.x2) and (self.y1 <= loc.y) and (loc.y <= self.y2)

    def approach_width_from(self, loc):
        theta = loc.angle_to(self)
        critical_theta = math.atan2(self.h, self.w)
        theta = abs(theta)
        if theta > math.pi / 2:
            theta = math.pi - theta
        aw = 0
        if theta == 0:
            aw = self.w
        elif theta == math.pi / 2:
            aw = self.h
        elif theta == critical_theta:
            aw = math.sqrt(self.w * self.w + self.h * self.h)
        elif theta < critical_theta:
            aw = self.w / math.cos(theta)
        else:
            aw = self.h / math.cos(math.pi / 2 - theta)
        return aw


class Color:

    def __init__(self, h=0.0, s=0.0, l=0.0):
        self.h = float(h)  # hue
        self.s = float(s)  # saturation
        self.l = float(l)  # lightness
        self.rgb = self.__convert_to_rgb(h, s, l)

    def __convert_to_rgb(self, h, s, l):
        return tuple(round(i * 255) for i in colorsys.hls_to_rgb(h, s / 100, l / 100))

    def __repr__(self):
        return str((self.h, self.s, self.l))

    def __eq__(self, other):
        if isinstance(other, Color):
            return abs(self.h - other.h) <= float_info.epsilon and abs(self.s - other.s) <= float_info.epsilon and \
                   abs(self.l - other.l) <= float_info.epsilon
        return False


class Face:
    w = 75
    h = 110
    w_scale = 1.8
    h_scale = 1.5
    EH = 0
    ES = 1
    NL = 2
    MH = 3
    features = ['eh', 'es', 'nl', 'mh']

    def __init__(self, eh=23.5, es=21.5, nl=9, mh=16.5):
        self.eh = eh
        self.es = es
        self.nl = nl
        self.mh = mh * 1.5


class Fraction:

    def __init__(self, numerator=0.0, denominator=0.0):
        self.numerator = numerator
        self.denominator = denominator

    def to_float(self):
        return self.numerator / self.denominator if self.denominator != 0 else None

    def to_int(self):
        float_ = self.to_float()
        return round(float_) if float_ is not None else None
