"""Bounded physical lift target clock; paused intervals never accumulate travel."""
class IncrementalLift:
    def __init__(self, speed_m_s=.012, maximum_m=.08):
        self.speed=float(speed_m_s)
        self.maximum=float(maximum_m)
        self.height=0.
        self.last_time=None

    def start(self, now):
        self.last_time=float(now)

    def advance(self, now, permitted):
        now=float(now)
        if self.last_time is not None:
            dt=now-self.last_time
            if dt < -1e-9:raise ValueError('Lift clock must be monotonic')
            if permitted:self.height=min(self.maximum,self.height+self.speed*max(0.,dt))
        self.last_time=now
        return self.height
