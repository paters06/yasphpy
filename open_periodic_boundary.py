class OpenPeriodicBoundary:
    def __init__(self, inlet_x: float, outlet_x: float,
                 down_wall: float, top_wall: float) -> None:
        self.inlet_x = inlet_x
        self.outlet_x = outlet_x
        self.down_wall = down_wall
        self.top_wall = top_wall

    def check_particle_in_domain(self, old_pos, old_vel):
        new_pos = old_pos.copy()
        new_vel = old_vel.copy()
        if old_pos[0] > self.outlet_x:
            new_pos[0] = self.inlet_x

        if old_pos[0] < self.inlet_x:
            new_pos[0] = self.inlet_x
            new_vel[0] = -old_vel[0]

        if old_pos[1] < self.down_wall:
            new_pos[1] = self.down_wall
            new_vel[1] = -old_vel[1]

        if old_pos[1] > self.top_wall:
            new_pos[1] = self.top_wall
            new_vel[1] = -old_vel[1]

        return new_pos, new_vel
