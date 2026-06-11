import time


class SessionManager:

    def __init__(self):

        self.start_time = time.time()

        self.waypoints_reached = 0

        self.collisions = 0

    def waypoint_reached(self):

        self.waypoints_reached += 1

    def collision(self):

        self.collisions += 1

    def summary(self):

        return {

            "duration":
            time.time()
            - self.start_time,

            "waypoints":
            self.waypoints_reached,

            "collisions":
            self.collisions
        }
