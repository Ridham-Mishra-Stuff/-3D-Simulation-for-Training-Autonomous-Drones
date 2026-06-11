import csv


class SessionLogger:

    def __init__(self):

        self.file = open(
            "mission_results.csv",
            "w",
            newline=""
        )

        self.writer = csv.writer(
            self.file
        )

        self.writer.writerow([
            "Session",
            "Duration",
            "Waypoints",
            "Collisions"
        ])

    def log(
        self,
        session,
        duration,
        waypoints,
        collisions
    ):

        self.writer.writerow([
            session,
            duration,
            waypoints,
            collisions
        ])
