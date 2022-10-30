import time


class Timer:
    def __init__(self):
        self.start = time.time()

    def measure_time(self, custom_logger=None):
        stop = time.time()
        elapsed_time = stop - self.start
        msg_time = "Execution time: {:.2f} s".format(elapsed_time)
        if custom_logger is not None:
            custom_logger.info(msg_time)
        else:
            print(msg_time)


def main():
    timer = Timer()

    for i in range(0, 100):
        a = 2

    timer.measure_time()


if __name__ == '__main__':
    main()
