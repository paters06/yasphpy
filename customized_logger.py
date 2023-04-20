import logging


class CustomizedLogger:
    def __init__(self) -> None:
        self.logger = logging.getLogger('custom_logger')
        self.logger.setLevel(logging.DEBUG)

        fh = logging.FileHandler('sph_simulation.log')
        fh.setLevel(logging.DEBUG)

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        format_log = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        format_console = '%(message)s'
        formatter_1 = logging.Formatter(format_log)
        formatter_2 = logging.Formatter(format_console)
        fh.setFormatter(formatter_1)
        ch.setFormatter(formatter_2)

        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def debug(self, msg):
        self.logger.debug(msg)

    def info(self, msg):
        self.logger.info(msg)


def test():
    custom_log = CustomizedLogger()
    custom_log.debug('Test_01')
    custom_log.info('Test_02')


if __name__ == '__main__':
    test()
