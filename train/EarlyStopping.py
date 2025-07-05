class EarlyStopping:
    def __init__(self, monitor: str, patience: int = 15, min_delta: float = 1e-4):
        self.__monitor = monitor
        self.__patience = patience
        self.__min_delta = min_delta
        self.__best = None
        self.__num_bad_epochs = 0
        self.__should_stop = False

    @property
    def should_stop(self) -> bool:
        return self.__should_stop

    def step(self, current_dict: dict) -> None:
        current = current_dict[self.__monitor]

        if self.__best is None:
            self.__best = current
            return

        improvement = current - self.__best

        if improvement > self.__min_delta:
            self.__best = current
            self.__num_bad_epochs = 0
        else:
            self.__num_bad_epochs += 1
            if self.__num_bad_epochs >= self.__patience:
                self.__should_stop = True