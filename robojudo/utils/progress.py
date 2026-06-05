from tqdm import tqdm


class ProgressBar:
    def __init__(self, tag, total):
        self.tag = tag
        self.continuous = total is None or total <= 0
        self.total = None if self.continuous else total
        if self.continuous:
            self.pbar = tqdm(
                total=None,
                desc=f"{tag} [playing]",
                unit="step",
                colour="magenta",
                ncols=100,
                dynamic_ncols=True,
                mininterval=0.01,
                leave=False,
                ascii=True,
                bar_format="{desc}: {n_fmt}{unit} [{elapsed}]",
            )
        else:
            self.pbar = tqdm(
                total=total,
                desc=tag,
                unit="step",
                colour="magenta",
                ncols=100,
                dynamic_ncols=True,
                mininterval=0.01,
                leave=False,
                ascii=True,
            )

    def update(self, step=1):
        self.pbar.update(step)

    def set(self, n):
        n = round(n, 3)
        if self.continuous:
            delta = int(n - self.pbar.n)
            if delta > 0:
                self.pbar.update(delta)
            return
        if self.total > 0:
            n = max(0, min(n, self.pbar.total))  # clamp
        delta = n - self.pbar.n
        self.pbar.update(delta)

    def close(self):
        self.pbar.close()
        # print("")


if __name__ == "__main__":
    import time

    pbar = ProgressBar("Test", 100)
    pbar2 = ProgressBar("Test alpha", 1)
    for i in range(1, 101):
        pbar.update(1)
        pbar2.set(i / 100)
        time.sleep(0.01)
    pbar.close()
    pbar2.close()
