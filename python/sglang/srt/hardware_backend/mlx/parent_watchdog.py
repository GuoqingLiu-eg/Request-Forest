import os
import select
import signal
import threading


def start_parent_death_watcher() -> None:

    original_ppid = os.getppid()

    def _watch_parent():
        kq = select.kqueue()
        kev = select.kevent(
            original_ppid,
            filter=select.KQ_FILTER_PROC,
            flags=select.KQ_EV_ADD,
            fflags=select.KQ_NOTE_EXIT,
        )
        try:

            kq.control([kev], 0, None)
        except (ProcessLookupError, OSError):


            os.kill(os.getpid(), signal.SIGKILL)
            return


        if os.getppid() != original_ppid:
            os.kill(os.getpid(), signal.SIGKILL)
            return

        kq.control(None, 1, None)
        os.kill(os.getpid(), signal.SIGKILL)

    watcher = threading.Thread(
        target=_watch_parent,
        name="parent-death-watcher",
        daemon=True,
    )
    watcher.start()
