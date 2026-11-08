import logging
import os
import signal
import subprocess
import sys
import time

POLL_SECONDS = 3
SUSTAINED_LIMIT = 15
PRESSURE_CMD = ["sysctl", "-n", "kern.memorystatus_vm_pressure_level"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("system-memory-watchdog")


def read_pressure_level() -> int:
    out = subprocess.run(
        PRESSURE_CMD, capture_output=True, text=True, timeout=10
    )
    return int(out.stdout.strip())


def kill_targets(pids: list[int]) -> None:
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
            logger.info("killed pid %d", pid)
        except ProcessLookupError:
            logger.info("pid %d already gone", pid)


def main() -> None:
    pids = [int(p) for p in sys.argv[1:] if p.isdigit()]
    logger.info("watchdog armed: pids=%s poll=%ds sustained_limit=%d", pids, POLL_SECONDS, SUSTAINED_LIMIT)
    sustained = 0
    while True:
        try:
            level = read_pressure_level()
        except Exception as exc:
            logger.warning("pressure read failed: %s", exc)
            level = 0
        if level >= 3:
            sustained += 1
            logger.warning("pressure level %d sustained %d/%d", level, sustained, SUSTAINED_LIMIT)
            if sustained >= SUSTAINED_LIMIT:
                logger.error("system memory pressure critical for %d polls - killing run pids %s", sustained, pids)
                kill_targets(pids)
                sys.exit(2)
        else:
            if sustained:
                logger.info("pressure back to level %d (streak reset)", level)
            sustained = 0
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
