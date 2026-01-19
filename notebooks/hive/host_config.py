import platform

available_hosts = ["ruche01.cluster", "research-Lambda-Vector"]

XP_savedir = {
    "research-Lambda-Vector": "/data/workspace/vincent/elicit_2",
    "ruche01.cluster": "/gpfs/workdir/auriauvi/honey_2",
}


def get_xp_savedir():
    global XP_savedir
    global available_hosts

    current_host = platform.node()
    try:
        return XP_savedir[current_host]
    except:
        # raise ValueError(f"Running platform {current_host} unknown.")
        return XP_savedir["ruche01.cluster"]