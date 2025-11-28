from mypy import api

stdout, stderr, ret = api.run(["chopper/", "--strict"])
print(f"mypy returned: {ret}\n{stdout}{stderr}", end="")
